import argparse
import logging
import os
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union, Any

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchaudio
import torchaudio.transforms as T
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from transformers import (
    HfArgumentParser,
    EarlyStoppingCallback,
    set_seed,
    TrainerCallback,
    Trainer,
    PretrainedConfig
)
from transformers import TrainingArguments as HfTrainingArguments
from datasets import load_dataset, DatasetDict, VerificationMode, Audio
import datasets

# Chatterbox imports
from chatterbox.tts import ChatterboxTTS # To load the full model initially
from chatterbox.models.s3gen import S3Gen, S3Token2Mel, S3GEN_SR, mel_spectrogram # S3GEN_SR for mels
from chatterbox.models.s3tokenizer import S3Tokenizer, S3_SR as S3_TOKENIZER_SR # S3Tokenizer operates at 16kHz
from chatterbox.models.flow import CausalMaskedDiffWithXvec # The actual module we want to finetune
from chatterbox.models.xvector import CAMPPlus # Speaker encoder used by S3Gen

logger = logging.getLogger(__name__)

# --- Training Arguments (can reuse CustomTrainingArguments) ---
@dataclass
class CustomTrainingArguments(HfTrainingArguments):
    early_stopping_patience: Optional[int] = field(
        default=None, metadata={"help": "Enable early stopping."}
    )

# --- Model Arguments ---
@dataclass
class S3GenModelArguments:
    model_name_or_path: Optional[str] = field(default=None, metadata={"help": "Path to base Chatterbox model"})
    local_model_dir: Optional[str] = field(default=None, metadata={"help": "Path to local Chatterbox model directory"})
    cache_dir: Optional[str] = field(default=None, metadata={"help": "HF cache directory"})
    # S3Gen specific finetuning args
    freeze_speaker_encoder: bool = field(default=True, metadata={"help": "Freeze S3Gen's internal speaker encoder (CAMPPlus)."})
    freeze_s3_tokenizer: bool = field(default=True, metadata={"help": "Freeze S3Gen's internal S3Tokenizer."})
    # The 'flow' part of S3Gen will be trained. HiFiGAN part will be frozen.

# --- Data Arguments ---
@dataclass
class S3GenDataArguments:
    dataset_name: Optional[str] = field(default=None, metadata={"help": "HF Dataset name"})
    dataset_config_name: Optional[str] = field(default=None, metadata={"help": "HF Dataset config name"})
    train_split_name: str = field(default="train", metadata={"help": "Train split name"})
    eval_split_name: Optional[str] = field(default="validation", metadata={"help": "Eval split name"})
    audio_column_name: str = field(default="audio", metadata={"help": "Audio column in dataset"})
    # No text column needed for S3Gen training directly, but might be in dataset
    
    max_speech_token_len: int = field(default=750, metadata={"help": "Max S3 speech tokens for target sequence."})
    max_mel_len: int = field(default=1500, metadata={"help": "Max mel frames for target mel (max_speech_token_len * 2)"})
    # For CausalMaskedDiffWithXvec, we need prompt_tokens and prompt_feats for conditioning
    prompt_audio_duration_s: float = field(default=3.0, metadata={"help": "Duration of audio for prompt_token and prompt_feat."})
    
    eval_split_size: float = field(default=0.01, metadata={"help": "Eval split fraction"}) # Increased default
    ignore_verifications: bool = field(default=False, metadata={"help":"Ignore dataset verifications."})


# --- S3Gen Finetuning Dataset ---
class S3GenFineTuningDataset(Dataset):
    def __init__(self,
                 data_args: S3GenDataArguments,
                 s3gen_model: S3Gen, # Pass the S3Gen model directly
                 hf_dataset: datasets.Dataset):
        self.data_args = data_args
        self.s3gen_model = s3gen_model # Contains tokenizer, mel_extractor, speaker_encoder
        self.dataset_source = hf_dataset

        self.s3_tokenizer_sr = S3_TOKENIZER_SR # 16kHz for S3Tokenizer
        self.s3_gen_native_sr = S3GEN_SR     # 24kHz for mel spectrograms and CAMPPlus input
        
        self.prompt_audio_samples_16k = int(data_args.prompt_audio_duration_s * self.s3_tokenizer_sr)
        self.prompt_audio_samples_24k = int(data_args.prompt_audio_duration_s * self.s3_gen_native_sr)

        self._resamplers = {}

    def _get_resampler(self, orig_sr: int, target_sr: int) -> T.Resample:
        if (orig_sr, target_sr) not in self._resamplers:
            self._resamplers[(orig_sr, target_sr)] = T.Resample(orig_sr, target_sr)
        return self._resamplers[(orig_sr, target_sr)]

    def __len__(self):
        return len(self.dataset_source)

    def _load_and_preprocess_audio(self, audio_data_from_hf):
        waveform: Optional[torch.Tensor] = None
        original_sr: Optional[int] = None

        if isinstance(audio_data_from_hf, str):
            try: waveform, original_sr = torchaudio.load(audio_data_from_hf)
            except Exception: return None, None
        elif isinstance(audio_data_from_hf, dict) and "array" in audio_data_from_hf and "sampling_rate" in audio_data_from_hf:
            np_array = audio_data_from_hf["array"]
            if not isinstance(np_array, np.ndarray): return None, None
            waveform = torch.from_numpy(np_array).float()
            original_sr = audio_data_from_hf["sampling_rate"]
            if waveform.ndim == 1: waveform = waveform.unsqueeze(0)
        else: return None, None

        if waveform is None or original_sr is None or waveform.numel() == 0: return None, None
        if waveform.shape[0] > 1: waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Ensure float32
        waveform = waveform.float()

        # Prepare 16kHz version (for S3Tokenizer and potentially CAMPPlus if it expects 16k)
        if original_sr != self.s3_tokenizer_sr:
            resampler_16k = self._get_resampler(original_sr, self.s3_tokenizer_sr)
            wav_16k_tensor = resampler_16k(waveform)
        else:
            wav_16k_tensor = waveform.clone() # Clone to avoid modifying dataset source waveform
        
        # Prepare 24kHz version (for mel_extractor and CAMPPlus if it expects 24k)
        # Note: s3gen.speaker_encoder (CAMPPlus) in s3gen.py takes 16kHz input.
        # s3gen.mel_extractor takes 24kHz input.
        if original_sr != self.s3_gen_native_sr:
            resampler_24k = self._get_resampler(original_sr, self.s3_gen_native_sr)
            wav_24k_tensor = resampler_24k(waveform)
        else:
            wav_24k_tensor = waveform.clone()

        return wav_16k_tensor.squeeze(0), wav_24k_tensor.squeeze(0) # Return (L,) tensors

    def __getitem__(self, idx) -> Optional[Dict[str, torch.Tensor]]:
        item = self.dataset_source[idx]
        audio_data_hf = item[self.data_args.audio_column_name]

        wav_16k_tensor, wav_24k_tensor = self._load_and_preprocess_audio(audio_data_hf)
        if wav_16k_tensor is None or wav_24k_tensor is None or wav_16k_tensor.numel() == 0 or wav_24k_tensor.numel() == 0:
            return None

        # 1. Target Mel Spectrograms (speech_feat) from full 24kHz audio
        # mel_spectrogram returns (B, F, T_mel), we need (F, T_mel) then transpose to (T_mel, F) for flow model
        try:
            target_mel = self.s3gen_model.mel_extractor(wav_24k_tensor.unsqueeze(0)).squeeze(0).transpose(0, 1) # (T_mel, F)
            if target_mel.size(0) > self.data_args.max_mel_len:
                target_mel = target_mel[:self.data_args.max_mel_len, :]
            speech_feat_len = torch.tensor(target_mel.size(0), dtype=torch.long)
        except Exception as e:
            logger.error(f"Item {idx}: Error extracting target_mel: {e}", exc_info=True)
            return None

        # 2. Target S3 Speech Tokens (speech_token) from full 16kHz audio
        # S3Tokenizer expects list of numpy arrays
        try:
            speech_tokens_batch, speech_token_lengths_batch = self.s3gen_model.tokenizer.forward(
                [wav_16k_tensor.numpy()], max_len=self.data_args.max_speech_token_len
            )
            if speech_tokens_batch is None or speech_token_lengths_batch is None: return None
            target_s3_tokens = speech_tokens_batch.squeeze(0) # (T_tokens_s3)
            speech_token_len = speech_token_lengths_batch.squeeze(0)
            # Ensure token length matches mel length (T_mel = 2 * T_tokens_s3 usually)
            # This alignment is crucial. If S3Tokenizer's max_len truncates differently than mel's max_len, adjust.
            # For simplicity, assume S3Tokenizer's max_len aligns with max_mel_len / 2
            if target_mel.size(0) != 2 * target_s3_tokens.size(0) and target_mel.size(0) // 2 < target_s3_tokens.size(0):
                target_s3_tokens = target_s3_tokens[:target_mel.size(0)//2]
                speech_token_len = torch.tensor(target_s3_tokens.size(0), dtype=torch.long)
            elif target_mel.size(0) // 2 > target_s3_tokens.size(0) : # Pad tokens if mel is longer due to truncation
                 pad_size = target_mel.size(0)//2 - target_s3_tokens.size(0)
                 target_s3_tokens = F.pad(target_s3_tokens, (0, pad_size), value=0) # Pad with 0
                 # speech_token_len remains the original length before padding for this case

        except Exception as e:
            logger.error(f"Item {idx}: Error tokenizing target speech: {e}", exc_info=True)
            return None

        # 3. Speaker Embedding (embedding) from 16kHz audio (as per S3Gen's CAMPPlus usage)
        try:
            # speaker_encoder.inference expects (B, L) tensor
            speaker_embedding = self.s3gen_model.speaker_encoder.inference(wav_16k_tensor.unsqueeze(0)).squeeze(0) # (D_spk)
        except Exception as e:
            logger.error(f"Item {idx}: Error getting speaker_embedding: {e}", exc_info=True)
            return None

        # 4. Prompt features for CausalMaskedDiffWithXvec conditioning
        # prompt_token, prompt_token_len, prompt_feat, prompt_feat_len
        prompt_wav_16k_segment = wav_16k_tensor[:self.prompt_audio_samples_16k]
        prompt_wav_24k_segment = wav_24k_tensor[:self.prompt_audio_samples_24k]

        if prompt_wav_16k_segment.numel() == 0 or prompt_wav_24k_segment.numel() == 0: # Handle very short audio
            # logger.warning(f"Item {idx}: Prompt audio segment too short. Using zero prompts.")
            # Max prompt token length for Causal model, assumed to be something like 150 (3s * 25Hz * 2 for mel ratio)
            # This needs to be consistent with how CausalMaskedDiffWithXvec expects prompt lengths.
            # For now, let's set a placeholder; this might need adjustment based on model internals.
            # The `embed_ref` in S3Gen handles this. Let's try to replicate part of it.
            # S3Gen's embed_ref uses self.t3.hp.speech_cond_prompt_len for T3, which is not right here.
            # S3Gen uses up to 10s for ref for S3Gen itself. Let's assume prompt for causal flow is also significant.
            # Max len for prompt tokens used by Causal flow (this is a guess, check model)
            max_flow_prompt_token_len = self.s3gen_model.flow.encoder.hp.get("prompt_token_max_len", 75) # Example, should come from flow model config
            
            prompt_s3_tokens = torch.zeros(max_flow_prompt_token_len, dtype=torch.long)
            prompt_s3_token_len = torch.tensor(0, dtype=torch.long)
            prompt_mel = torch.zeros(max_flow_prompt_token_len * 2, target_mel.size(1), dtype=torch.float) # (T_prompt_mel, F)
            prompt_mel_len = torch.tensor(0, dtype=torch.long)
        else:
            try:
                # Prompt S3 tokens (from 16kHz prompt audio)
                # Assuming CausalMaskedDiffWithXvec might have a max prompt length for tokens
                max_flow_prompt_token_len = getattr(self.s3gen_model.flow.encoder, 'prompt_token_max_len', 75) # Check actual attribute if exists
                prompt_s3_tokens_batch, prompt_s3_token_lengths_batch = self.s3gen_model.tokenizer.forward(
                    [prompt_wav_16k_segment.numpy()], max_len=max_flow_prompt_token_len
                )
                if prompt_s3_tokens_batch is None: return None
                prompt_s3_tokens = prompt_s3_tokens_batch.squeeze(0)
                prompt_s3_token_len = prompt_s3_token_lengths_batch.squeeze(0)

                # Prompt Mel (from 24kHz prompt audio)
                prompt_mel = self.s3gen_model.mel_extractor(prompt_wav_24k_segment.unsqueeze(0)).squeeze(0).transpose(0,1) # (T_mel_prompt, F)
                # Ensure prompt_mel length aligns with prompt_s3_tokens length (T_mel = 2 * T_tokens)
                if prompt_mel.size(0) > prompt_s3_tokens.size(0) * 2:
                    prompt_mel = prompt_mel[:prompt_s3_tokens.size(0) * 2, :]
                prompt_mel_len = torch.tensor(prompt_mel.size(0), dtype=torch.long)
                
                # If prompt_s3_tokens were truncated by max_len, adjust its length for consistency
                if prompt_mel.size(0) // 2 < prompt_s3_tokens.size(0):
                    prompt_s3_tokens = prompt_s3_tokens[:prompt_mel.size(0)//2]
                    prompt_s3_token_len = torch.tensor(prompt_s3_tokens.size(0), dtype=torch.long)


            except Exception as e:
                logger.error(f"Item {idx}: Error processing prompt features: {e}", exc_info=True)
                return None
        
        return {
            "speech_token": target_s3_tokens.long(),       # Target S3 tokens
            "speech_token_len": speech_token_len.long(), # Length of target S3 tokens
            "speech_feat": target_mel.float(),            # Target mel spectrogram (T_mel, F)
            "speech_feat_len": speech_feat_len.long(),    # Length of target mel (num frames)
            "embedding": speaker_embedding.float(),       # Speaker embedding (x-vector)
            # For CausalMaskedDiffWithXvec, prompt info is part of conditioning:
            "prompt_token": prompt_s3_tokens.long(),
            "prompt_token_len": prompt_s3_token_len.long(),
            "prompt_feat": prompt_mel.float(),            # (T_mel_prompt, F)
            # "prompt_feat_len" is implicitly prompt_mel.size(0), flow model might expect it explicitly
            # S3Gen's embed_ref sets prompt_feat_len=None, CausalFlow seems to use prompt_feat.shape[1]
        }

# --- S3Gen Data Collator ---
@dataclass
class S3GenDataCollator:
    # Padding values. Flow model might expect specific padding for mels (e.g., 0 or -log(1e-5))
    # S3 tokens are padded with 0.
    speech_token_pad_id: int = 0
    mel_pad_value: float = 0.0 # Or whatever the flow model expects for padded mel frames

    # Max lengths for prompts, if flow model expects fixed size padded prompts during training
    # This needs to match how CausalMaskedDiffWithXvec's encoder processes prompts
    # For now, assume dynamic padding for prompts in batch.
    # max_prompt_token_len_collator: Optional[int] = 75 # Example
    # max_prompt_mel_len_collator: Optional[int] = 150 # Example

    def __call__(self, features: List[Optional[Dict[str, torch.Tensor]]]) -> Dict[str, Any]:
        valid_features = [f for f in features if f is not None]
        if not valid_features: return {}
        
        batch_size = len(valid_features)
        # Device from first valid sample
        device = valid_features[0]["speech_token"].device if batch_size > 0 else "cpu"


        # Pad speech_token (target S3 tokens)
        speech_tokens = [f["speech_token"] for f in valid_features]
        max_len_st = max(s.size(0) for s in speech_tokens)
        padded_speech_tokens = torch.stack(
            [F.pad(s, (0, max_len_st - s.size(0)), value=self.speech_token_pad_id) for s in speech_tokens]
        )
        speech_token_lens = torch.stack([f["speech_token_len"] for f in valid_features])

        # Pad speech_feat (target mel spectrograms)
        # Mels are (T_mel, F). Pad along T_mel dimension (dim 0).
        speech_feats = [f["speech_feat"] for f in valid_features]
        max_len_sf = max(s.size(0) for s in speech_feats)
        mel_dim = speech_feats[0].size(1) # F dimension
        padded_speech_feats = torch.stack(
            [F.pad(s, (0, 0, 0, max_len_sf - s.size(0)), value=self.mel_pad_value) for s in speech_feats] # Pads last dim first (T_mel)
        ) # Result (B, T_mel_max, F)
        speech_feat_lens = torch.stack([f["speech_feat_len"] for f in valid_features])
        
        # Stack speaker embeddings
        embeddings = torch.stack([f["embedding"] for f in valid_features])

        # --- Pad prompt features ---
        prompt_tokens = [f["prompt_token"] for f in valid_features]
        # Use a fixed max prompt length if required by model, else pad to max in batch
        # target_prompt_token_len = self.max_prompt_token_len_collator or max(pt.size(0) for pt in prompt_tokens)
        target_prompt_token_len = max(pt.size(0) for pt in prompt_tokens) if prompt_tokens else 0
        target_prompt_token_len = max(1, target_prompt_token_len) # Ensure not 0

        padded_prompt_tokens = torch.stack(
             [F.pad(pt, (0, target_prompt_token_len - pt.size(0)), value=self.speech_token_pad_id) for pt in prompt_tokens]
        )
        prompt_token_lens = torch.stack([f["prompt_token_len"] for f in valid_features])

        prompt_feats = [f["prompt_feat"] for f in valid_features]
        # target_prompt_mel_len = self.max_prompt_mel_len_collator or max(pf.size(0) for pf in prompt_feats)
        target_prompt_mel_len = max(pf.size(0) for pf in prompt_feats) if prompt_feats else 0
        target_prompt_mel_len = max(1, target_prompt_mel_len)
        
        padded_prompt_feats = torch.stack(
            [F.pad(pf, (0, 0, 0, target_prompt_mel_len - pf.size(0)), value=self.mel_pad_value) for pf in prompt_feats]
        )
        # prompt_feat_lens = torch.stack([f["prompt_feat_len"] for f in valid_features]) # If dataset provides it

        # The CausalMaskedDiffWithXvec.forward in flow.py might not take all these directly
        # It was designed for inference. Its training `forward` (if like MaskedDiffWithXvec)
        # expects a `batch` dict.
        # The S3Gen model's `flow` attribute is CausalMaskedDiffWithXvec.
        # We need to construct the `batch` dict that ITS `forward` method expects for training.

        # Looking at CausalMaskedDiffWithXvec.inference, it takes:
        # token, token_len, prompt_token, prompt_token_len, prompt_feat, embedding
        # (prompt_feat_len is derived from prompt_feat.shape[1] for mel length)
        # For training, it likely expects `speech_token` as `token`, `speech_feat` as the target `feat`.
        
        # The MaskedDiffWithXvec.forward(self, batch, device) is the training entry point.
        # We need to provide a dictionary that matches this structure.
        return {
            'speech_token': padded_speech_tokens.to(device),
            'speech_token_len': speech_token_lens.to(device),
            'speech_feat': padded_speech_feats.to(device), # This is the target mel
            'speech_feat_len': speech_feat_lens.to(device),
            'embedding': embeddings.to(device),
            # These are needed for CausalMaskedDiffWithXvec's conditioning during its internal forward call.
            # The structure for Causal model might embed these within `h` or `conds` for CFM.
            # For now, let's pass them explicitly, the wrapper will need to handle them.
            # CausalMaskedDiffWithXvec's *training* forward is NOT DEFINED in flow.py.
            # It only has an `inference` method.
            # MaskedDiffWithXvec has a training forward. We assume Causal* would be similar.
            # The `compute_loss` of its `decoder` (CausalConditionalCFM) expects:
            # x (target_mel), mask, cond (encoder_out_h), spk_emb, cond (prompt_mel)
            # So, the flow model's forward needs to prepare these.

            # For the Trainer, the keys passed to S3GenFlowForFineTuning.forward need to be here.
            # Let's assume S3GenFlowForFineTuning.forward will take these and construct the necessary inputs for
            # s3gen_model.flow.decoder.compute_loss, potentially via a simplified s3gen_model.flow.train_forward().

            # Required by CausalMaskedDiffWithXvec.inference, likely also needed for training cond:
            "prompt_token_input": padded_prompt_tokens.to(device),
            "prompt_token_len_input": prompt_token_lens.to(device),
            "prompt_feat_input": padded_prompt_feats.to(device),
            # "prompt_feat_len_input": prompt_feat_lens.to(device) # Derived from shape usually
        }


# --- Model Wrapper for S3Gen's Flow component ---
class S3GenFlowForFineTuning(torch.nn.Module):
    def __init__(self, s3gen_token_to_mel: S3Token2Mel): # Pass S3Token2Mel instance
        super().__init__()
        self.s3_token_to_mel = s3gen_token_to_mel
        self.flow_model: CausalMaskedDiffWithXvec = self.s3_token_to_mel.flow # type: ignore
        
        # Create a dummy HF Compatible Config
        class HFCompatibleConfig(PretrainedConfig):
            model_type = "chatterbox_s3gen_flow_finetune"
            # Add any S3Gen/Flow specific hparams you want to save in config.json
            def __init__(self, **kwargs): super().__init__(**kwargs)
        self.config = HFCompatibleConfig()
        # Populate self.config with relevant params from s3_token_to_mel or flow_model if needed

    def forward(self, 
                speech_token: torch.Tensor,        # Target S3 tokens for the flow model's encoder
                speech_token_len: torch.Tensor,
                speech_feat: torch.Tensor,         # Target mel features (T_mel, F)
                speech_feat_len: torch.Tensor,
                embedding: torch.Tensor,           # Speaker embedding
                prompt_token_input: torch.Tensor,  # Prompt S3 tokens for conditioning
                prompt_token_len_input: torch.Tensor,
                prompt_feat_input: torch.Tensor,   # Prompt mel features for conditioning
                labels = None # Unused, for Trainer compatibility
                ):
        # The CausalMaskedDiffWithXvec does not have a `forward` for training in your flow.py.
        # It has `inference`. We need to call the `decoder.compute_loss` part.
        # This usually happens within a training-specific forward method of the flow model.
        # Let's assume we add a `train_step` or `compute_train_loss` to CausalMaskedDiffWithXvec.
        # Or, we replicate the logic from MaskedDiffWithXvec.forward here.

        # Replicating logic from MaskedDiffWithXvec.forward:
        # 1. Project speaker embedding
        projected_speaker_emb = self.flow_model.spk_embed_affine_layer(F.normalize(embedding, dim=1))

        # 2. Encode input `speech_token` (these are the main tokens, not prompt)
        #    For Causal model, the encoder takes concatenated prompt_token + speech_token.
        #    The `token` input to Causal*.inference is already the main content tokens.
        #    The `prompt_token` is separate.
        
        # Encoder input for Causal model: concat prompt_token and main speech_token
        full_input_tokens = torch.cat([prompt_token_input, speech_token], dim=1)
        full_input_token_lens = prompt_token_len_input + speech_token_len

        input_token_embed = self.flow_model.input_embedding(torch.clamp(full_input_tokens, min=0))
        # Apply mask based on full_input_token_lens if encoder expects it
        # mask = (~make_pad_mask(full_input_token_lens)).float().unsqueeze(-1).to(input_token_embed.device)
        # input_token_embed = input_token_embed * mask
        
        # Text encode (actually, semantic token encode)
        # h is the output of the upsample_encoder
        h, h_lengths = self.flow_model.encoder(input_token_embed, full_input_token_lens) 
        # For Causal, if `finalize=False` was used in inference, it slices h.
        # For training, we usually use the full length.
        # The target `speech_feat` corresponds to the `speech_token` part, not the prompt part.
        # So `h` needs to be sliced or length adjusted if it was for full_input_tokens.
        # The `CausalMaskedDiffWithXvec.inference` does:
        #   h, h_lengths = self.encoder(token_embed_for_full_sequence, token_len_for_full_sequence)
        #   if finalize is False: h = h[:, :-self.pre_lookahead_len * self.token_mel_ratio]
        #   mel_len2 = h.shape[1] - prompt_feat.shape[1] (mel_len1)
        # So `h` from encoder corresponds to `prompt_mel_len + actual_mel_len`.
        # The target `speech_feat` has length `speech_feat_len`.
        
        # We need `h` that corresponds in length to `speech_feat` after upsampling and projection.
        # The upsample encoder in Causal flow model directly outputs features at mel_frame rate.
        # `h` from `self.flow_model.encoder` is already at mel rate.
        # It needs to be sliced to get the part corresponding to `speech_token` (not `prompt_token_input`).
        
        # `h` has shape (B, T_encoded_prompt + T_encoded_speech, D_encoder_proj)
        # We need the part of `h` that corresponds to `speech_token` for conditioning the CFM decoder.
        # Length of encoded prompt part: prompt_token_len_input (token rate) -> prompt_mel_len_input (mel rate)
        # Length of encoded speech part: speech_token_len (token rate) -> speech_feat_len (mel rate)
        # The upsampler encoder in Causal flow does this length change.
        # `h_lengths` should correspond to `prompt_mel_len + speech_mel_len`.
        
        h_projected = self.flow_model.encoder_proj(h) # (B, T_prompt_mel + T_speech_mel, D_output)

        # The CFM decoder (`self.flow_model.decoder`) needs:
        # x (target_mel): speech_feat.transpose(1,2) -> (B, F, T_mel)
        # cond (encoder_out): h_for_target_mel.transpose(1,2) -> (B, D_output, T_mel)
        # spks: projected_speaker_emb
        # cond (prompt_mel for CFM): prompt_feat_input.transpose(1,2) -> (B, F, T_prompt_mel)

        # Slice h_projected to get only the part for the target speech tokens
        # prompt_mel_len_from_input_feat = prompt_feat_input.size(1) # T_prompt_mel
        # h_for_target_mels = h_projected[:, prompt_mel_len_from_input_feat : prompt_mel_len_from_input_feat + speech_feat.size(1), :]
        # This slicing based on prompt_feat_input might be tricky if padding differs.
        # Simpler: `h_projected` from encoder covers both prompt and target. Decoder needs to know prompt length.

        # The CausalConditionalCFM.compute_loss needs `cond` (h_projected) and `x` (speech_feat)
        # and it internally handles the prompt via its `cond` argument if it's `prompt_feat_input`.
        # The `mu` in CausalConditionalCFM is `h_projected`.
        # `self.flow_model.decoder` is `CausalConditionalCFM`
        
        # Target mels: (B, T_mel, F) -> need (B, F, T_mel) for CFM
        target_mels_for_cfm = speech_feat.transpose(1,2).contiguous()
        
        # Conditioning from encoder: (B, T_total_mel, D) -> need (B, D, T_total_mel) for CFM
        encoder_output_for_cfm = h_projected.transpose(1,2).contiguous()

        # Prompt mels for CFM conditioning: (B, T_prompt_mel, F) -> (B, F, T_prompt_mel)
        prompt_mels_for_cfm_cond = prompt_feat_input.transpose(1,2).contiguous()
        
        # Create mask for target_mels_for_cfm based on speech_feat_len
        # Mask shape (B, 1, T_mel)
        target_mel_mask = (~self.make_pad_mask(speech_feat_len, max_len=target_mels_for_cfm.size(2))).unsqueeze(1).to(target_mels_for_cfm.device)
        # The CFM model internally requires mu (encoder_output_for_cfm) to have same T dim as x (target_mels_for_cfm)
        # This implies CausalMaskedDiffWithXvec's encoder output `h` must be sliced to match only the target part for CFM loss calculation,
        # OR the CFM decoder handles prompts by taking the full `h` and a `prompt_len` argument.
        # CausalConditionalCFM's forward pass takes `prompt_len`.

        # From CausalMaskedDiffWithXvec.inference:
        # feat, _ = self.decoder(mu=h.transpose(1,2), ..., cond=prompt_mels_for_cfm_cond, ..., prompt_len=mel_len1)
        # This means `mu` (h_projected) must correspond to the *full* sequence (prompt + target).
        # And `x` in `compute_loss` (target_mels_for_cfm) corresponds to the *target* part.
        # The `prompt_len` argument to CFM.compute_loss tells it how much of `mu` is prompt.
        
        prompt_mel_len_for_cfm = prompt_feat_input.size(1) # T_prompt_mel from input

        loss_dict = self.flow_model.decoder.compute_loss(
            x=target_mels_for_cfm,            # Target mels (B, F, T_target_mel)
            mask=target_mel_mask,             # Mask for target mels (B, 1, T_target_mel)
            mu=encoder_output_for_cfm,        # Encoder output for (prompt+target) (B, D, T_total_mel)
            spks=projected_speaker_emb,       # Speaker embedding
            cond=prompt_mels_for_cfm_cond,    # Prompt mels for conditioning CFM (B, F, T_prompt_mel)
            prompt_len=prompt_mel_len_for_cfm # Length of prompt part in mu and cond
        )
        
        # loss_dict likely contains {'loss': tensor, 'reg_loss': tensor, ...}
        # We primarily care about the main 'loss' for backprop.
        # Can return others for logging if needed.
        main_loss = loss_dict['loss']
        if 'reg_loss' in loss_dict: # Add regularization if present
            main_loss += loss_dict['reg_loss']

        return (main_loss, loss_dict.get('loss'), loss_dict.get('reg_loss')) # Return tuple for Trainer

    def make_pad_mask(self, lengths: torch.Tensor, max_len: Optional[int] = None) -> torch.Tensor:
        # Helper from CosyVoice/utils/mask.py, adapted
        if max_len is None:
            max_len = torch.max(lengths).item()
        bs = lengths.size(0)
        seq_range = torch.arange(0, max_len, device=lengths.device)
        seq_range_expand = seq_range.unsqueeze(0).expand(bs, max_len)
        seq_length_expand = lengths.unsqueeze(1).expand_as(seq_range_expand)
        return seq_range_expand >= seq_length_expand


# Global trainer instance
trainer_instance: Optional[Trainer] = None

def main():
    global trainer_instance
    parser = HfArgumentParser((S3GenModelArguments, S3GenDataArguments, CustomTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", datefmt="%m/%d/%Y %H:%M:%S",
                        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN)
    logger.info("Training/evaluation parameters %s", training_args)
    logger.info("Model parameters %s", model_args)
    logger.info("Data parameters %s", data_args)
    set_seed(training_args.seed)

    # --- Load Base Chatterbox Model to get S3Gen ---
    logger.info("Loading base ChatterboxTTS model to extract S3Gen components...")
    # This part is similar to T3 finetuning: load the full model first.
    if model_args.local_model_dir:
        chatterbox_model = ChatterboxTTS.from_local(ckpt_dir=model_args.local_model_dir, device="cpu")
    else:
        repo_to_download = model_args.model_name_or_path or REPO_ID # Fallback to default REPO_ID
        download_dir = Path(training_args.output_dir) / "pretrained_chatterbox_download"
        download_dir.mkdir(parents=True, exist_ok=True)
        from huggingface_hub import hf_hub_download
        files_to_download = ["ve.safetensors", "t3_cfg.safetensors", "s3gen.safetensors", "tokenizer.json", "conds.pt"]
        for f in files_to_download:
            try: hf_hub_download(repo_id=repo_to_download, filename=f, local_dir=download_dir, local_dir_use_symlinks=False, cache_dir=model_args.cache_dir)
            except Exception as e: logger.warning(f"Could not download {f} from {repo_to_download}: {e}")
        chatterbox_model = ChatterboxTTS.from_local(ckpt_dir=download_dir, device="cpu")

    s3gen_model_to_finetune: S3Gen = chatterbox_model.s3gen
    s3gen_token_to_mel_part: S3Token2Mel = s3gen_model_to_finetune # S3Gen inherits from S3Token2Wav which inherits from S3Token2Mel

    # --- Freeze parts of S3Gen ---
    # Freeze HiFiGAN part (mel2wav)
    for param in s3gen_token_to_mel_part.mel2wav.parameters():
        param.requires_grad = False
    logger.info("S3Gen HiFiGAN (mel2wav) part frozen.")

    if model_args.freeze_speaker_encoder:
        for param in s3gen_token_to_mel_part.speaker_encoder.parameters():
            param.requires_grad = False
        logger.info("S3Gen Speaker Encoder (CAMPPlus) frozen.")
    
    if model_args.freeze_s3_tokenizer:
        # S3Tokenizer in Chatterbox doesn't have trainable params in its dummy/provided version,
        # but a real one might.
        if hasattr(s3gen_token_to_mel_part.tokenizer, 'parameters'):
             for param in s3gen_token_to_mel_part.tokenizer.parameters(): # type: ignore
                param.requires_grad = False
        logger.info("S3Gen S3Tokenizer frozen (if it has parameters).")

    # Ensure the flow model part is trainable
    for param in s3gen_token_to_mel_part.flow.parameters():
        param.requires_grad = True
    logger.info("S3Gen Flow Model (CausalMaskedDiffWithXvec) set to trainable.")


    # --- Prepare Dataset ---
    logger.info("Loading and processing dataset for S3Gen finetuning...")
    verification_mode = VerificationMode.NO_CHECKS if data_args.ignore_verifications else VerificationMode.BASIC_CHECKS
    if data_args.dataset_name:
        raw_datasets = load_dataset(data_args.dataset_name, data_args.dataset_config_name,
                                    cache_dir=model_args.cache_dir, verification_mode=verification_mode)
        train_hf_dataset = raw_datasets[data_args.train_split_name]
        eval_hf_dataset = raw_datasets.get(data_args.eval_split_name) if data_args.eval_split_name else None
        if training_args.do_eval and not eval_hf_dataset and data_args.eval_split_size > 0 and len(train_hf_dataset) > 1:
            split_dataset = train_hf_dataset.train_test_split(test_size=data_args.eval_split_size, seed=training_args.seed)
            train_hf_dataset, eval_hf_dataset = split_dataset["train"], split_dataset["test"]
    else:
        raise ValueError("S3Gen finetuning currently requires a Hugging Face dataset_name.") # Simpler for now

    train_dataset = S3GenFineTuningDataset(data_args, s3gen_model_to_finetune, train_hf_dataset)
    eval_dataset = None
    if eval_hf_dataset and training_args.do_eval:
        eval_dataset = S3GenFineTuningDataset(data_args, s3gen_model_to_finetune, eval_hf_dataset)

    # --- Data Collator ---
    data_collator = S3GenDataCollator() # Uses defaults for padding

    # --- Model Wrapper for Trainer ---
    s3gen_flow_trainable_model = S3GenFlowForFineTuning(s3gen_token_to_mel_part)

    # --- Compute Metrics (can log individual loss components) ---
    def compute_metrics_s3gen(eval_preds):
        metrics = {}
        if isinstance(eval_preds.predictions, tuple) and len(eval_preds.predictions) >= 1:
            # eval_preds.predictions[0] is main_loss
            # eval_preds.predictions[1] is cfm_loss (if returned)
            # eval_preds.predictions[2] is reg_loss (if returned)
            # Trainer automatically logs eval_preds.predictions[0] as eval_loss
            if len(eval_preds.predictions) > 1 and eval_preds.predictions[1] is not None:
                metrics["eval_cfm_loss"] = float(np.mean(eval_preds.predictions[1]))
            if len(eval_preds.predictions) > 2 and eval_preds.predictions[2] is not None:
                metrics["eval_reg_loss"] = float(np.mean(eval_preds.predictions[2]))
        return metrics

    # --- Callbacks ---
    callbacks = []
    if training_args.early_stopping_patience is not None and training_args.early_stopping_patience > 0:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=training_args.early_stopping_patience))
    # Add audio generation callback later if needed (more complex for S3Gen eval)

    # --- Trainer ---
    logger.info(f"Using dataloader_pin_memory: {training_args.dataloader_pin_memory}")
    trainer_instance = Trainer(
        model=s3gen_flow_trainable_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics_s3gen if training_args.do_eval and eval_dataset else None,
        callbacks=callbacks if callbacks else None
    )
    if training_args.label_names is None: trainer_instance.label_names = [] # Model handles its own targets

    # --- Training ---
    if training_args.do_train:
        logger.info("*** Finetuning S3Gen Flow Model ***")
        train_result = trainer_instance.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        trainer_instance.save_model() # Saves Trainer-wrapped S3GenFlowForFineTuning
        
        logger.info("Saving finetuned S3Gen (flow part) model weights for ChatterboxTTS...")
        # The S3GenFlowForFineTuning model IS the s3_token_to_mel part (or its flow sub-module)
        # We need to save the state_dict of s3gen_token_to_mel_part
        # The trainer saves s3gen_flow_trainable_model.state_dict(), which contains s3_token_to_mel.xxx keys
        
        # To save in the original Chatterbox S3Gen format:
        # We need the state_dict of the s3gen_model_to_finetune, which now has updated flow params.
        finetuned_s3gen_state_dict = s3gen_model_to_finetune.state_dict()
        output_s3gen_safetensor_path = Path(training_args.output_dir) / "s3gen.safetensors"
        from safetensors.torch import save_file
        save_file(finetuned_s3gen_state_dict, output_s3gen_safetensor_path)
        logger.info(f"Finetuned S3Gen model weights saved to {output_s3gen_safetensor_path}")

        # Copy other necessary files for a complete local model dir from the *original* download
        # This assumes you want to package the finetuned s3gen with the original T3, VE etc.
        # If original_model_dir_for_copy is defined from the chatterbox_model loading:
        # if original_model_dir_for_copy: # (Define this based on initial load path)
        #     import shutil
        #     for f_name in ["ve.safetensors", "t3_cfg.safetensors", "tokenizer.json", "conds.pt"]:
        #         src_path = original_model_dir_for_copy / f_name
        #         if src_path.exists(): shutil.copy2(src_path, Path(training_args.output_dir) / f_name)
        #     logger.info(f"Other original model components copied to {training_args.output_dir}")


        metrics = train_result.metrics
        trainer_instance.log_metrics("train", metrics)
        trainer_instance.save_metrics("train", metrics)
        trainer_instance.save_state()

    # --- Evaluation ---
    if training_args.do_eval and eval_dataset:
        logger.info("*** Evaluating S3Gen Flow Model ***")
        metrics = trainer_instance.evaluate()
        trainer_instance.log_metrics("eval", metrics)
        trainer_instance.save_metrics("eval", metrics)

    logger.info("S3Gen finetuning script finished.")

if __name__ == "__main__":
    main()