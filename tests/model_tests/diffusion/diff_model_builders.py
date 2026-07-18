import os
import tempfile

import torch
from diffusers import (
    AutoencoderKLLTX2Audio,
    AutoencoderKLLTX2Video,
    FlowMatchEulerDiscreteScheduler,
    LTX2Pipeline,
    LTX2VideoTransformer3DModel,
)
from diffusers.models.autoencoders.autoencoder_kl_flux2 import AutoencoderKLFlux2
from diffusers.models.transformers.transformer_flux2 import Flux2Transformer2DModel
from diffusers.pipelines.flux2.pipeline_flux2_klein import Flux2KleinPipeline
from diffusers.pipelines.ltx2 import LTX2TextConnectors
from diffusers.pipelines.ltx2.vocoder import LTX2VocoderWithBWE
from transformers import AutoTokenizer, Gemma3ForConditionalGeneration, Qwen3Config, Qwen3ForCausalLM

TINY_MODEL_DIR = os.path.join(tempfile.gettempdir(), "vllm-omni-tiny-models")


def _get_tiny_model_path(name: str) -> str:
    path = os.path.join(TINY_MODEL_DIR, name)
    os.makedirs(path, exist_ok=True)
    return path


def tiny_flux2_klein_builder() -> str:
    """Build a tiny Flux2Klein model."""
    model_dir = _get_tiny_model_path("Flux2KleinPipeline")

    pipe = Flux2KleinPipeline(
        scheduler=FlowMatchEulerDiscreteScheduler(),
        vae=AutoencoderKLFlux2(
            down_block_types=("DownEncoderBlock2D",),
            up_block_types=("UpDecoderBlock2D",),
            block_out_channels=(32,),
            layers_per_block=1,
            latent_channels=16,
            norm_num_groups=16,
        ),
        # NOTE: For now we need 28 layers because of hardcoded stuff in the model :(
        text_encoder=Qwen3ForCausalLM(
            Qwen3Config(
                hidden_size=32,
                intermediate_size=64,
                num_hidden_layers=28,
                num_attention_heads=2,
                num_key_value_heads=2,
                head_dim=16,
                vocab_size=151936,
                max_position_embeddings=512,
            )
        ),
        tokenizer=AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B"),
        # NOTE: For now we need at least 2 layers for the transformer
        # due to hardcoded hacks in CacheDiT for Flux2Klein specifically.
        transformer=Flux2Transformer2DModel(
            in_channels=64,
            num_layers=2,
            num_single_layers=2,
            attention_head_dim=16,
            num_attention_heads=2,
            joint_attention_dim=96,
            timestep_guidance_channels=32,
            axes_dims_rope=(4, 4, 4, 4),
        ),
    )
    # Need dtypes to be consistent; for now we just put it on bfloat16
    pipe.to(torch.bfloat16).save_pretrained(model_dir)
    return model_dir


def tiny_ltx23_builder() -> str:
    """Build a tiny LTX-2.3 text-to-video model."""
    model_dir = _get_tiny_model_path("LTX23Pipeline")
    text_encoder_id = "hf-internal-testing/tiny-gemma3"
    tokenizer = AutoTokenizer.from_pretrained(text_encoder_id)
    tokenizer.model_max_length = 16
    text_encoder = Gemma3ForConditionalGeneration.from_pretrained(text_encoder_id)
    caption_channels = text_encoder.config.text_config.hidden_size

    torch.manual_seed(0)
    transformer = LTX2VideoTransformer3DModel(
        in_channels=4,
        out_channels=4,
        patch_size=1,
        patch_size_t=1,
        num_attention_heads=2,
        attention_head_dim=8,
        cross_attention_dim=16,
        audio_in_channels=4,
        audio_out_channels=4,
        audio_num_attention_heads=2,
        audio_attention_head_dim=4,
        audio_cross_attention_dim=8,
        num_layers=2,
        qk_norm="rms_norm_across_heads",
        caption_channels=caption_channels,
        rope_double_precision=False,
        rope_type="split",
        use_prompt_embeddings=False,
        perturbed_attn=True,
        gated_attn=True,
        cross_attn_mod=True,
        audio_gated_attn=True,
        audio_cross_attn_mod=True,
    )
    torch.manual_seed(0)
    connectors = LTX2TextConnectors(
        caption_channels=caption_channels,
        text_proj_in_factor=text_encoder.config.text_config.num_hidden_layers + 1,
        video_connector_num_attention_heads=2,
        video_connector_attention_head_dim=8,
        video_connector_num_layers=1,
        video_connector_num_learnable_registers=None,
        video_gated_attn=True,
        audio_connector_num_attention_heads=2,
        audio_connector_attention_head_dim=4,
        audio_connector_num_layers=1,
        audio_connector_num_learnable_registers=None,
        audio_gated_attn=True,
        connector_rope_base_seq_len=32,
        rope_theta=10000.0,
        rope_double_precision=False,
        causal_temporal_positioning=False,
        rope_type="split",
        per_modality_projections=True,
        video_hidden_dim=16,
        audio_hidden_dim=8,
        proj_bias=True,
    )
    torch.manual_seed(0)
    vae = AutoencoderKLLTX2Video(
        in_channels=3,
        out_channels=3,
        latent_channels=4,
        block_out_channels=(8,),
        decoder_block_out_channels=(8,),
        layers_per_block=(1,),
        decoder_layers_per_block=(1, 1),
        spatio_temporal_scaling=(True,),
        decoder_spatio_temporal_scaling=(True,),
        decoder_inject_noise=(False, False),
        downsample_type=("spatial",),
        upsample_residual=(False,),
        upsample_factor=(1,),
        timestep_conditioning=False,
        patch_size=1,
        patch_size_t=1,
        encoder_causal=True,
        decoder_causal=False,
    )
    vae.use_framewise_encoding = False
    vae.use_framewise_decoding = False
    torch.manual_seed(0)
    audio_vae = AutoencoderKLLTX2Audio(
        base_channels=4,
        output_channels=2,
        ch_mult=(1,),
        num_res_blocks=1,
        attn_resolutions=None,
        in_channels=2,
        resolution=32,
        latent_channels=2,
        norm_type="pixel",
        causality_axis="height",
        dropout=0.0,
        mid_block_add_attention=False,
        sample_rate=16000,
        mel_hop_length=160,
        is_causal=True,
        mel_bins=8,
    )
    torch.manual_seed(0)
    vocoder = LTX2VocoderWithBWE(
        in_channels=audio_vae.config.output_channels * audio_vae.config.mel_bins,
        hidden_channels=64,
        out_channels=2,
        upsample_kernel_sizes=[11, 4, 4, 4, 4, 4],
        upsample_factors=[5, 2, 2, 2, 2, 2],
        resnet_kernel_sizes=[3],
        resnet_dilations=[[1, 3, 5]],
        act_fn="leaky_relu",
        leaky_relu_negative_slope=0.1,
        antialias=False,
        final_act_fn="tanh",
        final_bias=True,
        bwe_in_channels=audio_vae.config.output_channels * audio_vae.config.mel_bins,
        bwe_hidden_channels=32,
        bwe_out_channels=2,
        bwe_upsample_kernel_sizes=[12, 11, 4, 4, 4],
        bwe_upsample_factors=[6, 5, 2, 2, 2],
        bwe_resnet_kernel_sizes=[3],
        bwe_resnet_dilations=[[1, 3, 5]],
        bwe_act_fn="leaky_relu",
        bwe_antialias=False,
        bwe_final_act_fn="tanh",
        bwe_final_bias=True,
        filter_length=512,
        hop_length=80,
        window_length=512,
        num_mel_channels=audio_vae.config.mel_bins,
        input_sampling_rate=16000,
        output_sampling_rate=48000,
    )
    pipe = LTX2Pipeline(
        scheduler=FlowMatchEulerDiscreteScheduler(),
        vae=vae,
        audio_vae=audio_vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        connectors=connectors,
        transformer=transformer,
        vocoder=vocoder,
        processor=None,
    )
    pipe.to(torch.bfloat16).save_pretrained(model_dir)
    return model_dir
