# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the additive structured Omni config."""

from __future__ import annotations

import warnings
from dataclasses import fields
from pathlib import Path

import pytest
from pydantic import ValidationError
from transformers import Qwen3OmniMoeConfig

from tests.helpers.stage_config import get_deploy_config_path
from vllm_omni.config import omni_config as omni_config_module
from vllm_omni.config.omni_config import (
    BaseVllmOmniStageConfig,
    OmniStageCacheConfig,
    OmniStageConnectorConfig,
    OmniStageDiffusionParallelConfig,
    OmniStageLoadConfig,
    OmniStageModelConfig,
    OmniStageParallelConfig,
    OmniStageRuntimeConfig,
    OmniStageSchedulerConfig,
    VllmOmniARStageConfig,
    VllmOmniConfig,
    VllmOmniDiffusionStageConfig,
    VllmOmniGenerationStageConfig,
)
from vllm_omni.config.pipeline_registry import OMNI_PIPELINES, resolve_pipeline_config
from vllm_omni.config.stage_config import (
    _STAGE_DEPLOY_FIELDS,
    PIPELINE_WIDE_ENGINE_FIELDS,
    DeployConfig,
    PipelineConfig,
    StageDeployConfig,
    StageExecutionType,
    load_deploy_config,
    merge_pipeline_deploy,
)
from vllm_omni.diffusion.data import OmniDiffusionConfig, TransformerConfig
from vllm_omni.engine.stage_init_utils import (
    _strict_diffusion_config_kwargs,
    build_diffusion_config,
    build_engine_args_dict,
    extract_stage_metadata,
    get_stage_devices_per_replica,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_DEPLOY_DIR = Path(__file__).parents[2] / "vllm_omni" / "deploy"


@pytest.fixture(autouse=True)
def _stable_test_platform(monkeypatch):
    from vllm_omni import platforms

    platform = platforms.current_omni_platform
    monkeypatch.setattr(platform, "device_name", "cpu", raising=False)
    monkeypatch.setattr(platform, "device_type", "cpu", raising=False)


def _load_default_deploy(pipeline: PipelineConfig) -> DeployConfig:
    if pipeline.default_deploy_config_name is not None:
        return load_deploy_config(_DEPLOY_DIR / pipeline.default_deploy_config_name)
    return DeployConfig()


def _resolve_pipeline_or_skip(model_type: str, hf_config=None) -> PipelineConfig:
    pipeline = resolve_pipeline_config(model_type, hf_config)
    if pipeline is None:
        pytest.skip(f"Pipeline {model_type!r} requires an HF config to resolve")
    return pipeline


def _from_pipeline_key(
    model_type: str,
    hf_config=None,
    deploy_config_path: str | None = None,
    cli_overrides: dict | None = None,
) -> VllmOmniConfig:
    return VllmOmniConfig.from_pipeline_config(
        _resolve_pipeline_or_skip(model_type, hf_config),
        deploy_config_path=deploy_config_path,
        cli_overrides=cli_overrides,
    )


def _build_dreamzero_stage(
    *,
    engine_extras: dict | None = None,
    cli_overrides: dict | None = None,
    **stage_values,
) -> VllmOmniDiffusionStageConfig:
    pipeline = _resolve_pipeline_or_skip("dreamzero")
    stage_deploy = StageDeployConfig(stage_id=0, engine_extras=engine_extras or {}, **stage_values)
    stage = VllmOmniConfig.from_pipeline_config(
        pipeline,
        user_deploy_config=DeployConfig(stages=[stage_deploy]),
        cli_overrides=cli_overrides,
    ).stage_by_id(0)
    assert isinstance(stage, VllmOmniDiffusionStageConfig)
    return stage


@pytest.mark.parametrize("model_type", sorted(OMNI_PIPELINES))
def test_vllm_omni_config_from_pipeline_config_matches_merge_pipeline_deploy(model_type: str):
    pipeline = _resolve_pipeline_or_skip(model_type)
    legacy_deploy = _load_default_deploy(pipeline)

    legacy_stages = merge_pipeline_deploy(pipeline, legacy_deploy)
    omni_config = VllmOmniConfig.from_pipeline_config(pipeline)

    assert omni_config.pipeline_config is pipeline
    assert len(omni_config.stage_configs) == len(legacy_stages)

    for legacy_stage, omni_stage in zip(legacy_stages, omni_config.stage_configs, strict=True):
        assert omni_config.stage_by_id(legacy_stage.stage_id) is omni_stage

        assert omni_stage.stage_pipeline_config is pipeline.get_stage(legacy_stage.stage_id)
        assert omni_stage.model_config.default_sampling_params == legacy_stage.yaml_extras.get(
            "default_sampling_params"
        )
        assert omni_stage.connector_config.output_connectors == legacy_stage.yaml_extras.get("output_connectors")
        assert omni_stage.connector_config.input_connectors == legacy_stage.yaml_extras.get("input_connectors")
        assert omni_stage.runtime_config.devices == legacy_stage.yaml_runtime.get("devices")
        assert omni_stage.runtime_config.num_replicas == legacy_stage.yaml_runtime.get("num_replicas", 1)

        engine_args = legacy_stage.yaml_engine_args
        assert omni_stage.model_config.enforce_eager == engine_args.get("enforce_eager", False)
        assert omni_stage.load_config.load_format == engine_args.get("load_format", "auto")
        assert omni_stage.load_config.tokenizer_mode == engine_args.get("tokenizer_mode", "auto")
        assert omni_stage.cache_config.gpu_memory_utilization == engine_args.get("gpu_memory_utilization", 0.90)
        assert omni_stage.scheduler_config.max_num_seqs == engine_args.get("max_num_seqs", 128)
        assert omni_stage.scheduler_config.max_num_batched_tokens == engine_args.get("max_num_batched_tokens")
        assert omni_stage.scheduler_config.async_scheduling == engine_args.get("async_scheduling", True)
        legacy_parallel_config = engine_args.get("parallel_config") or {}
        assert omni_stage.parallel_config.tensor_parallel_size == legacy_parallel_config.get(
            "tensor_parallel_size",
            engine_args.get("tensor_parallel_size", 1),
        )

        if omni_stage.stage_pipeline_config.execution_type == StageExecutionType.DIFFUSION:
            assert isinstance(omni_stage, VllmOmniDiffusionStageConfig)
            assert omni_stage.diffusion_config is not None
            assert omni_stage.diffusion_config.stage_id == legacy_stage.stage_id
            assert omni_stage.diffusion_config.model_arch == engine_args.get("model_arch")
            assert omni_stage.diffusion_config.omni_kv_config == engine_args.get("omni_kv_config", {})
        elif omni_stage.stage_pipeline_config.execution_type == StageExecutionType.LLM_AR:
            assert isinstance(omni_stage, VllmOmniARStageConfig)
            assert not hasattr(omni_stage, "diffusion_config")
        else:
            assert isinstance(omni_stage, VllmOmniGenerationStageConfig)
            assert not hasattr(omni_stage, "diffusion_config")


def test_stage_by_id_raises_for_unknown_stage():
    omni_config = _from_pipeline_key("qwen3_tts")

    with pytest.raises(KeyError, match="no stage 99"):
        omni_config.stage_by_id(99)


def test_resolve_execution_mode_rejects_unknown_execution_type():
    with pytest.raises(ValueError, match="Unsupported stage execution type"):
        omni_config_module._resolve_execution_mode("unknown_execution_type")


def test_from_pipeline_config_preserves_current_pipeline_config_object():
    omni_config = _from_pipeline_key("minicpmo_4_5")
    pipeline = _resolve_pipeline_or_skip("minicpmo_4_5")

    assert omni_config.pipeline_config is pipeline
    assert not hasattr(omni_config, "pipeline")
    assert "hf_config_predicate" in {f.name for f in fields(PipelineConfig)}
    assert omni_config.pipeline_config.hf_config_predicate is pipeline.hf_config_predicate


def test_from_pipeline_config_normalizes_stage_engine_extras_without_expanding_stage_deploy_config():
    assert not hasattr(StageDeployConfig, "model_config")
    assert not hasattr(StageDeployConfig, "parallel_config")

    stage = _from_pipeline_key("dreamzero", deploy_config_path="dreamzero_tp1_cfg2").stage_by_id(0)

    assert isinstance(stage, VllmOmniDiffusionStageConfig)
    assert stage.parallel_config.tensor_parallel_size == 1
    assert stage.parallel_config.cfg_parallel_size == 2
    assert stage.diffusion_config.model_config["default_robot_embodiment"] == "roboarena"


@pytest.mark.parametrize(
    ("engine_extras_yaml", "expected"),
    [
        pytest.param("", True, id="stage-value-absent"),
        pytest.param(
            """\
    engine_extras:
      enable_session_state_manager: null
""",
            True,
            id="stage-null-is-unset",
        ),
        pytest.param(
            """\
    engine_extras:
      enable_session_state_manager: false
""",
            False,
            id="stage-false-overrides-pipeline-true",
        ),
    ],
)
def test_session_state_manager_deploy_reaches_legacy_and_structured_configs(
    tmp_path,
    monkeypatch,
    engine_extras_yaml,
    expected,
):
    from vllm_omni.engine import stage_init_utils

    deploy_path = tmp_path / "dreamzero_session_state.yaml"
    deploy_path.write_text(
        f"""\
pipeline: dreamzero
async_chunk: false
enable_session_state_manager: true
stages:
  - stage_id: 0
{engine_extras_yaml}"""
    )
    pipeline = _resolve_pipeline_or_skip("dreamzero")
    deploy = load_deploy_config(deploy_path)
    legacy_stage = merge_pipeline_deploy(pipeline, deploy)[0].to_omegaconf()
    structured_stage = VllmOmniConfig.from_pipeline_config(pipeline, user_deploy_config=deploy).stage_by_id(0)
    monkeypatch.setattr(stage_init_utils.current_omni_platform, "get_device_count", lambda: 1)
    monkeypatch.setattr(OmniDiffusionConfig, "_resolve_master_port", lambda self: 29500)

    legacy_config = build_diffusion_config("unused", legacy_stage, extract_stage_metadata(legacy_stage))

    assert deploy.enable_session_state_manager is True
    assert legacy_config.enable_session_state_manager is expected
    assert structured_stage.diffusion_config.enable_session_state_manager is expected


def test_from_pipeline_config_applies_cli_overrides_without_stage_config_runtime_bridge():
    omni_config = _from_pipeline_key(
        "qwen3_tts",
        cli_overrides={
            "stage_0_max_num_seqs": 7,
            "stage_1_tensor_parallel_size": 2,
        },
    )

    stage0 = omni_config.stage_by_id(0)
    stage1 = omni_config.stage_by_id(1)

    assert stage0.scheduler_config.max_num_seqs == 7
    assert stage1.parallel_config.tensor_parallel_size == 2
    assert stage1.runtime_config.num_gpus == stage1.parallel_config.world_size


def test_runtime_num_gpus_is_derived_from_parallel_world_size():
    omni_config = _from_pipeline_key("hunyuan_image3_dit")
    stage = omni_config.stage_by_id(0)

    assert stage.parallel_config.tensor_parallel_size == 4
    assert stage.parallel_config.world_size == 4
    assert stage.runtime_config.num_gpus == 4


def test_runtime_num_gpus_ignores_stale_runtime_override():
    omni_config = _from_pipeline_key(
        "hunyuan_image3_dit",
        cli_overrides={
            "stage_0_num_gpus": 1,
        },
    )
    stage = omni_config.stage_by_id(0)

    assert stage.parallel_config.world_size == 4
    assert stage.runtime_config.num_gpus == 4


def test_from_pipeline_config_does_not_route_server_cli_keys_to_diffusion_stage():
    omni_config = _from_pipeline_key(
        "dreamzero",
        deploy_config_path="dreamzero_tp1_cfg2",
        cli_overrides={
            "host": "0.0.0.0",
            "port": 8000,
            "api_key": "secret",
            "max_generated_image_size": 1048576,
            "tts_max_instructions_length": 1000,
            "stage_0_host": "127.0.0.1",
            "stage_0_port": 23456,
        },
    )

    stage = omni_config.stage_by_id(0)

    assert isinstance(stage, VllmOmniDiffusionStageConfig)
    assert stage.diffusion_config.host == "127.0.0.1"
    assert stage.diffusion_config.port == 23456
    assert not hasattr(stage.diffusion_config, "api_key")
    assert not hasattr(stage.diffusion_config, "max_generated_image_size")
    assert not hasattr(stage.diffusion_config, "tts_max_instructions_length")


def test_pipeline_deploy_cli_fields_reuse_legacy_pipeline_wide_engine_fields():
    assert omni_config_module._PIPELINE_DEPLOY_CLI_FIELDS is PIPELINE_WIDE_ENGINE_FIELDS
    assert "active_stream_window" in omni_config_module._PIPELINE_DEPLOY_CLI_FIELDS
    assert "custom_voice_dir" in omni_config_module._PIPELINE_DEPLOY_CLI_FIELDS


def test_pipeline_wide_model_fields_are_retained_on_structured_stage_configs(tmp_path):
    custom_voice_dir = tmp_path / "voices"
    omni_config = _from_pipeline_key(
        "qwen3_tts",
        cli_overrides={
            "active_stream_window": 2,
            "custom_voice_dir": str(custom_voice_dir),
        },
    )

    assert {stage.model_config.active_stream_window for stage in omni_config.stage_configs} == {2}
    assert {stage.model_config.custom_voice_dir for stage in omni_config.stage_configs} == {str(custom_voice_dir)}


def test_stage_deploy_engine_fields_reuse_legacy_stage_deploy_fields():
    assert omni_config_module._STAGE_DEPLOY_ENGINE_FIELDS == tuple(_STAGE_DEPLOY_FIELDS)
    assert "tensor_parallel_size" in omni_config_module._STAGE_DEPLOY_ENGINE_FIELDS
    assert "stage_id" not in omni_config_module._STAGE_DEPLOY_ENGINE_FIELDS


def test_public_config_exports_use_stage_specific_sub_config_names():
    import vllm_omni.config as config_pkg

    generic_names = {
        "CacheConfig",
        "ConnectorConfig",
        "LoadConfig",
        "ModelConfig",
        "OrchestratorConfig",
        "ParallelConfig",
        "RuntimeConfig",
        "SchedulerConfig",
    }

    assert generic_names.isdisjoint(config_pkg.__all__)
    assert {
        "OmniStageCacheConfig",
        "OmniStageConnectorConfig",
        "OmniStageDiffusionParallelConfig",
        "OmniStageLoadConfig",
        "OmniStageModelConfig",
        "VllmOmniOrchestratorConfig",
        "OmniStageParallelConfig",
        "OmniStageRuntimeConfig",
        "OmniStageSchedulerConfig",
        "StageConfigType",
    }.issubset(config_pkg.__all__)


def test_from_pipeline_config_keeps_worker_backend_separate_from_distributed_executor_backend():
    omni_config = _from_pipeline_key("dreamzero", deploy_config_path="dreamzero_tp1_cfg2")

    stage = omni_config.stage_by_id(0)
    assert isinstance(stage, VllmOmniDiffusionStageConfig)
    assert stage.diffusion_config.distributed_executor_backend == "mp"
    assert omni_config.orchestrator_config.worker_backend == "multi_process"


def test_from_pipeline_config_maps_orchestrator_cli_overrides():
    omni_config = _from_pipeline_key(
        "qwen3_tts",
        cli_overrides={
            "stage_init_timeout": 1200,
            "init_timeout": 1800,
            "worker_backend": "ray",
            "ray_address": "ray://127.0.0.1:10001",
            "omni_master_address": "127.0.0.1",
            "omni_master_port": 12345,
            "omni_dp_size_local": 2,
            "omni_lb_policy": "round_robin",
            "omni_heartbeat_timeout": 9.5,
            "batch_timeout": 3,
        },
    )

    orchestrator_config = omni_config.orchestrator_config
    assert orchestrator_config.stage_init_timeout == 1200
    assert orchestrator_config.init_timeout == 1800
    assert orchestrator_config.worker_backend == "ray"
    assert orchestrator_config.ray_address == "ray://127.0.0.1:10001"
    assert orchestrator_config.omni_master_address == "127.0.0.1"
    assert orchestrator_config.omni_master_port == 12345
    assert orchestrator_config.omni_dp_size_local == 2
    assert orchestrator_config.omni_lb_policy == "round_robin"
    assert orchestrator_config.omni_heartbeat_timeout == 9.5
    assert orchestrator_config.batch_timeout == 3


def test_from_pipeline_config_records_loaded_deploy_path_on_orchestrator_config():
    omni_config = _from_pipeline_key("dreamzero", deploy_config_path="dreamzero_tp1_cfg2")

    assert omni_config.pipeline_config.model_type == "dreamzero"
    assert omni_config.orchestrator_config.deploy_config_path == str(_DEPLOY_DIR / "dreamzero_tp1_cfg2.yaml")


def test_from_pipeline_config_dispatches_async_chunk_processors_without_mutating_topology():
    pipeline = _resolve_pipeline_or_skip("qwen3_tts")

    async_config = _from_pipeline_key("qwen3_tts")
    assert async_config.stage_by_id(0).custom_process_next_stage_input_func.endswith("talker2code2wav_async_chunk")
    assert async_config.stage_by_id(1).custom_process_input_func is None

    sync_config = _from_pipeline_key("qwen3_tts", cli_overrides={"async_chunk": False})
    assert sync_config.stage_by_id(0).custom_process_next_stage_input_func.endswith("talker2code2wav_full_payload")
    assert sync_config.stage_by_id(1).custom_process_input_func.endswith("talker2code2wav_token_only")

    assert pipeline.get_stage(0).custom_process_next_stage_input_func.endswith("talker2code2wav_full_payload")
    assert pipeline.get_stage(1).custom_process_input_func is None


def test_vllm_omni_stage_config_public_fields_use_typed_stage_realizations():
    assert not hasattr(BaseVllmOmniStageConfig, "from_stage_config")
    assert not hasattr(BaseVllmOmniStageConfig, "to_legacy_stage_config")

    public_fields = {f.name for f in fields(BaseVllmOmniStageConfig)}

    assert public_fields == {
        "stage_pipeline_config",
        "model_config",
        "load_config",
        "cache_config",
        "scheduler_config",
        "connector_config",
        "runtime_config",
        "parallel_config",
        "quantization_config",
    }
    assert "diffusion_config" not in public_fields
    assert {f.name for f in fields(VllmOmniDiffusionStageConfig)} == public_fields | {"diffusion_config"}
    assert {f.name for f in fields(VllmOmniARStageConfig)} == public_fields
    assert {f.name for f in fields(VllmOmniGenerationStageConfig)} == public_fields


def test_runtime_config_fields_match_rfc_runtime_scope():
    assert {f.name for f in fields(OmniStageRuntimeConfig)} == {
        "devices",
        "num_replicas",
        "env",
        "num_gpus",
        "log_level",
        "log_stats",
        "profiler_config",
    }


def test_sub_config_fields_match_rfc_scopes():
    assert {f.name for f in fields(OmniStageModelConfig)} == {
        "active_stream_window",
        "enable_sleep_mode",
        "default_sampling_params",
        "subtalker_sampling_params",
        "has_sampling_extra_args",
        "custom_voice_dir",
        "task_type",
        "codec_frame_rate_hz",
        "enforce_eager",
        "enable_flashinfer_autotune",
        "compilation_config",
        "enable_multithread_weight_load",
        "num_weight_load_threads",
        "disable_autocast",
    }
    assert {f.name for f in fields(OmniStageLoadConfig)} == {
        "load_format",
        "tokenizer_mode",
        "config_format",
        "skip_mm_profiling",
    }
    assert {f.name for f in fields(OmniStageCacheConfig)} == {
        "gpu_memory_utilization",
        "enable_prefix_caching",
        "disable_hybrid_kv_cache_manager",
        "mm_processor_cache_gb",
    }
    assert {f.name for f in fields(OmniStageSchedulerConfig)} == {
        "max_num_seqs",
        "max_num_batched_tokens",
        "max_model_len",
        "enable_chunked_prefill",
        "async_scheduling",
    }
    assert {f.name for f in fields(OmniStageConnectorConfig)} == {
        "stage_connector",
        "output_connectors",
        "input_connectors",
    }
    assert {f.name for f in fields(OmniStageParallelConfig)} == {
        "pipeline_parallel_size",
        "data_parallel_size",
        "tensor_parallel_size",
        "enable_expert_parallel",
        "world_size",
    }
    assert {f.name for f in fields(OmniStageDiffusionParallelConfig)} == {
        "pipeline_parallel_size",
        "data_parallel_size",
        "tensor_parallel_size",
        "sequence_parallel_size",
        "ulysses_degree",
        "ring_degree",
        "allgather_degree",
        "ulysses_mode",
        "cfg_parallel_size",
        "vae_patch_parallel_size",
        "vae_parallel_mode",
        "use_hsdp",
        "mask_sp_padding",
        "hsdp_shard_size",
        "hsdp_replicate_size",
        "enable_expert_parallel",
        "world_size",
    }


def test_diffusion_parallel_config_fields_cover_legacy_surface():
    from vllm_omni.diffusion.data import DiffusionParallelConfig

    legacy_fields = {f.name for f in fields(DiffusionParallelConfig)}
    structured_fields = {f.name for f in fields(OmniStageDiffusionParallelConfig)}
    expected_upstream_fields = {"mask_sp_padding"}

    assert legacy_fields | expected_upstream_fields <= structured_fields
    assert structured_fields - legacy_fields - expected_upstream_fields == {"world_size"}


def test_diffusion_parallel_config_keeps_current_diffusion_parallel_surface():
    cfg = OmniStageDiffusionParallelConfig(
        pipeline_parallel_size=2,
        data_parallel_size=3,
        tensor_parallel_size=4,
        cfg_parallel_size=3,
        mask_sp_padding=True,
    )

    assert cfg.pipeline_parallel_size == 2
    assert cfg.data_parallel_size == 3
    assert cfg.cfg_parallel_size == 3
    assert cfg.mask_sp_padding is True
    assert cfg.world_size == 72


def test_parallel_config_derived_fields_are_not_init_inputs():
    with pytest.raises(ValidationError):
        OmniStageParallelConfig(world_size=4)

    with pytest.raises(ValidationError):
        OmniStageDiffusionParallelConfig(world_size=4)

    with pytest.raises(ValidationError):
        OmniStageDiffusionParallelConfig(sequence_parallel_size=2)


def test_diffusion_parallel_config_matches_diffusion_parallel_world_size_for_vae_patch_parallel():
    cfg = OmniStageDiffusionParallelConfig(
        tensor_parallel_size=2,
        cfg_parallel_size=2,
        vae_patch_parallel_size=4,
    )

    assert cfg.vae_patch_parallel_size == 4
    assert cfg.world_size == 4


def test_diffusion_parallel_config_supports_diffusion_hsdp_auto_sharding():
    cfg = OmniStageDiffusionParallelConfig(
        pipeline_parallel_size=2,
        ulysses_degree=2,
        use_hsdp=True,
        hsdp_shard_size=-1,
        hsdp_replicate_size=2,
    )

    assert cfg.hsdp_shard_size == 2
    assert cfg.world_size == 4


def test_diffusion_parallel_config_rejects_hsdp_with_tp_or_dp():
    with pytest.raises(ValueError, match="cannot be used with TP or DP"):
        OmniStageDiffusionParallelConfig(tensor_parallel_size=2, use_hsdp=True, hsdp_shard_size=2)

    with pytest.raises(ValueError, match="cannot be used with TP or DP"):
        OmniStageDiffusionParallelConfig(data_parallel_size=2, use_hsdp=True, hsdp_shard_size=2)


def test_from_pipeline_config_preserves_legacy_pp_dp_for_world_size():
    cfg = _from_pipeline_key("hunyuan_image3_dit").stage_by_id(0).parallel_config

    assert cfg.pipeline_parallel_size == 1
    assert cfg.data_parallel_size == 1
    assert cfg.tensor_parallel_size == 4
    assert cfg.world_size == 4


def test_from_pipeline_config_validates_sequence_parallel_size_from_degrees():
    stage = _build_dreamzero_stage(
        engine_extras={
            "parallel_config": {
                "sequence_parallel_size": "6",
                "ulysses_degree": "2",
                "ring_degree": "3",
            }
        }
    )

    assert stage.parallel_config.sequence_parallel_size == 6
    assert stage.parallel_config.world_size == 6


def test_from_pipeline_config_derives_sequence_parallel_size_from_allgather_degree(tmp_path):
    deploy_path = tmp_path / "dreamzero_allgather_parallel.yaml"
    deploy_path.write_text(
        "\n".join(
            [
                "pipeline: dreamzero",
                "async_chunk: false",
                "stages:",
                "  - stage_id: 0",
                "    parallel_config:",
                "      sequence_parallel_size: 99",
                "      allgather_degree: 2",
            ]
        )
    )

    stage = _from_pipeline_key("dreamzero", deploy_config_path=str(deploy_path)).stage_by_id(0)

    assert isinstance(stage, VllmOmniDiffusionStageConfig)
    assert stage.parallel_config.allgather_degree == 2
    assert stage.parallel_config.sequence_parallel_size == 2
    assert stage.parallel_config.world_size == 2


def test_diffusion_parallel_config_rejects_cfg_parallel_size_outside_current_bound():
    with pytest.raises(ValidationError):
        OmniStageDiffusionParallelConfig(cfg_parallel_size=4)


def test_diffusion_parallel_config_rejects_allgather_with_ulysses_or_ring():
    with pytest.raises(ValidationError):
        OmniStageDiffusionParallelConfig(allgather_degree=2, ulysses_degree=2)


def test_stage_realizations_use_stage_specific_parallel_config_types():
    qwen_config = _from_pipeline_key("qwen3_tts")
    ar_stage = qwen_config.stage_by_id(0)
    generation_stage = qwen_config.stage_by_id(1)
    diffusion_stage = _from_pipeline_key("hunyuan_image3_dit").stage_by_id(0)

    assert isinstance(ar_stage, VllmOmniARStageConfig)
    assert type(ar_stage.parallel_config) is OmniStageParallelConfig
    assert not hasattr(ar_stage.parallel_config, "cfg_parallel_size")
    assert not hasattr(ar_stage.parallel_config, "sequence_parallel_size")
    assert not hasattr(ar_stage.parallel_config, "ulysses_degree")

    assert isinstance(generation_stage, VllmOmniGenerationStageConfig)
    assert type(generation_stage.parallel_config) is OmniStageParallelConfig
    assert not hasattr(generation_stage.parallel_config, "cfg_parallel_size")
    assert not hasattr(generation_stage.parallel_config, "sequence_parallel_size")
    assert not hasattr(generation_stage.parallel_config, "ulysses_degree")

    assert isinstance(diffusion_stage, VllmOmniDiffusionStageConfig)
    assert isinstance(diffusion_stage.parallel_config, OmniStageDiffusionParallelConfig)
    assert diffusion_stage.parallel_config.cfg_parallel_size == 1
    assert diffusion_stage.parallel_config.sequence_parallel_size == 1
    assert diffusion_stage.parallel_config.ulysses_degree == 1


def test_from_pipeline_config_preserves_diffusion_parallel_mask_sp_padding(tmp_path):
    deploy_path = tmp_path / "dreamzero_mask_sp_padding.yaml"
    deploy_path.write_text(
        "\n".join(
            [
                "pipeline: dreamzero",
                "async_chunk: false",
                "stages:",
                "  - stage_id: 0",
                "    parallel_config:",
                "      mask_sp_padding: true",
            ]
        )
    )

    stage = _from_pipeline_key("dreamzero", deploy_config_path=str(deploy_path)).stage_by_id(0)

    assert isinstance(stage, VllmOmniDiffusionStageConfig)
    assert stage.parallel_config.mask_sp_padding is True


def test_from_pipeline_config_routes_regional_compile_dynamic(tmp_path):
    deploy_path = tmp_path / "dreamzero_compile.yaml"
    deploy_path.write_text(
        "\n".join(
            [
                "pipeline: dreamzero",
                "async_chunk: false",
                "stages:",
                "  - stage_id: 0",
                "    diffusion_compile_granularity: regional",
                "    diffusion_compile_dynamic: false",
            ]
        )
    )

    configured_stage = _from_pipeline_key("dreamzero", deploy_config_path=str(deploy_path)).stage_by_id(0)
    overridden_stage = _from_pipeline_key(
        "dreamzero",
        deploy_config_path=str(deploy_path),
        cli_overrides={
            "diffusion_compile_granularity": "full",
            "diffusion_compile_dynamic": True,
        },
    ).stage_by_id(0)

    assert configured_stage.diffusion_config.diffusion_compile_granularity == "regional"
    assert configured_stage.diffusion_config.diffusion_compile_dynamic is False
    assert overridden_stage.diffusion_config.diffusion_compile_granularity == "full"
    assert overridden_stage.diffusion_config.diffusion_compile_dynamic is True


def test_structured_diffusion_config_rejects_non_boolean_compile_dynamic():
    with pytest.raises(ValidationError, match="diffusion_compile_dynamic"):
        omni_config_module._DiffusionConfigProjection(diffusion_compile_dynamic="false")


def test_structured_diffusion_config_rejects_invalid_compile_granularity():
    with pytest.raises(ValidationError, match="diffusion_compile_granularity"):
        omni_config_module._DiffusionConfigProjection(diffusion_compile_granularity="block")


def test_from_pipeline_config_matches_stage_config_to_omegaconf_behavior_for_representative_stage():
    pipeline = _resolve_pipeline_or_skip("qwen3_tts")
    legacy_stage = merge_pipeline_deploy(pipeline, _load_default_deploy(pipeline))[0]
    omega_stage = legacy_stage.to_omegaconf()
    omni_stage = _from_pipeline_key("qwen3_tts").stage_by_id(legacy_stage.stage_id)

    assert omega_stage.stage_id == omni_stage.stage_id
    assert omega_stage.stage_type == omni_stage.stage_type.value
    assert omega_stage.engine_input_source == omni_stage.input_sources
    assert omega_stage.final_output == omni_stage.final_output
    assert omega_stage.final_output_type == omni_stage.final_output_type
    assert omega_stage.is_comprehension == omni_stage.is_comprehension
    assert omega_stage.engine_args.model_stage == omni_stage.model_stage
    assert omega_stage.engine_args.worker_type == omni_stage.worker_type
    assert omega_stage.engine_args.scheduler_cls == omni_stage.scheduler_cls
    assert omega_stage.runtime.process is True
    assert omega_stage.runtime.requires_multimodal_data == omni_stage.requires_multimodal_data


def test_from_pipeline_config_uses_hf_config_for_callable_resolver():
    hf_config = Qwen3OmniMoeConfig()
    hf_config.enable_audio_output = False

    omni_config = _from_pipeline_key("qwen3_omni_moe", hf_config=hf_config)

    assert omni_config.pipeline_config.model_type == "qwen3_omni_moe_thinker_only"
    assert len(omni_config.stage_configs) == 1
    assert omni_config.orchestrator_config.deploy_config_path is None

    thinker = omni_config.stage_configs[0]
    assert thinker.model_stage == "thinker"
    assert thinker.model_config.default_sampling_params == {"detokenize": True}


def test_from_pipeline_config_accepts_pre_resolved_pipeline():
    resolved_pipeline = PipelineConfig(model_type="callable_resolved_variant")

    omni_config = VllmOmniConfig.from_pipeline_config(resolved_pipeline)

    assert omni_config.pipeline_config is resolved_pipeline


def test_from_pipeline_config_prefers_loaded_user_deploy_config(monkeypatch):
    pipeline = _resolve_pipeline_or_skip("qwen3_tts")
    user_deploy_config = DeployConfig(
        stages=[StageDeployConfig(stage_id=0, max_num_seqs=7)],
    )
    monkeypatch.setattr(
        omni_config_module,
        "load_deploy_config",
        lambda _path: pytest.fail("default deploy config should not be loaded"),
    )

    omni_config = VllmOmniConfig.from_pipeline_config(
        pipeline,
        user_deploy_config=user_deploy_config,
    )

    assert omni_config.stage_by_id(0).scheduler_config.max_num_seqs == 7


def test_from_pipeline_config_default_deploy_name_ignores_cwd(monkeypatch, tmp_path):
    default_name = "pipeline_default.yaml"
    (tmp_path / default_name).write_text("stages: []\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    pipeline = PipelineConfig(
        model_type="pipeline_with_default",
        default_deploy_config_name=default_name,
    )
    loaded_paths = []

    def _load_deploy_config(path):
        loaded_paths.append(Path(path))
        return DeployConfig()

    monkeypatch.setattr(omni_config_module, "load_deploy_config", _load_deploy_config)

    omni_config = VllmOmniConfig.from_pipeline_config(pipeline)

    assert omni_config.orchestrator_config.deploy_config_path == str(_DEPLOY_DIR / default_name)
    assert loaded_paths == [_DEPLOY_DIR / default_name]


def test_from_pipeline_config_uses_resolved_deploy_pipeline():
    deploy_path = get_deploy_config_path("aura_omni.yaml")
    pipeline = _resolve_pipeline_or_skip("aura_omni")

    omni_config = VllmOmniConfig.from_pipeline_config(
        pipeline,
        deploy_config_path=str(deploy_path),
    )

    assert omni_config.pipeline_config.model_type == "aura_omni"
    assert [stage.model_stage for stage in omni_config.stage_configs] == [
        "asr",
        "aura",
        "qwen3_tts",
        "code2wav",
    ]


def test_from_pipeline_config_matches_to_omegaconf_diffusion_parallel_config():
    pipeline = _resolve_pipeline_or_skip("hunyuan_image3_dit")
    legacy_stage = merge_pipeline_deploy(pipeline, _load_default_deploy(pipeline))[0]
    omega_stage = legacy_stage.to_omegaconf()
    omni_stage = _from_pipeline_key("hunyuan_image3_dit").stage_by_id(legacy_stage.stage_id)

    assert (
        omega_stage.engine_args.parallel_config.pipeline_parallel_size
        == omni_stage.parallel_config.pipeline_parallel_size
    )
    assert omega_stage.engine_args.parallel_config.data_parallel_size == omni_stage.parallel_config.data_parallel_size
    assert (
        omega_stage.engine_args.parallel_config.tensor_parallel_size == omni_stage.parallel_config.tensor_parallel_size
    )
    assert (
        omega_stage.engine_args.parallel_config.sequence_parallel_size
        == omni_stage.parallel_config.sequence_parallel_size
    )
    assert omega_stage.engine_args.parallel_config.cfg_parallel_size == omni_stage.parallel_config.cfg_parallel_size
    assert (
        omega_stage.engine_args.parallel_config.vae_patch_parallel_size
        == omni_stage.parallel_config.vae_patch_parallel_size
    )


def test_from_pipeline_config_matches_build_engine_args_dict_behavior_for_representative_stage(monkeypatch):
    from vllm_omni.engine import stage_init_utils

    monkeypatch.setattr(stage_init_utils, "resolve_worker_cls", lambda engine_args: None)
    pipeline = _resolve_pipeline_or_skip("qwen3_tts")
    legacy_stage = merge_pipeline_deploy(pipeline, _load_default_deploy(pipeline))[0]
    omega_stage = legacy_stage.to_omegaconf()
    legacy_engine_args = build_engine_args_dict(
        omega_stage,
        model="/tmp/qwen3-tts",
        stage_connector_spec={"name": "SharedMemoryConnector", "extra": {}},
    )
    omni_stage = _from_pipeline_key("qwen3_tts").stage_by_id(legacy_stage.stage_id)

    assert legacy_engine_args["model"] == "/tmp/qwen3-tts"
    assert legacy_engine_args["stage_id"] == omni_stage.stage_id
    assert legacy_engine_args["model_stage"] == omni_stage.model_stage
    assert legacy_engine_args["worker_type"] == omni_stage.worker_type
    assert legacy_engine_args["scheduler_cls"] == omni_stage.scheduler_cls
    assert legacy_engine_args["stage_connector_spec"] == {"name": "SharedMemoryConnector", "extra": {}}
    assert legacy_engine_args["has_sampling_extra_args"] == bool(
        (omni_stage.model_config.default_sampling_params or {}).get("extra_args")
    )
    assert omni_stage.model_config.has_sampling_extra_args == legacy_engine_args["has_sampling_extra_args"]


def test_from_pipeline_config_derives_has_sampling_extra_args_from_stage_defaults():
    stage = _from_pipeline_key("voxtral_tts").stage_by_id(0)

    assert (stage.model_config.default_sampling_params or {}).get("extra_args")
    assert stage.model_config.has_sampling_extra_args is True


def test_diffusion_config_preserves_existing_coercion_hooks():
    import torch

    from vllm_omni.diffusion.data import AttentionConfig, DiffusionCacheConfig

    cfg = omni_config_module._DiffusionConfigProjection(
        dtype="float32",
        cache_config={"rel_l1_thresh": 0.3},
        diffusion_attention_config={"default": "flash_attn"},
        diffusion_kv_cache_skip_steps="0-2,4",
        diffusion_kv_cache_skip_layers=[1, 3],
    )

    assert cfg.dtype is torch.float32
    assert isinstance(cfg.cache_config, DiffusionCacheConfig)
    assert isinstance(cfg.diffusion_attention_config, AttentionConfig)
    assert cfg.diffusion_attention_config.default.backend == "flash_attn"
    assert cfg.diffusion_kv_cache_skip_step_indices == {0, 1, 2, 4}
    assert cfg.diffusion_kv_cache_skip_layer_indices == {1, 3}
    assert cfg.max_cpu_loras == 1


def test_diffusion_config_from_kwargs_reuses_legacy_normalization(monkeypatch):
    monkeypatch.setenv("DIFFUSION_CACHE_BACKEND", "TEA_CACHE")

    with pytest.warns(FutureWarning) as warnings:
        cfg = omni_config_module._DiffusionConfigProjection.from_kwargs(
            diffusion_attention_backend="flash_attn",
            kv_cache_dtype="fp8",
            diffusion_kv_cache_dtype=None,
            kv_cache_skip_steps="0-1",
            kv_cache_skip_layers=[2],
            static_lora_scale=0.25,
            diffusers_load_kwargs=None,
            diffusers_call_kwargs=None,
        )

    assert len(warnings) == 1
    assert cfg.diffusion_attention_config.default.backend == "flash_attn"
    assert cfg.diffusion_kv_cache_dtype == "fp8"
    assert cfg.diffusion_kv_cache_skip_step_indices == {0, 1}
    assert cfg.diffusion_kv_cache_skip_layer_indices == {2}
    assert cfg.lora_scale == 0.25
    assert cfg.cache_backend == "tea_cache"
    assert cfg.diffusers_load_kwargs == {}
    assert cfg.diffusers_call_kwargs == {}


def test_from_pipeline_config_normalizes_diffusion_config_aliases_from_engine_args():
    stage = _build_dreamzero_stage(
        max_num_seqs=2,
        vae_parallel_mode="spatial_shard_height",
        diffusion_attention_backend="flash_attn",
        diffusion_quantization_config={"method": "example_quant"},
        auxiliary_text_encoder="example/encoder",
        engine_extras={
            "engine_backend": "custom.engine.Backend",
            "model_config": {"default_robot_embodiment": "test"},
            "diffusion_model_runner_cls": "example.Runner",
        },
        cli_overrides={"seed": 7, "kv_cache_dtype": "fp8"},
    )

    assert stage.shared_engine_args == {"seed": 7, "kv_cache_dtype": "fp8"}
    assert stage.scheduler_config.max_num_seqs == 2
    assert stage.parallel_config.vae_parallel_mode == "spatial_shard_height"
    assert stage.diffusion_config.engine_backend == "custom.engine.Backend"
    assert stage.diffusion_config.model_config["default_robot_embodiment"] == "test"
    assert stage.diffusion_config.diffusion_model_runner_cls == "example.Runner"
    assert stage.diffusion_config.diffusion_attention_config.default.backend == "flash_attn"
    assert stage.diffusion_config.quantization_config == {"method": "example_quant"}
    assert stage.diffusion_config.extras["auxiliary_text_encoder"] == "example/encoder"


def test_diffusion_attention_backend_keeps_per_role_config():
    cfg = omni_config_module._DiffusionConfigProjection.from_kwargs(
        diffusion_attention_backend="flash_attn",
        diffusion_attention_config={
            "per_role": {
                "cross": {"backend": "torch_sdpa"},
            }
        },
    )

    assert cfg.diffusion_attention_config.default.backend == "flash_attn"
    assert cfg.diffusion_attention_config.per_role["cross"].backend == "torch_sdpa"


@pytest.mark.parametrize(
    ("factory", "error_type", "field_name"),
    [
        (omni_config_module._DiffusionConfigProjection.from_kwargs, ValidationError, "enable_sleep_mod"),
        (OmniDiffusionConfig.from_kwargs, ValueError, "enable_sleep_mod"),
    ],
)
def test_diffusion_config_entrypoints_reject_unowned_fields(factory, error_type, field_name):
    with pytest.raises(error_type, match=field_name):
        factory(**{field_name: "example"})


@pytest.mark.parametrize(
    ("kwargs", "field_name", "expected", "warning_count"),
    [
        ({"kv_cache_dtype": "fp8", "diffusion_kv_cache_dtype": None}, "diffusion_kv_cache_dtype", "fp8", 0),
        ({"quantization": {"method": "example"}}, "quantization_config", {"method": "example"}, 1),
    ],
)
def test_omni_diffusion_config_normalizes_direct_aliases(monkeypatch, kwargs, field_name, expected, warning_count):
    from vllm_omni.diffusion import data as diffusion_data

    monkeypatch.setattr(diffusion_data, "build_quant_config", lambda config: config)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        config = OmniDiffusionConfig.from_kwargs(**kwargs)

    assert getattr(config, field_name) == expected
    assert sum(item.category is FutureWarning for item in caught) == warning_count


@pytest.mark.parametrize(
    ("legacy_name", "canonical_name", "value"),
    [
        ("static_lora_scale", "lora_scale", 0.5),
        ("quantization", "quantization_config", "fp8"),
    ],
)
def test_omni_diffusion_config_rejects_deprecated_alias_conflicts(legacy_name, canonical_name, value):
    with pytest.raises(ValueError, match=rf"{legacy_name}.*{canonical_name}"):
        OmniDiffusionConfig.from_kwargs(**{legacy_name: value, canonical_name: value})


def test_startup_diffusion_payload_keeps_active_vllm_kv_cache_dtype_out_of_diffusion_config():
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        payload = _strict_diffusion_config_kwargs({"kv_cache_dtype": "fp8", "diffusion_kv_cache_dtype": "auto"})

    assert "kv_cache_dtype" not in payload
    assert payload["diffusion_kv_cache_dtype"] == "auto"


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({"static_lora_scale": 0.5, "lora_scale": None}, {"lora_scale": 0.5}),
        (
            {"vae_parallel_mode": "spatial_shard_height"},
            {"parallel_config": {"vae_parallel_mode": "spatial_shard_height"}},
        ),
        (
            {
                "diffusion_quantization_config": {"method": "example_quant"},
                "auxiliary_text_encoder": "example/encoder",
            },
            {
                "quantization_config": {"method": "example_quant"},
                "extras": {"auxiliary_text_encoder": "example/encoder"},
            },
        ),
    ],
)
def test_startup_diffusion_payload_normalizes_owned_fields(kwargs, expected):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        payload = _strict_diffusion_config_kwargs({"stage_id": 2, **kwargs})

    for name, value in expected.items():
        assert payload[name] == value


def test_startup_diffusion_payload_treats_nested_none_as_unset():
    payload = _strict_diffusion_config_kwargs(
        {
            "parallel_config": {"vae_parallel_mode": None},
            "vae_parallel_mode": "spatial_shard_height",
        }
    )

    assert payload["parallel_config"]["vae_parallel_mode"] == "spatial_shard_height"


def test_startup_diffusion_payload_treats_compatibility_none_as_unset():
    payload = _strict_diffusion_config_kwargs(
        {
            "auxiliary_text_encoder": "example/encoder",
            "extras": {"auxiliary_text_encoder": None},
        }
    )

    assert payload["extras"]["auxiliary_text_encoder"] == "example/encoder"


def test_legacy_diffusion_deploy_flat_parallel_fields_drive_preflight_and_config(tmp_path, monkeypatch):
    from vllm_omni.engine import stage_init_utils

    deploy_path = tmp_path / "dreamzero_flat_parallel.yaml"
    deploy_path.write_text(
        """\
pipeline: dreamzero
async_chunk: false
stages:
  - stage_id: 0
    ulysses_degree: 2
    sequence_parallel_size: 2
    vae_parallel_mode: spatial_shard_height
"""
    )
    pipeline = _resolve_pipeline_or_skip("dreamzero")
    deploy = load_deploy_config(deploy_path)
    stage = merge_pipeline_deploy(pipeline, deploy)[0].to_omegaconf()
    metadata = extract_stage_metadata(stage)
    monkeypatch.setattr(stage_init_utils.current_omni_platform, "get_device_count", lambda: 2)

    config = build_diffusion_config("unused", stage, metadata)

    assert stage.engine_args.parallel_config.ulysses_degree == 2
    assert stage.engine_args.parallel_config.vae_parallel_mode == "spatial_shard_height"
    assert get_stage_devices_per_replica(stage) == 2
    assert config.parallel_config.ulysses_degree == 2
    assert config.parallel_config.vae_parallel_mode == "spatial_shard_height"


def test_explicit_sequence_parallel_size_conflict_has_legacy_structured_parity():
    pipeline = _resolve_pipeline_or_skip("dreamzero")
    deploy = DeployConfig(
        async_chunk=False,
        stages=[
            StageDeployConfig(
                stage_id=0,
                ulysses_degree=2,
                engine_extras={"parallel_config": {"sequence_parallel_size": 1}},
            )
        ],
    )
    legacy_stage = merge_pipeline_deploy(pipeline, deploy)[0].to_omegaconf()

    with pytest.raises(ValueError, match="Sequence parallel size"):
        build_diffusion_config("unused", legacy_stage, extract_stage_metadata(legacy_stage))
    with pytest.raises(ValueError, match="Sequence parallel size"):
        VllmOmniConfig.from_pipeline_config(pipeline, user_deploy_config=deploy)


@pytest.mark.parametrize(
    (
        "stage_kwargs",
        "engine_extras",
        "cli_overrides",
        "field_name",
        "expected",
        "device_count",
    ),
    [
        pytest.param(
            {"vae_parallel_mode": "spatial_shard_height"},
            {"parallel_config": {"vae_parallel_mode": None}},
            {},
            "vae_parallel_mode",
            "spatial_shard_height",
            1,
            id="deploy-flat-fills-nested-none",
        ),
        pytest.param(
            {},
            {"parallel_config": {"vae_parallel_mode": "tile"}},
            {"stage_0_vae_parallel_mode": "spatial_shard_height"},
            "vae_parallel_mode",
            "spatial_shard_height",
            1,
            id="cli-flat-overrides-deploy-nested",
        ),
        pytest.param(
            {},
            {
                "parallel_config": {
                    "sequence_parallel_size": 99,
                    "allgather_degree": 2,
                }
            },
            {},
            "sequence_parallel_size",
            2,
            2,
            id="allgather-derives-sequence-size",
        ),
        pytest.param(
            {},
            {
                "parallel_config": {
                    "sequence_parallel_size": 2,
                    "allgather_degree": 2,
                }
            },
            {"stage_0_allgather_degree": 1},
            "sequence_parallel_size",
            1,
            1,
            id="cli-disables-allgather-and-rederives-sequence-size",
        ),
    ],
)
def test_diffusion_parallel_source_precedence_has_legacy_structured_parity(
    monkeypatch,
    stage_kwargs,
    engine_extras,
    cli_overrides,
    field_name,
    expected,
    device_count,
):
    from vllm_omni.config.config_factory import StageConfigFactory
    from vllm_omni.engine import stage_init_utils

    pipeline = _resolve_pipeline_or_skip("dreamzero")
    deploy = DeployConfig(
        async_chunk=False,
        stages=[
            StageDeployConfig(
                stage_id=0,
                engine_extras=engine_extras,
                **stage_kwargs,
            )
        ],
    )
    legacy_stage_config = merge_pipeline_deploy(pipeline, deploy)[0]
    legacy_stage_config.runtime_overrides = StageConfigFactory._merge_cli_overrides(
        legacy_stage_config,
        cli_overrides,
    )
    legacy_stage = legacy_stage_config.to_omegaconf()
    structured_stage = VllmOmniConfig.from_pipeline_config(
        pipeline,
        user_deploy_config=deploy,
        cli_overrides=cli_overrides,
    ).stage_by_id(0)
    monkeypatch.setattr(stage_init_utils.current_omni_platform, "get_device_count", lambda: device_count)
    monkeypatch.setattr(OmniDiffusionConfig, "_resolve_master_port", lambda self: 29500)

    legacy_config = build_diffusion_config("unused", legacy_stage, extract_stage_metadata(legacy_stage))

    assert getattr(legacy_config.parallel_config, field_name) == expected
    assert getattr(structured_stage.parallel_config, field_name) == expected


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"max_num_seqs": 4, "enable_sleep_mod": True}, r"stage 2.*enable_sleep_mod"),
        ({"enable_sleep_mod": None}, r"stage 2.*enable_sleep_mod"),
        ({"static_lora_scale": 0.5, "lora_scale": 0.7}, r"static_lora_scale.*lora_scale"),
        (
            {
                "diffusion_quantization_config": {"method": "example"},
                "quantization_config": {"method": "fallback"},
            },
            r"diffusion_quantization_config.*quantization_config",
        ),
        ({"max_generated_image_size": 1000}, "max_generated_image_size"),
        (
            {
                "auxiliary_text_encoder": "top-level/encoder",
                "extras": {"auxiliary_text_encoder": "extras/encoder"},
            },
            r"auxiliary_text_encoder.*top level.*extras",
        ),
    ],
)
def test_startup_diffusion_payload_rejects_invalid_sources(kwargs, match):
    with pytest.raises(ValueError, match=match):
        _strict_diffusion_config_kwargs({"stage_id": 2, **kwargs})


def test_from_pipeline_config_rejects_unknown_diffusion_yaml_field(tmp_path):
    deploy_path = tmp_path / "dreamzero_unknown_diffusion_field.yaml"
    deploy_path.write_text(
        """\
pipeline: dreamzero
async_chunk: false
stages:
  - stage_id: 0
    enable_sleep_mod: true
"""
    )

    with pytest.raises(ValueError, match=r"stage 0.*enable_sleep_mod"):
        _from_pipeline_key("dreamzero", deploy_config_path=str(deploy_path))


def test_default_diffusion_factory_builds_without_orchestrator_only_field(monkeypatch):
    from vllm_omni.config.yaml_util import create_config
    from vllm_omni.engine import stage_init_utils
    from vllm_omni.engine.async_omni_engine import AsyncOmniEngine

    stage = AsyncOmniEngine._create_default_diffusion_stage_cfg({})[0]
    stage_config = create_config([stage])[0]
    monkeypatch.setattr(stage_init_utils.current_omni_platform, "get_device_count", lambda: 1)

    config = build_diffusion_config("unused", stage_config, extract_stage_metadata(stage_config))

    assert "enable_ar_profiler" not in stage["engine_args"]
    assert config.stage_id == 0


def test_default_diffusion_factory_serializes_declared_passthrough_fields(monkeypatch):
    from vllm_omni.config.yaml_util import create_config
    from vllm_omni.engine import stage_init_utils
    from vllm_omni.engine.async_omni_engine import AsyncOmniEngine
    from vllm_omni.entrypoints.utils import _convert_dataclasses_to_dict

    stage = AsyncOmniEngine._create_default_diffusion_stage_cfg(
        {
            "diffusion_model_runner_cls": "example.Runner",
            "enable_session_state_manager": True,
            "tf_model_config": TransformerConfig(),
        }
    )[0]
    stage_config = create_config(_convert_dataclasses_to_dict([stage]))[0]
    monkeypatch.setattr(stage_init_utils.current_omni_platform, "get_device_count", lambda: 1)

    config = build_diffusion_config("unused", stage_config, extract_stage_metadata(stage_config))

    assert "tf_model_config" not in stage["engine_args"]
    assert config.diffusion_model_runner_cls == "example.Runner"
    assert config.enable_session_state_manager is True
    assert isinstance(config.tf_model_config, TransformerConfig)


@pytest.mark.parametrize(
    ("build_kwargs", "match"),
    [
        ({"cli_overrides": {"stage_0_enable_sleep_mod": True}}, r"stage 0.*enable_sleep_mod"),
        ({"engine_extras": {"enable_sleep_mod": None}}, r"stage 0.*enable_sleep_mod"),
    ],
)
def test_from_pipeline_config_rejects_invalid_diffusion_sources(build_kwargs, match):
    with pytest.raises(ValueError, match=match):
        _build_dreamzero_stage(**build_kwargs)


def test_from_pipeline_config_keeps_ar_engine_extras_permissive():
    pipeline = _resolve_pipeline_or_skip("qwen3_tts")
    deploy = DeployConfig(stages=[StageDeployConfig(stage_id=0, engine_extras={"custom_ar_option": True})])
    config = VllmOmniConfig.from_pipeline_config(pipeline, user_deploy_config=deploy)
    assert isinstance(config.stage_by_id(0), VllmOmniARStageConfig)


def test_diffusion_config_field_classification_covers_current_fields():
    classified_fields = (
        omni_config_module._DIFFUSION_SHARED_CONFIG_FIELDS
        | omni_config_module._DIFFUSION_RUNTIME_CONFIG_FIELDS
        | omni_config_module._DIFFUSION_ONLY_CONFIG_FIELDS
    )

    assert classified_fields == {f.name for f in fields(omni_config_module._DiffusionConfigProjection)}
    assert {
        "enable_prompt_embed_cache",
        "prompt_embed_cache_size",
        "diffusion_kv_cache_dtype",
    } <= omni_config_module._DIFFUSION_ONLY_CONFIG_FIELDS
    assert {
        "revision",
        "trust_remote_code",
        "distributed_executor_backend",
        "omni_kv_config",
    } <= omni_config_module._DIFFUSION_SHARED_CONFIG_FIELDS
    assert "prompt_file_path" in omni_config_module._DIFFUSION_RUNTIME_CONFIG_FIELDS


def test_diffusion_config_projection_keeps_mapping_quantization_config_serializable():
    quantization_config = {
        "method": "example_quant",
        "weights": "weights.bin",
    }

    cfg = omni_config_module._DiffusionConfigProjection.from_kwargs(quantization_config=quantization_config)

    assert cfg.quantization_config == quantization_config
