from __future__ import annotations

import json
import os
import subprocess
from contextlib import contextmanager
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import pytest
import requests
import torch
from PIL import Image

from tests.helpers.runtime import OmniServer, OmniServerParams


def pytest_addoption(parser):
    group = parser.getgroup("accuracy-e2e")
    group.addoption("--gebench-root", action="store", default=None, help="Local GEBench dataset root")
    group.addoption("--gedit-root", action="store", default=None, help="Local GEdit-Bench dataset root")
    group.addoption(
        "--gebench-model", action="store", default="Qwen/Qwen-Image-2512", help="Generate model for GEBench smoke"
    )
    group.addoption(
        "--gedit-model", action="store", default="Qwen/Qwen-Image-Edit", help="Generate model for GEdit-Bench smoke"
    )
    group.addoption(
        "--accuracy-judge-model",
        action="store",
        default="QuantTrio/Qwen3-VL-30B-A3B-Instruct-AWQ",
        help="Judge model path",
    )
    group.addoption("--accuracy-gpu", action="store", default="0", help="Single GPU id used sequentially")
    group.addoption("--gebench-port", action="store", type=int, default=8093, help="Generate port for GEBench")
    group.addoption("--gedit-port", action="store", type=int, default=8093, help="Generate port for GEdit-Bench")
    group.addoption(
        "--gebench-samples-per-type",
        action="store",
        type=int,
        default=10,
        help="Balanced sample count per GEBench type",
    )
    group.addoption(
        "--gedit-samples-per-group",
        action="store",
        type=int,
        default=20,
        help="Balanced sample count per GEdit task group",
    )
    group.addoption("--accuracy-workers", action="store", type=int, default=1, help="Worker count for accuracy benches")
    group.addoption(
        "--wan22-i2v-image-source",
        action="store",
        default=None,
        help="Image source for Wan2.2 I2V accuracy tests. Can be local path or remote URL.",
    )
    group.addoption(
        "--wan22-i2v-online-timeout-seconds",
        action="store",
        type=int,
        default=1200,
        help="Online serving timeout in seconds for Wan2.2 I2V accuracy tests.",
    )
    group.addoption(
        "--gebench-devices",
        action="store",
        default=None,
        help="CUDA_VISIBLE_DEVICES for GEBench generate server (e.g. '0,1,2,3'); TP size is derived from device count",
    )
    group.addoption(
        "--gebench-stage-overrides",
        action="store",
        default=None,
        help="JSON string passed to --stage-overrides for GEBench generate server",
    )
    group.addoption(
        "--gebench-extra-server-args",
        action="store",
        default=None,
        help='JSON array of extra CLI args for GEBench generate server (e.g. \'["--dtype","bfloat16"]\')',
    )
    group.addoption(
        "--gebench-num-inference-steps",
        action="store",
        type=int,
        default=8,
        help="Number of diffusion inference steps for GEBench generate",
    )


def _hf_cache_root() -> Path:
    return Path(os.environ.get("HF_HOME", "/root/.cache/huggingface"))


def _dataset_cache_dirs(dataset_id: str) -> list[Path]:
    cache_root = _hf_cache_root() / "hub" / f"datasets--{dataset_id.replace('/', '--')}" / "snapshots"
    if not cache_root.exists():
        return []
    return sorted(
        (path for path in cache_root.iterdir() if path.is_dir()), key=lambda path: path.stat().st_mtime, reverse=True
    )


def _ensure_dataset_snapshot(dataset_id: str) -> Path:
    candidates = _dataset_cache_dirs(dataset_id)
    if candidates:
        return candidates[0]

    subprocess.run(
        ["huggingface-cli", "download", "--repo-type", "dataset", dataset_id],
        check=True,
    )
    candidates = _dataset_cache_dirs(dataset_id)
    if not candidates:
        raise FileNotFoundError(
            f"Dataset {dataset_id} was downloaded but no snapshot was found under {_hf_cache_root()}"
        )
    return candidates[0]


def _resolve_dataset_root(request: pytest.FixtureRequest, option_name: str, dataset_id: str) -> Path:
    value = request.config.getoption(option_name)
    if value:
        path = Path(value)
        if not path.exists():
            pytest.skip(f"Dataset path does not exist: {path}")
        return path
    return _ensure_dataset_snapshot(dataset_id)


@dataclass
class AccuracyServerConfig:
    generate_params: OmniServerParams
    judge_params: OmniServerParams
    run_level: str
    model_prefix: str

    @contextmanager
    def generate_server(self):
        params = self.generate_params
        model = self.model_prefix + params.model
        server_args = params.server_args or []
        if params.use_omni and params.stage_init_timeout is not None:
            server_args = ["--stage-init-timeout", str(params.stage_init_timeout), *server_args]
        with OmniServer(
            model,
            server_args,
            port=params.port,
            env_dict=params.env_dict,
            use_omni=params.use_omni,
        ) as server:
            yield server

    @contextmanager
    def judge_server(self):
        params = self.judge_params
        model = self.model_prefix + params.model
        server_args = params.server_args or []
        with OmniServer(
            model,
            server_args,
            port=params.port,
            env_dict=params.env_dict,
            use_omni=params.use_omni,
        ) as server:
            yield server


@pytest.fixture(scope="session")
def gebench_dataset_root(request: pytest.FixtureRequest) -> Path:
    return _resolve_dataset_root(request, "gebench_root", "stepfun-ai/GEBench")


@pytest.fixture(scope="session")
def gedit_dataset_root(request: pytest.FixtureRequest) -> Path:
    return _resolve_dataset_root(request, "gedit_root", "stepfun-ai/GEdit-Bench")


@pytest.fixture(scope="session")
def accuracy_workers(request: pytest.FixtureRequest) -> int:
    return int(request.config.getoption("accuracy_workers"))


@pytest.fixture(scope="session")
def wan22_i2v_image_source(request: pytest.FixtureRequest) -> str | None:
    value = request.config.getoption("wan22_i2v_image_source")
    return str(value) if value else None


@pytest.fixture(scope="session")
def wan22_i2v_online_timeout_seconds(request: pytest.FixtureRequest) -> int:
    return int(request.config.getoption("wan22_i2v_online_timeout_seconds"))


@pytest.fixture(scope="session")
def gebench_samples_per_type(request: pytest.FixtureRequest) -> int:
    return int(request.config.getoption("gebench_samples_per_type"))


@pytest.fixture(scope="session")
def gebench_num_inference_steps(request: pytest.FixtureRequest) -> int:
    return int(request.config.getoption("gebench_num_inference_steps"))


@pytest.fixture(scope="session")
def gedit_samples_per_group(request: pytest.FixtureRequest) -> int:
    return int(request.config.getoption("gedit_samples_per_group"))


@pytest.fixture(scope="session")
def accuracy_artifact_root() -> Path:
    root = Path(__file__).resolve().parent / "artifacts"
    root.mkdir(parents=True, exist_ok=True)
    return root


@pytest.fixture(scope="session")
def qwen_bear_image(accuracy_artifact_root: Path) -> Image.Image:
    """Download the Qwen bear image from the URL and save it to the accuracy artifact root."""
    QWEN_BEAR_IMAGE_URL = "https://vllm-public-assets.s3.us-west-2.amazonaws.com/omni-assets/qwen-bear.png"
    response = requests.get(QWEN_BEAR_IMAGE_URL, timeout=60)
    response.raise_for_status()
    image = Image.open(BytesIO(response.content)).convert("RGB")
    image.save(accuracy_artifact_root / "qwen_bear.png")
    return image


@pytest.fixture(scope="session")
def rabbit_image(accuracy_artifact_root: Path) -> Image.Image:
    """Download the rabbit image from the URL and save it to the accuracy artifact root."""
    RABBIT_IMAGE_URL = "https://vllm-public-assets.s3.us-west-2.amazonaws.com/omni-assets/rabbit.png"
    response = requests.get(RABBIT_IMAGE_URL, timeout=60)
    response.raise_for_status()
    image = Image.open(BytesIO(response.content)).convert("RGB")
    image.save(accuracy_artifact_root / "rabbit.png")
    return image


def _build_accuracy_server_config(
    *,
    generate_model: str,
    judge_model: str,
    shared_gpu: str,
    port: int,
    run_level: str,
    model_prefix: str,
    generate_devices: str | None = None,
    extra_generate_args: list[str] | None = None,
    stage_init_timeout: int = 300,
    init_timeout: int | None = None,
) -> AccuracyServerConfig:
    if torch.cuda.device_count() < 1:
        pytest.skip("Need at least 1 CUDA GPU for accuracy benchmark smoke tests.")

    if not generate_model:
        pytest.skip("No generate model configured for accuracy benchmark test.")

    devices = generate_devices or shared_gpu
    num_devices = len([d for d in devices.split(",") if d.strip()])
    if torch.cuda.device_count() < num_devices:
        pytest.skip(f"Need at least {num_devices} CUDA GPUs for this accuracy benchmark.")

    generate_server_args = extra_generate_args if extra_generate_args is not None else ["--num-gpus", "1"]
    judge_server_args = [
        "--max-model-len",
        "32768",
        "--gpu-memory-utilization",
        "0.8",
    ]

    generate_params_kwargs: dict = dict(
        model=generate_model,
        port=port,
        server_args=generate_server_args,
        env_dict={"CUDA_VISIBLE_DEVICES": devices},
        use_omni=True,
        stage_init_timeout=stage_init_timeout,
    )
    if init_timeout is not None:
        generate_params_kwargs["init_timeout"] = init_timeout

    return AccuracyServerConfig(
        generate_params=OmniServerParams(**generate_params_kwargs),
        judge_params=OmniServerParams(
            model=judge_model,
            port=port,
            server_args=judge_server_args,
            env_dict={"CUDA_VISIBLE_DEVICES": shared_gpu},
            use_omni=False,
        ),
        run_level=run_level,
        model_prefix=model_prefix,
    )


@pytest.fixture
def gebench_accuracy_servers(
    request: pytest.FixtureRequest,
    run_level: str,
    model_prefix: str,
) -> AccuracyServerConfig:
    devices_opt: str | None = request.config.getoption("gebench_devices")
    stage_overrides: str | None = request.config.getoption("gebench_stage_overrides")
    extra_args_json: str | None = request.config.getoption("gebench_extra_server_args")

    extra_args: list[str] | None = None
    stage_init_timeout = 300
    init_timeout: int | None = None

    if devices_opt:
        num_devices = len([d for d in devices_opt.split(",") if d.strip()])
        extra_args = ["--tensor-parallel-size", str(num_devices)]
        if stage_overrides:
            extra_args += ["--stage-overrides", stage_overrides]
        if extra_args_json:
            extra_args += json.loads(extra_args_json)
        stage_init_timeout = 600
        init_timeout = 1800

    return _build_accuracy_server_config(
        generate_model=request.config.getoption("gebench_model"),
        judge_model=request.config.getoption("accuracy_judge_model"),
        shared_gpu=str(request.config.getoption("accuracy_gpu")),
        port=int(request.config.getoption("gebench_port")),
        run_level=run_level,
        model_prefix=model_prefix,
        generate_devices=devices_opt,
        extra_generate_args=extra_args,
        stage_init_timeout=stage_init_timeout,
        init_timeout=init_timeout,
    )


@pytest.fixture
def gedit_accuracy_servers(
    request: pytest.FixtureRequest,
    run_level: str,
    model_prefix: str,
) -> AccuracyServerConfig:
    return _build_accuracy_server_config(
        generate_model=request.config.getoption("gedit_model"),
        judge_model=request.config.getoption("accuracy_judge_model"),
        shared_gpu=str(request.config.getoption("accuracy_gpu")),
        port=int(request.config.getoption("gedit_port")),
        run_level=run_level,
        model_prefix=model_prefix,
    )
