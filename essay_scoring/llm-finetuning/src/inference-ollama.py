import time
from pathlib import Path
import os
import modal

from fastapi.responses import StreamingResponse
import subprocess

from .common import Colors, VOLUME_CONFIG, MINUTES, ALLOW_WANDB

INFERENCE_GPU_CONFIG = "A10G:2"


# def get_model_path_from_run(path: Path) -> Path:
#     with (path / "config.yml").open() as f:
#         return path / yaml.safe_load(f.read())["output_dir"] / "merged"

def init_ollama():
    import httpx
    subprocess.run(["systemctl", "daemon-reload"])
    subprocess.run(["systemctl", "enable", "ollama"])
    subprocess.run(["systemctl", "start", "ollama"])

    start_time = time.time()
    timeout = 30
    interval = 2

    while True:
        try:
            response = httpx.get("http://localhost:11434/api/version")
            if response.status_code == 200:
                print("Ollama service is ready")
                return
        except httpx.ConnectError:
            if time.time() - start_time > timeout:
                raise TimeoutError("Ollama service failed to start")
            print(
                f"Waiting for Ollama service... ({int(time.time() - start_time)}s)"
            )
            time.sleep(interval)


ollama_image = (
    modal.Image.debian_slim()
    .apt_install("curl", "systemctl")
    .run_commands(  # from https://github.com/ollama/ollama/blob/main/docs/linux.md
        "curl -L https://ollama.com/download/ollama-linux-amd64.tgz -o ollama-linux-amd64.tgz",
        "tar -C /usr -xzf ollama-linux-amd64.tgz",
        "useradd -r -s /bin/false -U -m -d /usr/share/ollama ollama",
        "usermod -a -G ollama $(whoami)",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})  # faster model transfers
    .copy_local_file("ollama.service", "/etc/systemd/system/ollama.service")
    .pip_install("ollama",
                 "httpx",
                 "loguru",
                 "huggingface_hub[hf_transfer]==0.30.1"
                 )
    .run_function(init_ollama)
    .entrypoint([])
)

inference_app = modal.App(
    "inference",
    secrets=[
        modal.Secret.from_name("huggingface-rw-joaquin"),
        modal.Secret.from_dict({"ALLOW_WANDB": os.environ.get("ALLOW_WANDB", "false")}),
        *([modal.Secret.from_name("wandb")] if ALLOW_WANDB else []),
    ],
)


@app.cls(
    gpu=INFERENCE_GPU_CONFIG,
    image=ollama_image,
    volumes=VOLUME_CONFIG,
    allow_concurrent_inputs=30,
    container_idle_timeout=15 * MINUTES,
)
class Inference:
    def __init__(self, run_name: str = "", run_dir: str = "/runs") -> None:
        self.run_name = run_name
        self.run_dir = run_dir

    @modal.enter()
    def init(self):
        if self.run_name:
            path = Path(self.run_dir) / self.run_name
            VOLUME_CONFIG[self.run_dir].reload()
            model_path = get_model_path_from_run(path)
        else:
            # Pick the last run automatically
            run_paths = list(Path(self.run_dir).iterdir())
            for path in sorted(run_paths, reverse=True):
                model_path = get_model_path_from_run(path)
                if model_path.exists():
                    break

        print(
            Colors.GREEN,
            Colors.BOLD,
            f"🧠: Initializing vLLM engine for model at {model_path}",
            Colors.END,
            sep="",
        )

        engine_args = AsyncEngineArgs(
            model=model_path,
            gpu_memory_utilization=0.95,
            tensor_parallel_size=N_INFERENCE_GPUS,
            disable_custom_all_reduce=True,  # brittle as of v0.5.0
        )
        self.engine = AsyncLLMEngine.from_engine_args(engine_args)

    async def _stream(self, input: str):
        if not input:
            return

        sampling_params = SamplingParams(
            repetition_penalty=1.1,
            temperature=0.2,
            top_p=0.95,
            top_k=50,
            max_tokens=1024,
        )
        request_id = random_uuid()
        results_generator = self.engine.generate(input, sampling_params, request_id)

        t0 = time.time()
        index, tokens = 0, 0
        async for request_output in results_generator:
            if (
                    request_output.outputs[0].text
                    and "\ufffd" == request_output.outputs[0].text[-1]
            ):
                continue
            yield request_output.outputs[0].text[index:]
            index = len(request_output.outputs[0].text)

            # Token accounting
            new_tokens = len(request_output.outputs[0].token_ids)
            tokens = new_tokens

        throughput = tokens / (time.time() - t0)
        print(
            Colors.GREEN,
            Colors.BOLD,
            f"🧠: Effective throughput of {throughput:.2f} tok/s",
            Colors.END,
            sep="",
        )

    @modal.method()
    async def completion(self, input: str):
        async for text in self._stream(input):
            yield text

    @modal.method()
    async def non_streaming(self, input: str):
        output = [text async for text in self._stream(input)]
        return "".join(output)

    @modal.fastapi_endpoint()
    async def web(self, input: str):
        return StreamingResponse(self._stream(input), media_type="text/event-stream")

    @modal.exit()
    def stop_engine(self):
        if N_INFERENCE_GPUS > 1:
            import ray

            ray.shutdown()

        # access private attribute to ensure graceful termination
        self.engine._background_loop_unshielded.cancel()


@app.local_entrypoint()
def inference_main(run: str = "", prompt: str = ""):
    while True:
        if not prompt:
            prompt = input(
                "Enter a prompt (including the prompt template, e.g. [INST] ... [/INST]):\n"
            )
        print(
            Colors.GREEN, Colors.BOLD, f"🧠: Querying model {run}", Colors.END, sep=""
        )
        response = ""
        for chunk in Inference(run).completion.remote_gen(prompt):
            response += chunk  # not streaming to avoid mixing with server logs
        print(Colors.BLUE, f"👤: {prompt}", Colors.END, sep="")
        print(Colors.GRAY, f"🤖: {response}", Colors.END, sep="")
