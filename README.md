# MAGIC AES

To run inference on a single model (baseline)
`modal run --detach -m src.inference --backend=vllm --baseline`

To run inference with 5 agents and orchestrator
`modal run --detach -m src.inference --backend=vllm --orchestration`

To run both at the same time
`modal run --detach -m src.inference --backend=vllm --orchestration --baseline`

All the prompts used are found in `common.py` and are separated in classes
- `GREGeneralGraderPrompts` has the prompts for the baseline model (single agent)
- `GREAgentPrompts` has the prompts for all 5 agents including their individual rubrics
- `GREOrchestratorPrompts` has the prompts for the orchestrator agent

TODO: Refactor
For now, to run the project, `cd essay_scoring/llm-finetuning` and `pip install modal`.



