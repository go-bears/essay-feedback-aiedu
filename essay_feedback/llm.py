from enum import Enum
from dataclasses import dataclass
from typing import Literal
import json

# LLM backends
from openai import OpenAI
from openai.types.chat import ChatCompletion
from anthropic import Anthropic
from anthropic.types import Message, TextBlock
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request
from anthropic.types.messages.message_batch_succeeded_result import MessageBatchSucceededResult

import dotenv
try:
    dotenv.load_dotenv()
except:
    print(f"Could not load environment variables. Continuing without them ...")

from .task import Task

@dataclass
class LLM:
    model: str
    api: Literal["oai", "anthropic"]

    def _get_call_args(self, system_prompt: str, message_prompt: str) -> dict:
        match self.api:
            case "oai":
                return {
                    'model' : self.model,
                    'messages' : [
                        { "role" : "system",
                        "content" : system_prompt },
                        { "role" : "user",
                        "content" : message_prompt },
                    ]
                }
            case "anthropic":
                return {
                    'model' : self.model,
                    'system' : system_prompt,
                    'messages' : [
                        { "role" : "user",
                        "content" : message_prompt },
                    ]
                }

    def call(self, task: Task, index: int | None = None, params: dict = {}) -> str | list[str]:
        if index is None:
            return [
                self.call(task, i, params)
                for i in range(len(task.essay_ids))
            ]

        args = self._get_call_args(task.system_prompts[index], task.message_prompts[index]) | params
        match self.api:
            case "oai":
                client = OpenAI()
                oai_resp: ChatCompletion = client.chat.completions.create(**args)
                assert isinstance(oai_resp, ChatCompletion), f"Got a non-chat response from openai!: {oai_resp}"
                oai_content = oai_resp.choices[0].message.content
                assert oai_content is not None
                return oai_content
            case "anthropic":
                client = Anthropic()
                anthropic_resp: Message = client.messages.create(**args)
                assert isinstance(anthropic_resp, Message), f"Got a non-message response from anthropic!: {anthropic_resp}"
                msg = anthropic_resp.content[0]
                assert isinstance(msg, TextBlock), f"Got a non-text block from anthropic!: {msg}"
                return msg.text
    
    class BatchStatus(Enum):
        Incomplete = 1
        Failed = -1

    def call_batch(self, task: Task) -> str:
        """
        Call the specified model/api pair on the provided essay IDs. Produce the ID of the spawned batch job.
        """
        arg_sets = [
            (eid, self._get_call_args(sys, msg))
            for eid, sys, msg in zip(task.essay_ids, task.system_prompts, task.message_prompts)
        ]

        match self.api:
            case "oai":
                client = OpenAI()
                oai_tasks = [
                    {
                        "custom_id": f"essay-{eid}",
                        "method": "POST",
                        "url": "/v1/chat/completions",
                        "body": args
                    }
                    for eid, args in arg_sets
                ]

                oai_batch_file = client.files.create(
                    purpose="batch",
                    file=bytes(
                        '\n'.join([json.dumps(obj) for obj in oai_tasks]),
                        encoding="utf-8"
                    )
                )
                oai_batch = client.batches.create(input_file_id=oai_batch_file.id, endpoint="/v1/chat/completions", completion_window="24h")
                return oai_batch.id
            case "anthropic":
                client = Anthropic()
                anthropic_batch = client.messages.batches.create(
                    requests=[
                        Request(
                            custom_id = f"essay-{eid}",
                            params = MessageCreateParamsNonStreaming(**args)
                        )
                        for eid, args in arg_sets
                    ]
                )
                return anthropic_batch.id
    
    def get_batch(self, batch_id: str) -> Literal["LLM.BatchStatus.Incomplete", "LLM.BatchStatus.Failed"] | dict[str, str | Literal["LLM.BatchStatus.Failed"]]:
        match self.api:
            case "oai":
                client = OpenAI()
                oai_batch_job = client.batches.retrieve(batch_id)

                if oai_batch_job.status in ['validating', 'in_progress', 'finalizing']:
                    return LLM.BatchStatus.Incomplete
                if oai_batch_job.status in  ['failed', 'expired', 'cancelling', 'cancelled']:
                    return LLM.BatchStatus.Failed

                assert oai_batch_job.status == 'completed'
                assert oai_batch_job.output_file_id is not None
                resp_bytes = client.files.content(oai_batch_job.output_file_id).content
                
                responses = str(resp_bytes, encoding="utf-8").split('\n')
                oai_results = { }
                for response_str in responses:
                    resp = json.loads(response_str)
                    essay_id = resp['custom_id'].split('-')[-1]
                    result = resp['response']['body']['choices'][0]['message']['content']
                    oai_results[essay_id] = result

                return oai_results

            case "anthropic":
                client = Anthropic()
                anthropic_batch = client.messages.batches.retrieve(batch_id)
                if anthropic_batch.processing_status == 'in_progress':
                    return LLM.BatchStatus.Incomplete
                if anthropic_batch.processing_status == 'cancelling':
                    return LLM.BatchStatus.Failed

                assert anthropic_batch.processing_status == "ended"
                result = client.messages.batches.results(batch_id)

                outputs = { }
                for entry in result:
                    essay_id = entry.custom_id.split('-')[-1]
                    if entry.result.type in ['errored', 'canceled', 'expired']:
                        outputs[essay_id] = LLM.BatchStatus.Failed
                    else:
                        assert isinstance(entry.result, MessageBatchSucceededResult)
                        content_block = entry.result.message.content
                        assert isinstance(content_block, TextBlock)
                        outputs[essay_id] = content_block.text
                return outputs
