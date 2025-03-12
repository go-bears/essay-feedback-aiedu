from enum import Enum
from typing import Literal
import importlib.resources
import json

# LLM backends
from openai import OpenAI
from openai.types.chat import ChatCompletion
from anthropic import Anthropic
from anthropic.types import Message, TextBlock
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request
from anthropic.types.messages.message_batch_succeeded_result import MessageBatchSucceededResult
type SupportedAPI = Literal["oai", "anthropic"] # TODO: make sure adding an entry here breaks type checking

# mustache templating library
import chevron
import dotenv
try:
    dotenv.load_dotenv()
except:
    print(f"Could not load environment variables. Continuing without them ...")

MODULE_FOLDER = importlib.resources.files("essay_feedback")

DATA_FOLDER = MODULE_FOLDER.joinpath("data")
ESSAYS_FOLDER = DATA_FOLDER.joinpath("essays")

TEMPLATE_FOLDER = MODULE_FOLDER.joinpath("templates")

def _get_templated(name: str, **kwargs) -> str:
    with open(f"{TEMPLATE_FOLDER}/{name}.mustache", "r") as f:
        return chevron.render(f, data=kwargs)

def get_essay(id: str) -> str:
    """
    Get the body of an essay
    """

    with open(f"{ESSAYS_FOLDER}/{f"{id.rstrip(".txt")}.txt"}", "r") as f:
        return (
            f
            .read()
            .replace("\n\n\n", "\n\n") # remove excess newlines
            .replace(u'\xc2\xa0', ' ') # turn non-breaking unicode spaces into normal spaces
        )

def _get_llm_call_args(essay_id: str, model: str, api: SupportedAPI) -> dict:
    essay = get_essay(essay_id)
    system_prompt = _get_templated("system_prompt")
    message_prompt = _get_templated("message_prompt", text=essay)
    match api:
        case "oai":
            return {
                'model' : model,
                'messages' : [
                    { "role" : "system",
                    "content" : system_prompt },
                    { "role" : "user",
                    "content" : message_prompt },
                ]
            }
        case "anthropic":
            return {
                'model' : model,
                'system' : system_prompt,
                'messages' : [
                    { "role" : "user",
                    "content" : message_prompt },
                ]
            }

def call_llm(essay_id: str, model: str, api: SupportedAPI, params: dict = {}) -> str:
    args = _get_llm_call_args(essay_id, model, api) | params
    match api:
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

def call_llm_batch(essay_ids: list[str], model: str, api: SupportedAPI) -> str:
    """
    Call the specified model/api pair on the provided essay IDs. Produce the ID of the spawned batch job.
    """
    arg_sets = [
        (eid, _get_llm_call_args(eid, model, api))
        for eid in essay_ids
    ]

    match api:
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

def get_batch(batch_id: str, api: SupportedAPI) -> Literal[BatchStatus.Incomplete, BatchStatus.Failed] | dict[str, str | Literal[BatchStatus.Failed]]:
    match api:
        case "oai":
            client = OpenAI()
            oai_batch_job = client.batches.retrieve(batch_id)

            if oai_batch_job.status in ['validating', 'in_progress', 'finalizing']:
                return BatchStatus.Incomplete
            if oai_batch_job.status in  ['failed', 'expired', 'cancelling', 'cancelled']:
                return BatchStatus.Failed

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
            # return BatchStatus.Incomplete
            client = Anthropic()
            anthropic_batch = client.messages.batches.retrieve(batch_id)
            if anthropic_batch.processing_status == 'in_progress':
                return BatchStatus.Incomplete
            if anthropic_batch.processing_status == 'cancelling':
                return BatchStatus.Failed

            assert anthropic_batch.processing_status == "ended"
            result = client.messages.batches.results(batch_id)

            outputs = { }
            for entry in result:
                essay_id = entry.custom_id.split('-')[-1]
                if entry.result.type in ['errored', 'canceled', 'expired']:
                    outputs[essay_id] = BatchStatus.Failed
                else:
                    assert isinstance(entry.result, MessageBatchSucceededResult)
                    content_block = entry.result.message.content
                    assert isinstance(content_block, TextBlock)
                    outputs[essay_id] = content_block.text
            return outputs

__all__ = [
    "call_llm",
    "call_llm_batch",
    "get_batch",
    "get_essay",
    "BatchStatus"
]
