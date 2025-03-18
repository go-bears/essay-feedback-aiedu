from dataclasses import dataclass
import functools
import importlib.resources

# mustache templating library
import chevron

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

    with open(f"{ESSAYS_FOLDER}/{id.rstrip('.txt') + '.txt'}", "r") as f:
        return (
            f
            .read()
            .replace("\n\n\n", "\n\n") # remove excess newlines
            .replace(u'\xc2\xa0', ' ') # turn non-breaking unicode spaces into normal spaces
        )

@dataclass
class Task:
    essay_ids: list[str]
    system_prompt_file: str
    message_prompt_file: str

    @property
    def system_prompts(self) -> list[str]:
        return [
            _get_templated(self.system_prompt_file, essay=get_essay(eid))
            for eid in self.essay_ids
        ]

    @property
    def message_prompts(self) -> list[str]:
        return [
            _get_templated(self.message_prompt_file, essay=get_essay(eid))
            for eid in self.essay_ids
        ]

Annotate = functools.partial(Task, system_prompt_file="annotate.system.mustache", message_prompt_file="annotate.message.mustache")
