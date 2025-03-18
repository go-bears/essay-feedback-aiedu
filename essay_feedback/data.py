import importlib.resources
import os

MODULE_FOLDER = importlib.resources.files("essay_feedback")

DATA_FOLDER = MODULE_FOLDER.joinpath("data")
ESSAYS_FOLDER = DATA_FOLDER.joinpath("essays")

def get_all_essay_ids() -> list[str]:
    return [ p.name for p in os.scandir(ESSAYS_FOLDER) ]

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
