from importlib.resources import files


def load(name: str) -> str:
    return (files(__package__) / f"{name}.md").read_text(encoding="utf-8")
