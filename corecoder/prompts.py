from __future__ import annotations

from dataclasses import dataclass
from string import Template


@dataclass(frozen=True)
class PromptTemplate:
    name: str
    version: str
    template: str

    def render(self, **values: str) -> str:
        return Template(self.template).substitute(**values)


class PromptRegistry:
    """集中管理可版本化 Prompt，便于评测和灰度对比。"""

    def __init__(self):
        self._templates: dict[tuple[str, str], PromptTemplate] = {}

    def register(self, prompt: PromptTemplate) -> None:
        self._templates[(prompt.name, prompt.version)] = prompt

    def get(self, name: str, version: str) -> PromptTemplate:
        try:
            return self._templates[(name, version)]
        except KeyError as error:
            raise KeyError(f"Prompt 不存在: {name}@{version}") from error

    def list_versions(self, name: str) -> list[str]:
        return sorted(
            version
            for prompt_name, version in self._templates
            if prompt_name == name
        )