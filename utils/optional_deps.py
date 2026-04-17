import importlib
from typing import Iterable


class MissingOptionalDependency(RuntimeError):
    def __init__(
        self,
        feature_name: str,
        module_name: str,
        missing_name: str,
        install_hint: str = "",
    ):
        self.feature_name = str(feature_name or "").strip() or "This feature"
        self.module_name = str(module_name or "").strip()
        self.missing_name = str(missing_name or "").strip() or self.module_name
        self.install_hint = str(install_hint or "").strip()
        super().__init__(self._build_message())

    def _build_message(self) -> str:
        lines = [
            f"{self.feature_name} is unavailable because '{self.missing_name}' is not installed."
        ]
        if self.install_hint:
            lines.append(self.install_hint)
        return "\n".join(lines)


def import_optional_module(
    module_name: str,
    *,
    feature_name: str,
    install_hint: str = "",
):
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        raise MissingOptionalDependency(
            feature_name=feature_name,
            module_name=module_name,
            missing_name=getattr(exc, "name", "") or module_name,
            install_hint=install_hint,
        ) from exc


def format_missing_dependency_message(exc: MissingOptionalDependency) -> str:
    return str(exc)


def ensure_optional_modules(
    modules: Iterable[str],
    *,
    feature_name: str,
    install_hint: str = "",
) -> None:
    for module_name in modules or ():
        name = str(module_name or "").strip()
        if not name:
            continue
        import_optional_module(
            name,
            feature_name=feature_name,
            install_hint=install_hint,
        )
