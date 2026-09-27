from pathlib import Path
from typing import Iterable


def print_tree(
    root: Path,
    prefix: str = "",
    *,
    skip_dir_contains: Iterable[str] | None = None,
    skip_file_contains: Iterable[str] | None = None,
) -> None:
    skip_dir_contains = skip_dir_contains or []
    skip_file_contains = skip_file_contains or []

    entries = sorted(root.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
    visible_entries = []

    for p in entries:
        # 目录名过滤
        if p.is_dir() and any(s in p.name for s in skip_dir_contains):
            continue
        # 文件名过滤
        if p.is_file() and any(s in p.name for s in skip_file_contains):
            continue
        visible_entries.append(p)

    for i, path in enumerate(visible_entries):
        connector = "└── " if i == len(visible_entries) - 1 else "├── "
        print(prefix + connector + path.name)

        if path.is_dir():
            extension = "    " if i == len(visible_entries) - 1 else "│   "
            print_tree(
                path,
                prefix + extension,
                skip_dir_contains=skip_dir_contains,
                skip_file_contains=skip_file_contains,
            )


if __name__ == "__main__":
    dir_path = "./"
    print_tree(
        Path(dir_path),
        skip_dir_contains=["output", "__pycache__", ".git", 'data', 'third'],
        skip_file_contains=[".log", ".pth", ".pt", ".onnx", '.jpg', '.png'],
    )