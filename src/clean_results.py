"""Remove generated results while preserving data and unrecognized files."""
import shutil
from pathlib import Path

from src.configuration import CONFIG, results_root


def main():
    root = results_root().resolve()
    protected = [Path.cwd().resolve(), Path('/'),
                 Path(CONFIG['paths']['raw']).resolve(),
                 Path(CONFIG['paths']['data']).resolve(),
                 Path(CONFIG['paths']['reports']).resolve()]
    if any(root == p or root in p.parents or p in root.parents for p in protected[2:]) or root in protected[:2]:
        raise ValueError('Refusing to clean a project or data directory')
    for name in ('models', 'predictions', 'metrics', 'figures', 'tables', 'logs', 'README.md'):
        path = root / name
        if path.is_symlink() or path.is_file():
            path.unlink()
        elif path.is_dir():
            shutil.rmtree(path)


if __name__ == '__main__':
    main()
