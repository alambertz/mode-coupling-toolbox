"""Regenerate the paper figures.

    python -m mctoolbox.figures            # all figures
    python -m mctoolbox.figures 4 7        # selected figures

Must be run from the repository root (the figure folders live in figures/).
"""
import runpy
import sys
import time
from pathlib import Path

FIGURES = Path.cwd()/'figures'
AVAILABLE = [4, 5, 6, 7]


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    wanted = [int(a) for a in argv] or AVAILABLE
    if not FIGURES.is_dir():
        sys.exit('run from the repository root (no figures/ folder here)')
    for n in wanted:
        script = FIGURES/f'fig{n}'/f'make_fig{n}.py'
        t = time.time()
        print(f'Fig. {n} ...', flush=True)
        runpy.run_path(str(script), run_name='__main__')
        print(f'Fig. {n} done in {time.time() - t:.0f} s -> {script.parent}')


if __name__ == '__main__':
    main()
