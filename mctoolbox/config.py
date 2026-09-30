"""Per-figure configuration (figures/figN/config.toml)."""
import tomllib
from pathlib import Path


def load_config(fig_dir):
    """Read config.toml of a figure folder; relative file names resolve inside that folder."""
    fig_dir = Path(fig_dir)
    with open(fig_dir/'config.toml', 'rb') as fh:
        cfg = tomllib.load(fh)
    cfg['dir'] = fig_dir
    return cfg


def path(cfg, name):
    return Path(cfg['dir'])/name
