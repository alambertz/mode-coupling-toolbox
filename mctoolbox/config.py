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


def _toml_value(v):
    if isinstance(v, bool):
        return 'true' if v else 'false'
    if isinstance(v, (int, float)):
        return repr(v)
    if isinstance(v, (list, tuple)):
        return '[' + ', '.join(_toml_value(x) for x in v) + ']'
    return '"' + str(v).replace('\\', '\\\\').replace('"', '\\"') + '"'


def dump_config(cfg, file):
    """Write a configuration (scalars, lists, tables and arrays of tables) as TOML.

    The inverse of load_config for the figure configs; the 'dir' entry is not written.
    """
    scalars = {k: v for k, v in cfg.items() if k != 'dir' and not isinstance(v, dict)
               and not (isinstance(v, list) and v and isinstance(v[0], dict))}
    lines = [f'{k} = {_toml_value(v)}' for k, v in scalars.items()]
    for k, v in cfg.items():
        if isinstance(v, dict):
            lines += ['', f'[{k}]'] + [f'{kk} = {_toml_value(vv)}' for kk, vv in v.items()]
        elif isinstance(v, list) and v and isinstance(v[0], dict):
            for item in v:
                lines += ['', f'[[{k}]]'] + [f'{kk} = {_toml_value(vv)}' for kk, vv in item.items()]
    Path(file).write_text('\n'.join(lines) + '\n')
