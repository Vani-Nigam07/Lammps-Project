#!/usr/bin/env python3
import re
import sys
from pathlib import Path

try:
    import tomllib  # py3.11+
except ModuleNotFoundError:  # pragma: no cover
    try:
        import tomli as tomllib  # type: ignore
    except ModuleNotFoundError:
        tomllib = None

try:
    import yaml  # type: ignore
except ModuleNotFoundError:
    yaml = None

try:
    from packaging.requirements import Requirement  # type: ignore
except ModuleNotFoundError:
    Requirement = None  # type: ignore


def norm_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def parse_req(req: str):
    if Requirement is not None:
        r = Requirement(req)
        spec = str(r.specifier) if r.specifier else ""
        return norm_name(r.name), spec
    # Very small fallback: split on first specifier char
    m = re.match(r"^([A-Za-z0-9_.-]+)(.*)$", req.strip())
    if not m:
        return norm_name(req.strip()), ""
    name, rest = m.group(1), m.group(2).strip()
    return norm_name(name), rest


def _load_env_pip_minimal(text: str):
    pip_list = []
    in_pip = False
    pip_indent = None
    for line in text.splitlines():
        raw = line.rstrip()
        if not raw or raw.lstrip().startswith("#"):
            continue
        if re.match(r"^\s*-\s*pip\s*:\s*$", raw):
            in_pip = True
            pip_indent = len(line) - len(line.lstrip())
            continue
        if in_pip:
            indent = len(line) - len(line.lstrip())
            if pip_indent is not None and indent <= pip_indent:
                in_pip = False
                pip_indent = None
                continue
            m = re.match(r"^\s*-\s*(.+)$", raw)
            if m:
                pip_list.append(m.group(1).strip())
    return pip_list


def load_env(path: Path):
    text = path.read_text()
    if yaml is None:
        pip_list = _load_env_pip_minimal(text)
    else:
        data = yaml.safe_load(text) or {}
        deps = data.get("dependencies", [])
        pip_list = []
        for item in deps:
            if isinstance(item, dict) and "pip" in item:
                pip_list.extend(item.get("pip", []) or [])
    env_pip = {}
    for req in pip_list:
        name, spec = parse_req(req)
        env_pip[name] = spec
    return env_pip


def load_pyproject(path: Path):
    if tomllib is None:
        raise RuntimeError("tomllib/tomli required to parse pyproject.toml")
    data = tomllib.loads(path.read_text())
    deps = data.get("project", {}).get("dependencies", []) or []
    proj = {}
    for req in deps:
        name, spec = parse_req(req)
        proj[name] = spec
    return proj


def main():
    root = Path(__file__).resolve().parents[1]
    env_path = root / "environment.yml"
    py_path = root / "pyproject.toml"

    if not env_path.exists() or not py_path.exists():
        print("Missing environment.yml or pyproject.toml", file=sys.stderr)
        return 2

    env = load_env(env_path)
    proj = load_pyproject(py_path)

    common = sorted(set(env) & set(proj))
    only_env = sorted(set(env) - set(proj))
    only_proj = sorted(set(proj) - set(env))

    mismatched = []
    for name in common:
        if (env.get(name) or "") != (proj.get(name) or ""):
            mismatched.append(name)

    def fmt(name, mapping):
        spec = mapping.get(name)
        return f"{name}{spec}" if spec else name

    print("Common deps with version/spec mismatches:")
    if mismatched:
        for name in mismatched:
            print(f"- {name}: env '{env.get(name) or ''}' vs pyproject '{proj.get(name) or ''}'")
    else:
        print("- none")

    print("\nIn environment.yml (pip) but missing from pyproject.toml:")
    if only_env:
        for name in only_env:
            print(f"- {fmt(name, env)}")
    else:
        print("- none")

    print("\nIn pyproject.toml but missing from environment.yml (pip):")
    if only_proj:
        for name in only_proj:
            print(f"- {fmt(name, proj)}")
    else:
        print("- none")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
