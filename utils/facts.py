# utils/facts.py
from pathlib import Path
import yaml
from typing import Optional, Dict

_CACHE: Dict[str, dict] = {}

def load_facts(tenant_id: str = "ashar") -> dict:
    if tenant_id in _CACHE:
        return _CACHE[tenant_id]
    p = Path("clients") / tenant_id / "project_facts.yaml"
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    _CACHE[tenant_id] = data
    return data

def get_project_field(project_key: str, field: str, tenant_id: str = "ashar") -> Optional[str]:
    data = load_facts(tenant_id)
    proj = data["projects"].get(project_key or "", {})
    return proj.get(field)

def list_projects_by_category(category: str, tenant_id: str = "ashar") -> list:
    data = load_facts(tenant_id)
    return data["categories"].get(category, [])
