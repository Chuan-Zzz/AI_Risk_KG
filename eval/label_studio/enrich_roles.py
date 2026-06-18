"""Enrich pre_extracted_entities in annotation_tasks.json with RoleAssignment data.

For each Stakeholder entity in the HTML, if a matching RoleAssignment node exists
in the event's event_subgraph.json, add a role badge (e.g., (AI开发者)) after the name.
"""

import json
import os
import re
import sys

ROLE_CN = {
    "AIDeveloper": "AI开发者",
    "AIProvider": "AI提供商",
    "AIDeployer": "AI部署方",
    "AIUser": "AI用户",
    "Regulator": "监管机构",
    "AffectedActor": "受影响方",
    "Stakeholder": "",
}

ROLE_COLOR = {
    "AIDeveloper": "#20c997",
    "AIProvider": "#339af0",
    "AIDeployer": "#51cf66",
    "AIUser": "#38d9a9",
    "Regulator": "#e64980",
    "AffectedActor": "#f06595",
}


def load_role_mapping(event_id: str) -> dict[str, str]:
    """Load RoleAssignment nodes from event_subgraph.json.

    Returns dict mapping entity_name -> role_subtype (e.g., "Apple Inc" -> "AIDeveloper").
    """
    json_path = os.path.join("output", event_id, "event_subgraph.json")
    if not os.path.isfile(json_path):
        return {}

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    mapping = {}
    for node in data.get("nodes", []):
        if node.get("entity_type") != "RoleAssignment":
            continue
        name = node.get("name", "")
        # Format: "EntityName_RoleType" — split from the last underscore
        last_underscore = name.rfind("_")
        if last_underscore <= 0:
            continue
        entity_name = name[:last_underscore]
        role_type = name[last_underscore + 1:]
        if role_type in ROLE_CN:
            mapping[entity_name] = role_type

    return mapping


def make_role_badge(role_type: str) -> str:
    """Generate HTML badge for a role type."""
    cn = ROLE_CN.get(role_type, role_type)
    color = ROLE_COLOR.get(role_type, "#9775fa")
    return (
        f'<span style="background:{color};color:#fff;padding:2px 8px;'
        f'border-radius:10px;font-size:11px;margin-left:8px;'
        f'font-weight:normal">{cn}</span>'
    )


def inject_roles(html: str, role_map: dict[str, str]) -> str:
    """Inject role badges into Stakeholder entity name divs in the HTML.

    Matches entity cards in the Stakeholder section and adds a role badge
    after the entity name if a matching role is found.
    """
    if not role_map:
        return html

    # Find the Stakeholder section
    stakeholder_match = re.search(r'\(Stakeholder\)', html)
    if not stakeholder_match:
        return html

    # We only modify entity name divs that appear AFTER the Stakeholder section header
    stakeholder_pos = stakeholder_match.start()

    # Pattern: entity name div inside an entity card (bold text in Stakeholder section)
    # <div style="font-weight:bold;color:#333">ENTITY_NAME</div>
    def replace_name(match):
        pos = match.start()
        # Only modify divs after the Stakeholder section header
        if pos < stakeholder_pos:
            return match.group(0)

        name = match.group(1)
        # Skip if it's a section header (contains parentheses with entity type)
        if "(" in name and ("个" in name or "-" in name):
            return match.group(0)

        role_type = role_map.get(name, "")
        if not role_type:
            return match.group(0)

        badge = make_role_badge(role_type)
        return match.group(0).replace(
            f">{name}</div>",
            f">{name}{badge}</div>"
        )

    result = re.sub(
        r'<div style="font-weight:bold;color:#333">([^<]+)</div>',
        replace_name,
        html
    )

    return result


def main():
    tasks_path = os.path.join("eval", "label_studio", "level1_entity", "annotation_tasks.json")

    with open(tasks_path, "r", encoding="utf-8") as f:
        tasks = json.load(f)

    enriched_count = 0
    badge_count = 0

    for task in tasks:
        event_id = task["data"].get("event_id", "")
        if not event_id:
            continue

        role_map = load_role_mapping(event_id)
        if not role_map:
            continue

        html = task["data"].get("pre_extracted_entities", "")
        original_html = html

        html = inject_roles(html, role_map)

        if html != original_html:
            task["data"]["pre_extracted_entities"] = html
            enriched_count += 1
            badge_count += sum(1 for name in role_map if name in html)

    with open(tasks_path, "w", encoding="utf-8") as f:
        json.dump(tasks, f, ensure_ascii=False, indent=2)

    print(f"Done: enriched {enriched_count} tasks, added {badge_count} role badges")


if __name__ == "__main__":
    main()
