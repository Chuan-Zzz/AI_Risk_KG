"""合并显式实体评测1 + annotation_tasks_b"""
import json, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

with open('eval/标注结果/显式实体评测1.json', 'r', encoding='utf-8') as f:
    d1 = json.load(f)
with open('eval/label_studio/level1_entity/annotation_tasks_b.json', 'r', encoding='utf-8') as f:
    db = json.load(f)

eids1 = {t['data']['event_id'] for t in d1}
eidsb = {t['data']['event_id'] for t in db}
common = eids1 & eidsb
only1 = eids1 - eidsb
onlyb = eidsb - eids1

print(f'评测1: {len(d1)} 个任务')
print(f'annotation_tasks_b: {len(db)} 个任务')
print(f'共同: {len(common)}')
print(f'评测1独有: {len(only1)}')
print(f'tasks_b独有: {len(onlyb)}')

t = db[0]
has_ann = 'annotations' in t
has_pred = 'predictions' in t
print(f'tasks_b 有 annotations: {has_ann}')
print(f'tasks_b 有 predictions: {has_pred}')

# 合并: 评测1全部保留 + tasks_b中不重复的任务
merged = list(d1)
for t in db:
    if t['data']['event_id'] not in eids1:
        merged.append(t)

with open('eval/标注结果/显式实体评测_合并.json', 'w', encoding='utf-8') as f:
    json.dump(merged, f, ensure_ascii=False, indent=2)

print(f'\n合并后: {len(merged)} 个任务')
print(f'已保存到: eval/标注结果/显式实体评测_合并.json')
