"""分析 Level 2 风险链的分歧程度分布"""
import json, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

fields = ['risk_source_score','risk_score','consequence_score','impact_score','affected_actor_score','risk_control_score']

for pair, fa, fb in [('A26_vs_A44','风险链评测1.json','风险链评测3.json'),('A41_vs_A45','风险链评测2.json','风险链评测4.json')]:
    with open(f'eval/标注结果/{fa}','r',encoding='utf-8') as f: da = json.load(f)
    with open(f'eval/标注结果/{fb}','r',encoding='utf-8') as f: db = json.load(f)

    ea = {t['data']['event_id']:t for t in da}
    eb = {t['data']['event_id']:t for t in db}
    common = set(ea) & set(eb)

    diff_counts = {}
    diff_details = []

    for eid in sorted(common):
        ra = {}
        rb = {}
        for r in ea[eid]['annotations'][0]['result']:
            if r['type']=='rating': ra[r['from_name']]=r['value']['rating']
        for r in eb[eid]['annotations'][0]['result']:
            if r['type']=='rating': rb[r['from_name']]=r['value']['rating']

        n_diff = sum(1 for f in fields if ra.get(f)!=rb.get(f))
        diff_counts[n_diff] = diff_counts.get(n_diff,0)+1

        if n_diff > 0:
            diffs = {f: (ra.get(f), rb.get(f)) for f in fields if ra.get(f)!=rb.get(f)}
            gap = sum(abs(ra.get(f,0)-rb.get(f,0)) for f in fields if ra.get(f)!=rb.get(f))
            diff_details.append((eid, n_diff, gap, diffs))

    print(f'=== {pair} ===')
    print(f'分歧槽位数分布:')
    for k in sorted(diff_counts):
        print(f'  {k}个槽位不同: {diff_counts[k]}个任务')

    diff_details.sort(key=lambda x: -x[1])
    print(f'分歧最严重的前5个:')
    for eid, n, gap, diffs in diff_details[:5]:
        print(f'  {eid}: {n}个槽位不同, 总差距={gap}')
        for f,(a,b) in diffs.items():
            print(f'    {f}: A={a} B={b} 差={abs((a or 0)-(b or 0))}')

    gap_dist = {}
    for _,_,gap,_ in diff_details:
        gap_dist[gap] = gap_dist.get(gap,0)+1
    print(f'总评分差距分布:')
    for g in sorted(gap_dist):
        print(f'  差距={g}: {gap_dist[g]}个任务')

    # 只差1分的槽位占比
    one_point_diffs = 0
    total_diff_slots = 0
    for _,_,_,diffs in diff_details:
        for f,(a,b) in diffs.items():
            total_diff_slots += 1
            if abs((a or 0)-(b or 0)) == 1:
                one_point_diffs += 1
    print(f'分歧槽位中差1分的占比: {one_point_diffs}/{total_diff_slots} = {one_point_diffs/total_diff_slots*100:.1f}%')
    print()
