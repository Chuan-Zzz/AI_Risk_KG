# -*- coding: utf-8 -*-
"""
Neo4j 查询工具

使用方法:
    python scripts/query_neo4j.py --stats          # 查看统计
    python scripts/query_neo4j.py --risks          # 查看所有风险
    python scripts/query_neo4j.py --vulnerabilities # 查看所有脆弱性
    python scripts/query_neo4j.py --cypher "MATCH (n) RETURN n LIMIT 5"  # 自定义Cypher
"""

import argparse
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.storage.neo4j_store import Neo4jStore


def show_stats(store: Neo4jStore):
    """显示数据库统计信息"""
    stats = store.get_stats()
    print("\n=== Neo4j 数据库统计 ===")
    print(f"总节点数: {stats.get('total_nodes', 0)}")
    print(f"总关系数: {stats.get('total_relations', 0)}")
    print(f"\n各类型节点数:")
    for label, count in stats.get('nodes_by_type', {}).items():
        print(f"  - {label}: {count}")


def show_risks(store: Neo4jStore):
    """显示所有风险"""
    print("\n=== 所有风险 ===")
    with store.driver.session() as session:
        result = session.run("""
            MATCH (r:Risk)
            RETURN r.name AS name, r.description AS desc
            ORDER BY r.name
        """)
        for i, r in enumerate(result, 1):
            print(f"\n[{i}] {r['name']}")
            print(f"    描述: {r['desc'] or 'N/A'}")


def show_vulnerabilities(store: Neo4jStore):
    """显示所有脆弱性"""
    print("\n=== 所有脆弱性 ===")
    with store.driver.session() as session:
        result = session.run("""
            MATCH (v:Vulnerability)
            RETURN v.name AS name, v.description AS desc
            ORDER BY v.name
        """)
        for i, r in enumerate(result, 1):
            print(f"\n[{i}] {r['name']}")
            print(f"    描述: {r['desc'] or 'N/A'}")


def show_all_nodes(store: Neo4jStore, limit: int = 20):
    """显示所有节点"""
    print(f"\n=== 所有节点 (前 {limit} 个) ===")
    with store.driver.session() as session:
        result = session.run(f"""
            MATCH (n)
            RETURN n.name AS name, labels(n) AS type, n.description AS desc
            LIMIT {limit}
        """)
        for r in result:
            label = r['type'][0] if r['type'] else 'Unknown'
            desc = (r['desc'][:50] + '...') if r['desc'] and len(r['desc']) > 50 else (r['desc'] or 'N/A')
            print(f"[{label}] {r['name']}")
            print(f"    {desc}")


def show_relations(store: Neo4jStore, limit: int = 30):
    """显示所有关系"""
    print(f"\n=== 所有关系 (前 {limit} 个) ===")
    with store.driver.session() as session:
        result = session.run(f"""
            MATCH (a)-[r]->(b)
            RETURN a.name AS from, type(r) AS rel, b.name AS to
            LIMIT {limit}
        """)
        for r in result:
            print(f"{r['from']} --[{r['rel']}]--> {r['to']}")


def execute_cypher(store: Neo4jStore, query: str):
    """执行自定义 Cypher 查询"""
    print(f"\n=== 执行 Cypher ===")
    print(f"查询: {query}")
    print("-" * 50)

    with store.driver.session() as session:
        result = session.run(query)
        for r in result:
            print(dict(r))


def main():
    parser = argparse.ArgumentParser(description="Neo4j 查询工具")
    parser.add_argument("--stats", action="store_true", help="显示统计信息")
    parser.add_argument("--risks", action="store_true", help="显示所有风险")
    parser.add_argument("--vulnerabilities", action="store_true", help="显示所有脆弱性")
    parser.add_argument("--nodes", action="store_true", help="显示所有节点")
    parser.add_argument("--relations", action="store_true", help="显示所有关系")
    parser.add_argument("--all", action="store_true", help="显示全部信息")
    parser.add_argument("--cypher", type=str, help="执行自定义 Cypher 查询")
    parser.add_argument("--limit", type=int, default=20, help="限制返回数量")

    args = parser.parse_args()

    # 连接数据库
    store = Neo4jStore()
    if not store.connect():
        print("连接 Neo4j 失败")
        return

    try:
        if args.stats or args.all:
            show_stats(store)

        if args.risks or args.all:
            show_risks(store)

        if args.vulnerabilities or args.all:
            show_vulnerabilities(store)

        if args.nodes or args.all:
            show_all_nodes(store, args.limit)

        if args.relations or args.all:
            show_relations(store, args.limit)

        if args.cypher:
            execute_cypher(store, args.cypher)

        # 默认显示统计
        if not any([args.stats, args.risks, args.vulnerabilities, args.nodes,
                    args.relations, args.all, args.cypher]):
            show_stats(store)

    finally:
        store.close()


if __name__ == "__main__":
    main()
