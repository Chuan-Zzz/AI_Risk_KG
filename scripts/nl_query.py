# -*- coding: utf-8 -*-
"""
自然语言查询 Neo4j 知识图谱

使用方法:
    python scripts/nl_query.py "哪些脆弱性导致深度伪造风险？"
    python scripts/nl_query.py "AI系统有哪些风险？"
    python scripts/nl_query.py "谁受到了隐私侵害？"
"""

import argparse
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.storage.neo4j_store import Neo4jStore
from src.alignment.embedder import Embedder
from src.rag.text2cypher import Text2Cypher
from src.rag.answer_generator import AnswerGenerator


def main():
    parser = argparse.ArgumentParser(description="自然语言查询 Neo4j 知识图谱")
    parser.add_argument("question", type=str, help="自然语言问题")
    parser.add_argument("--verbose", "-v", action="store_true", help="显示详细信息")

    args = parser.parse_args()

    # 连接数据库
    store = Neo4jStore()
    if not store.connect():
        print("连接 Neo4j 失败")
        return

    embedder = Embedder()
    t2c = Text2Cypher(store, embedder)
    answer_gen = AnswerGenerator()

    print(f"\n问题: {args.question}")
    print("=" * 60)

    try:
        # 执行自然语言查询
        result = t2c.execute(args.question)

        if args.verbose:
            print(f"\n生成的 Cypher:")
            print(result.get('cypher', 'N/A'))
            print()

        # 显示查询结果
        records = result.get('result', [])
        if records:
            print("\n查询结果:")
            for i, r in enumerate(records, 1):
                print(f"[{i}] {r}")

            # 生成自然语言答案
            context = "\n".join([str(r) for r in records])
            print("\n" + "=" * 60)
            print("答案:")
            answer = answer_gen.generate(args.question, context)
            print(answer)
        else:
            print("\n未找到相关结果")

    except Exception as e:
        print(f"\n查询失败: {e}")

    finally:
        store.close()


if __name__ == "__main__":
    main()
