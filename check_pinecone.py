from __future__ import annotations

import argparse
import json
from pathlib import Path

from rlm.env_loader import load_project_env
from rlm.rag_retriever import init_knowledge_base


def main() -> int:
    parser = argparse.ArgumentParser(description="Check Pinecone RAG connectivity.")
    parser.add_argument(
        "--query",
        type=str,
        default="Blackwell CUDA optimization",
        help="Semantic query to run if the retriever is configured.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of Pinecone matches to request.",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    env_path = load_project_env(project_root)

    retriever = init_knowledge_base()
    status = retriever.status()
    payload = {
        "env_loaded": str(env_path) if env_path else None,
        "status": status,
        "query": args.query,
        "top_k": args.top_k,
        "indexes": retriever.list_indexes(),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))

    if status != "configured":
        return 1

    matches = retriever.search_many([args.query], top_k=args.top_k)
    print()
    print(retriever.format_matches(matches))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
