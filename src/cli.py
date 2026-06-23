"""Command-line entry point for the PageIndex RAG evaluation pipeline.

Usage:
  python -m src.cli pages                 # show PDF page counts vs free-tier cap
  python -m src.cli index [--force]       # index the 5 PDFs with PageIndex
  python -m src.cli draft-qa [--n 9]      # draft candidate QA pairs to review
  python -m src.cli validate-qa           # validate the verified qa_dataset.csv
  python -m src.cli ask --doc <slug> --q "<question>"
  python -m src.cli eval [--limit N]      # run the evaluation
  python -m src.cli report                # regenerate charts from results
"""
from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="src.cli", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("pages", help="show PDF page counts vs the free-tier cap")

    p_index = sub.add_parser("index", help="index the PDFs with PageIndex")
    p_index.add_argument("--force", action="store_true", help="re-index even if cached")
    p_index.add_argument("--doc", default=None, help="index only this slug (default: all)")

    p_draft = sub.add_parser("draft-qa", help="draft candidate QA pairs")
    p_draft.add_argument("--n", type=int, default=9, help="questions per document")

    sub.add_parser("validate-qa", help="validate qa_dataset.csv")

    p_ask = sub.add_parser("ask", help="answer a question against one document")
    p_ask.add_argument("--doc", required=True, help="document slug")
    p_ask.add_argument("--q", required=True, help="question text")
    p_ask.add_argument("--type", default=None, dest="qtype",
                       help="question_type hint (drives text vs vision grounding)")

    p_eval = sub.add_parser("eval", help="run the evaluation")
    p_eval.add_argument("--limit", type=int, default=None, help="max questions this run")

    sub.add_parser("report", help="regenerate result charts")

    args = parser.parse_args(argv)

    if args.command == "pages":
        from src.indexing.build_index import page_report
        import config
        pages = page_report()
        total = sum(pages.values())
        for slug, p in pages.items():
            print(f"  {slug:38s} {p:>3d}")
        print(f"  {'TOTAL':38s} {total:>3d}  (cap {config.PAGEINDEX_FREE_PAGE_CAP})")
        return 0

    if args.command == "index":
        if args.doc:
            from src.indexing.build_index import index_one
            index_one(args.doc, force=args.force)
        else:
            from src.indexing.build_index import index_all
            index_all(force=args.force)
        return 0

    if args.command == "draft-qa":
        from src.data.draft_qa import draft_all
        draft_all(n_per_doc=args.n)
        return 0

    if args.command == "validate-qa":
        from src.data.validate_qa import validate
        errs = validate()
        if errs:
            print(f"VALIDATION FAILED ({len(errs)} issues):")
            for e in errs:
                print("  -", e)
            return 1
        print("QA dataset is valid.")
        return 0

    if args.command == "ask":
        from src.generation.answer import answer_question
        res = answer_question(args.doc, args.q, question_type=args.qtype)
        print("\nANSWER:\n", res.answer)
        print(f"\nconfidence: {res.confidence:.2f}  |  grounding: {res.grounding_mode}")
        print("retrieved nodes:")
        for r in res.retrieved:
            print(f"  [{r.node_id}] {r.title} (page {r.page_index}) "
                  f"rel={r.relevance:.2f} — {r.reason}")
        return 0

    if args.command == "eval":
        from src.evaluation.run_eval import run
        run(limit=args.limit)
        return 0

    if args.command == "report":
        from src.evaluation.make_report import make_report
        make_report()
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
