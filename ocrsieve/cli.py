import argparse
import json
import sys
from collections import Counter

from . import __version__
from .classifier import MODEL_ENV_VAR, THREADS_ENV_VAR
from .pipeline import PageClassifier
from .sources import collect_paths
from .stage1 import DEFAULT_MAX_IMAGE_COVER


def build_parser():
    parser = argparse.ArgumentParser(
        prog="ocrsieve",
        description=(
            "Sieve pages by the extraction path they need: free text, "
            "OCR, or a strong model. Takes PDFs, images and directories."
        ),
    )
    parser.add_argument(
        "paths", nargs="+", help="PDF files, image files or directories"
    )
    parser.add_argument(
        "-o", "--output", help="write the JSON report to a file"
    )
    parser.add_argument(
        "--json", action="store_true", help="print the JSON report to stdout"
    )
    parser.add_argument(
        "-m",
        "--model",
        help=f"path to an .onnx model (also via {MODEL_ENV_VAR})",
    )
    parser.add_argument(
        "--no-model",
        action="store_true",
        help="run the deterministic stage only, without the visual model",
    )
    parser.add_argument(
        "--inspect",
        action="store_true",
        help="report per-page PDF signals instead of classifying",
    )
    parser.add_argument(
        "--max-image-cover",
        type=float,
        default=DEFAULT_MAX_IMAGE_COVER,
        help=(
            "maximum image coverage for a page with native text to stay "
            "digital_text; above it the page goes to the visual model. "
            f"0 sends any page containing an image (default: "
            f"{DEFAULT_MAX_IMAGE_COVER})"
        ),
    )
    parser.add_argument(
        "--threads",
        type=int,
        help=(
            "cores for ONNX Runtime; without it the whole machine is used "
            f"(also via {THREADS_ENV_VAR})"
        ),
    )
    parser.add_argument(
        "-q", "--quiet", action="store_true", help="do not print the summary"
    )
    parser.add_argument("-V", "--version", action="version", version=__version__)
    return parser


def print_report(documents, stream):
    for document in documents:
        print(document["file"], file=stream)
        for page in document["pages"]:
            confidence = page.get("confidence")
            detail = f" {confidence:.2f}" if confidence is not None else "     "
            print(
                f"  {page['page']:4d}  {page['label']:16s}{detail}  "
                f"{page['decided_by']:8s}  {page['reason']}",
                file=stream,
            )


def print_inspection(documents, stream):
    for document in documents:
        print(document["file"], file=stream)
        for page in document["pages"]:
            print(
                f"  {page['page']:4d}  {page['stage1']:16s}  "
                f"chars={page['chars']:<7d} cover={page['max_image_cover']:<6} "
                f"{page['reason']}",
                file=stream,
            )


def summarize(documents, stream):
    counts = Counter(p["label"] for d in documents for p in d["pages"])
    by_stage = Counter(p["decided_by"] for d in documents for p in d["pages"])
    total = sum(counts.values())
    if not total:
        print("no pages found", file=stream)
        return
    mixed = sum(1 for d in documents if d["mixed"])
    print(
        f"\n{total} pages in {len(documents)} files ({mixed} mixed)",
        file=stream,
    )
    for label, n in counts.most_common():
        print(f"  {label:24s} {n:6d} {100 * n / total:5.1f}%", file=stream)
    print("  decided by:", dict(by_stage), file=stream)


def main(argv=None):
    args = build_parser().parse_args(argv)

    files = []
    for path in args.paths:
        files.extend(collect_paths(path))
    if not files:
        print("no PDF or image found in the given paths", file=sys.stderr)
        return 1

    try:
        classifier = PageClassifier(
            model_path=args.model,
            use_model=not args.no_model,
            max_image_cover=args.max_image_cover,
            threads=args.threads,
        )
    except (ImportError, FileNotFoundError) as error:
        print(f"{error}\nOr run --no-model for the deterministic stage only.",
              file=sys.stderr)
        return 1

    if args.inspect:
        payload = [
            {
                "file": str(f),
                "pages": [p.to_dict() for p in classifier.inspect(f)],
            }
            for f in files
            if f.suffix.lower() == ".pdf"
        ]
    else:
        payload = [classifier.classify(f).to_dict() for f in files]

    text = json.dumps(payload, indent=1, ensure_ascii=False)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(text)
    if args.json:
        print(text)
    elif not args.output:
        if args.inspect:
            print_inspection(payload, sys.stdout)
        else:
            print_report(payload, sys.stdout)
    if not args.quiet and not args.inspect:
        sys.stdout.flush()
        summarize(payload, sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
