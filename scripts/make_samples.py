"""Generate the demo sample files in data/samples/.

Everything produced here is SYNTHETIC: the experiment curves come from a seeded random process, not from any
real training run, and the PDF is original explanatory text written for this project (MIT licensed).

    python scripts/make_samples.py
"""

from __future__ import annotations

import csv
import json
import math
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SAMPLES = ROOT / "data" / "samples"

PDF_PAGES = [
    (
        "A Short Primer on Graph Neural Networks",
        [
            "Synthetic sample document written for the MCP Research Agent demo. MIT licensed.",
            "",
            "1. Introduction",
            "Graph neural networks (GNNs) learn representations of nodes, edges and whole graphs by",
            "repeatedly exchanging information between neighbouring nodes. This process is called",
            "message passing. In each layer, every node aggregates messages from its neighbours,",
            "combines them with its own state, and applies a learned transformation.",
            "",
            "2. Graph Convolutional Networks",
            "A graph convolutional network (GCN) averages neighbour features using a symmetric",
            "normalisation based on node degrees, followed by a linear layer and a non-linearity.",
            "Because every neighbour is weighted by a fixed coefficient, GCN layers are cheap to",
            "compute and work well on homophilous graphs where connected nodes share labels.",
        ],
    ),
    (
        "Attention and Sampling",
        [
            "3. Graph Attention Networks",
            "A graph attention network (GAT) replaces the fixed coefficients of a GCN with learned",
            "attention weights. Each node scores its neighbours with a small attention function and",
            "normalises the scores with a softmax. Multiple attention heads are usually concatenated",
            "or averaged. Attention lets the model down-weight noisy or irrelevant neighbours.",
            "",
            "4. Neighbourhood Sampling",
            "GraphSAGE-style models sample a fixed number of neighbours per node instead of using the",
            "full neighbourhood. Sampling bounds memory use on very large graphs and enables",
            "inductive learning on nodes that were not seen during training.",
        ],
    ),
    (
        "Limitations and Evaluation",
        [
            "5. Known Limitations of Message Passing",
            "Over-smoothing: as more layers are stacked, node representations become increasingly",
            "similar, which hurts classification accuracy in deep GNNs.",
            "Over-squashing: information from distant nodes must pass through narrow bottlenecks,",
            "so long-range dependencies are compressed into fixed-size vectors.",
            "Expressiveness: standard message passing is at most as powerful as the 1-WL graph",
            "isomorphism test, so some non-isomorphic graphs cannot be distinguished.",
            "",
            "6. Evaluation Practice",
            "Results on small citation benchmarks vary noticeably with the random seed. Reports should",
            "therefore give the mean and standard deviation over several seeds, state whether lower",
            "or higher values of each metric are better, and use the same data splits for every model.",
        ],
    ),
]


def _pdf_escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def build_pdf(pages: list[tuple[str, list[str]]], title: str) -> bytes:
    objects: list[bytes] = []
    n_pages = len(pages)
    page_ids = [4 + 2 * i for i in range(n_pages)]
    objects.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    objects.append(
        f"<< /Type /Pages /Kids [{' '.join(f'{p} 0 R' for p in page_ids)}] /Count {n_pages} >>".encode()
    )
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    for heading, lines in pages:
        ops = ["BT", "/F1 16 Tf", "72 740 Td", f"({_pdf_escape(heading)}) Tj", "/F1 11 Tf", "0 -28 Td"]
        for line in lines:
            ops += [f"({_pdf_escape(line)}) Tj", "0 -16 Td"]
        ops.append("ET")
        stream = "\n".join(ops).encode("latin-1")
        content_id = len(objects) + 2
        objects.append(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents {content_id} 0 R "
            f"/Resources << /Font << /F1 3 0 R >> >> >>".encode()
        )
        objects.append(
            b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream"
        )
    info_id = len(objects) + 1
    objects.append(
        f"<< /Title ({_pdf_escape(title)}) /Producer (mcp-research-agent sample generator) >>".encode()
    )
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for i, obj in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n".encode() + obj + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    out += b"".join(f"{o:010d} 00000 n \n".encode() for o in offsets)
    out += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R /Info {info_id} 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    return bytes(out)


def curve(rng: random.Random, epoch: int, start: float, end: float, rate: float, noise: float) -> float:
    return end + (start - end) * math.exp(-rate * epoch) + rng.gauss(0, noise)


def synthetic_runs(model: str, seeds: list[int], params: dict[str, float], epochs: int = 20) -> list[dict]:
    rows = []
    for seed in seeds:
        rng = random.Random(f"{model}-{seed}")
        for epoch in range(1, epochs + 1):
            train_loss = curve(rng, epoch, 1.95, params["train_loss_end"], 0.25, 0.01)
            val_loss = curve(rng, epoch, 1.95, params["val_loss_end"], 0.22, 0.015)
            val_acc = curve(rng, epoch, 0.15, params["val_acc_end"], 0.30, 0.006)
            train_acc = min(curve(rng, epoch, 0.15, params["val_acc_end"] + 0.12, 0.30, 0.004), 0.999)
            rows.append(
                {
                    "experiment_id": f"{model.lower()}-s{seed}",
                    "model": model,
                    "dataset": "synthetic-citation-graph",
                    "seed": seed,
                    "epoch": epoch,
                    "train_loss": round(train_loss, 4),
                    "val_loss": round(val_loss, 4),
                    "train_accuracy": round(train_acc, 4),
                    "val_accuracy": round(val_acc, 4),
                }
            )
    return rows


def main() -> None:
    SAMPLES.mkdir(parents=True, exist_ok=True)
    (SAMPLES / "gnn_primer.pdf").write_bytes(build_pdf(PDF_PAGES, "A Short Primer on Graph Neural Networks"))

    rows = synthetic_runs(
        "GCN", [0, 1, 2], {"train_loss_end": 0.32, "val_loss_end": 0.71, "val_acc_end": 0.79}
    )
    rows += synthetic_runs(
        "GAT", [0, 1, 2], {"train_loss_end": 0.28, "val_loss_end": 0.64, "val_acc_end": 0.81}
    )
    with (SAMPLES / "synthetic_gnn_experiments.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    sage = synthetic_runs(
        "GraphSAGE", [0, 1], {"train_loss_end": 0.30, "val_loss_end": 0.69, "val_acc_end": 0.80}
    )
    runs = []
    for seed in (0, 1):
        history = [
            {k: r[k] for k in ("epoch", "train_loss", "val_loss", "val_accuracy")}
            for r in sage
            if r["seed"] == seed
        ]
        runs.append(
            {
                "experiment_id": f"graphsage-s{seed}",
                "model": "GraphSAGE",
                "dataset": "synthetic-citation-graph",
                "seed": seed,
                "history": history,
            }
        )
    payload = {
        "note": "SYNTHETIC demo data generated by scripts/make_samples.py; not real benchmark results.",
        "runs": runs,
    }
    (SAMPLES / "synthetic_graphsage_runs.json").write_text(json.dumps(payload, indent=1))
    print(f"Wrote samples to {SAMPLES}")


if __name__ == "__main__":
    main()
