from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from tgfm.data import RML2018Dataset, create_rml2018_split
from tgfm.models import build_teacher
from tgfm.utils import load_config, save_config, summarize_snr_acc, write_csv, write_json, make_run_dir
from train_raw_low_snr_expert import RawLowSNRExpert, ReliabilityRouter, route_from_snr


ROUTE_NAMES = ["very_low", "low", "edge", "reliable"]


def update_acc(stats, logits, labels, snrs) -> None:
    pred = logits.argmax(-1)
    ok = pred.eq(labels)
    for i in range(labels.numel()):
        s = int(float(snrs[i]))
        stats[s][0] += int(ok[i].item())
        stats[s][1] += 1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--teacher-run", required=True)
    ap.add_argument("--router-run", required=True)
    ap.add_argument("--expert-run", required=True)
    ap.add_argument("--output-tag", default="")
    args = ap.parse_args()

    cfg = load_config(args.config)
    create_rml2018_split(cfg["paths"]["rml2018_h5"], cfg["paths"]["split_path"], int(cfg["seed"]), float(cfg["data"]["train_ratio"]), float(cfg["data"]["val_ratio"]))
    device = torch.device(cfg.get("device", "cuda") if torch.cuda.is_available() else "cpu")
    tag = f"_{args.output_tag}" if args.output_tag else ""
    run_dir = make_run_dir(cfg["paths"]["output_root"], f"v8_raw_expert_diagnostics{tag}")
    save_config({**cfg, "args": vars(args)}, run_dir / "config.yaml")

    teacher = build_teacher(cfg).to(device)
    teacher.load_state_dict(torch.load(Path(args.teacher_run) / "best.pt", map_location="cpu")["model"])
    teacher.eval()
    router = ReliabilityRouter(int(cfg["teacher"]["z_dim"])).to(device)
    router.load_state_dict(torch.load(Path(args.router_run) / "best.pt", map_location="cpu")["router"])
    router.eval()

    ckpt = torch.load(Path(args.expert_run) / "best.pt", map_location="cpu")
    expert_args = ckpt.get("args", {})
    expert = RawLowSNRExpert(
        int(cfg["data"]["num_classes"]),
        width=int(expert_args.get("width", 384)),
        hidden=int(expert_args.get("hidden", 768)),
        dropout=float(expert_args.get("dropout", 0.20)),
        branch_mode=str(expert_args.get("branch_mode", "time_freq")),
    ).to(device)
    expert.load_state_dict(ckpt["expert"])
    expert.eval()

    ds = RML2018Dataset(cfg["paths"]["rml2018_h5"], cfg["paths"]["split_path"], "test", cfg["data"]["signal_length"])
    loader = DataLoader(
        ds,
        batch_size=int(cfg["eval"]["batch_size"]),
        shuffle=False,
        drop_last=False,
        num_workers=int(cfg["eval"]["num_workers"]),
        pin_memory=device.type == "cuda",
        persistent_workers=int(cfg["eval"]["num_workers"]) > 0,
    )

    stats = {m: defaultdict(lambda: [0, 0]) for m in ["direct", "hybrid"]}
    route_counts = defaultdict(lambda: [0, 0, 0, 0])
    route_conf = [[0 for _ in range(4)] for _ in range(4)]
    high = {
        "total": 0,
        "pred_changed": 0,
        "direct_correct": 0,
        "direct_correct_hybrid_wrong": 0,
        "kl_sum": 0.0,
    }

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            x = batch["x"].to(device, non_blocking=True)
            y = batch["label"].to(device, non_blocking=True)
            snr = batch["snr"].to(device, non_blocking=True)
            z = teacher.encode(x)
            direct_logits = teacher.classifier(z)
            expert_logits = expert(x)
            pred_route = router(z).argmax(dim=-1)
            oracle_route = route_from_snr(snr)
            hybrid_logits = direct_logits.clone()
            mask = pred_route.le(2)
            if bool(mask.any()):
                hybrid_logits[mask] = expert_logits[mask]

            update_acc(stats["direct"], direct_logits, y, snr)
            update_acc(stats["hybrid"], hybrid_logits, y, snr)
            for i in range(y.numel()):
                s = int(float(snr[i]))
                route_counts[s][int(pred_route[i])] += 1
                route_conf[int(oracle_route[i])][int(pred_route[i])] += 1

            high_mask = snr.ge(12)
            if bool(high_mask.any()):
                d_logits = direct_logits[high_mask]
                h_logits = hybrid_logits[high_mask]
                yy = y[high_mask]
                d_pred = d_logits.argmax(dim=-1)
                h_pred = h_logits.argmax(dim=-1)
                d_correct = d_pred.eq(yy)
                high["total"] += int(yy.numel())
                high["pred_changed"] += int(d_pred.ne(h_pred).sum())
                high["direct_correct"] += int(d_correct.sum())
                high["direct_correct_hybrid_wrong"] += int((d_correct & h_pred.ne(yy)).sum())
                kl = F.kl_div(F.log_softmax(h_logits, dim=-1), F.softmax(d_logits, dim=-1), reduction="sum")
                high["kl_sum"] += float(kl.detach().cpu())

            if batch_idx % 30 == 0:
                print(f"diagnostics batch={batch_idx}/{len(loader)}", flush=True)

    rows = []
    summary = {}
    for method in ["direct", "hybrid"]:
        by_snr = {int(k): v[0] / max(1, v[1]) for k, v in sorted(stats[method].items())}
        summary[method] = summarize_snr_acc(by_snr)
        for snr, acc in by_snr.items():
            rows.append({"method": method, "snr": snr, "acc": acc, "correct": stats[method][snr][0], "total": stats[method][snr][1]})
    write_csv(run_dir / "accuracy_by_snr.csv", rows)
    write_json(run_dir / "accuracy_overall.json", summary)

    route_rows = []
    for snr, counts in sorted(route_counts.items()):
        total = max(1, sum(counts))
        row = {"snr": snr, "total": total}
        for i, name in enumerate(ROUTE_NAMES):
            row[f"{name}_count"] = counts[i]
            row[f"{name}_ratio"] = counts[i] / total
        route_rows.append(row)
    write_csv(run_dir / "router_distribution_by_snr.csv", route_rows)
    write_json(run_dir / "router_confusion_oracle_rows_pred_cols.json", {"route_names": ROUTE_NAMES, "matrix": route_conf})

    high_total = max(1, high["total"])
    high_direct_correct = max(1, high["direct_correct"])
    high_metrics = {
        "high_total": high["total"],
        "prediction_change_rate": high["pred_changed"] / high_total,
        "direct_correct_to_hybrid_wrong_rate": high["direct_correct_hybrid_wrong"] / high_direct_correct,
        "kl_direct_to_hybrid_mean": high["kl_sum"] / high_total,
        "note": "KL is KL(p_direct || p_hybrid) over high-SNR samples; feature cosine is N/A because raw expert changes logits, not teacher features.",
    }
    write_json(run_dir / "high_snr_fidelity.json", high_metrics)
    print(f"DIAGNOSTICS_OK run_dir={run_dir}", flush=True)


if __name__ == "__main__":
    main()
