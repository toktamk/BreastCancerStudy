from __future__ import annotations

import argparse, json
from pathlib import Path
import pandas as pd

from src.eval.recalibration import LogisticRecalibrator

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--val", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--p-col", default="risk_prob_horizon")
    ap.add_argument("--y-col", default="y_true")
    ap.add_argument("--out-recal", required=True)
    ap.add_argument("--out-test", required=True)
    args = ap.parse_args()

    dfv = pd.read_parquet(args.val)
    dft = pd.read_parquet(args.test)

    rec = LogisticRecalibrator.fit(dfv[args.y_col].to_numpy(), dfv[args.p_col].to_numpy())
    Path(args.out_recal).write_text(json.dumps(rec.to_dict(), indent=2), encoding="utf-8")

    dft[args.p_col + "_recal"] = rec.transform(dft[args.p_col].to_numpy()).astype("float32")
    Path(args.out_test).parent.mkdir(parents=True, exist_ok=True)
    dft.to_parquet(args.out_test, index=False)

    print("[OK] wrote recal:", args.out_recal)
    print("[OK] wrote recal test preds:", args.out_test)

if __name__ == "__main__":
    main()
