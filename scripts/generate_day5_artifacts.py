from __future__ import annotations

from pathlib import Path
import sys
import json

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.config import FIGURES_DIR, METRICS_DIR
from src.data import load_bundle
from src.evaluation import simulate_new_user_transition


def main() -> None:
    bundle = load_bundle(download_if_missing=False)
    transition_df, detail_df = simulate_new_user_transition(
        bundle.ratings,
        bundle.items,
        thresholds=(0, 1, 3, 5, 10, 15),
        max_users=200,
    )

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    transition_df.to_csv(METRICS_DIR / "day5_cold_start_transition.csv", index=False)
    detail_df.to_csv(METRICS_DIR / "day5_cold_start_transition_user_detail.csv", index=False)

    model_comparison_path = METRICS_DIR / "day4_model_comparison.csv"
    new_item_strategy_df = pd.DataFrame()
    if model_comparison_path.exists():
        model_comparison = pd.read_csv(model_comparison_path)
        new_item_strategy_df = model_comparison.loc[
            model_comparison["model_name"].isin(
                [
                    "content_tfidf_metadata",
                    "switching_hybrid_lt_15",
                    "weighted_hybrid_alpha_0.85",
                    "popularity_baseline",
                ]
            ),
            ["model_name", "precision@10", "coverage", "cold_start_item_share"],
        ].copy()
        new_item_strategy_df["recommended_stage"] = new_item_strategy_df["model_name"].map(
            {
                "content_tfidf_metadata": "metadata-only launch",
                "switching_hybrid_lt_15": "early traction",
                "weighted_hybrid_alpha_0.85": "steady state",
                "popularity_baseline": "fallback baseline",
            }
        )
        new_item_strategy_df = new_item_strategy_df.sort_values(
            "cold_start_item_share",
            ascending=False,
        ).reset_index(drop=True)
        new_item_strategy_df.to_csv(METRICS_DIR / "day5_new_item_strategy.csv", index=False)

    summary_payload = {
        "best_precision_stage": transition_df.sort_values("precision@10", ascending=False).iloc[0].to_dict(),
        "recommended_cf_switch_threshold": int(
            transition_df.loc[transition_df["stage"] == "hybrid", "observed_ratings"].iloc[0]
        ),
        "recommended_hybrid_threshold_for_precision_recovery": int(
            transition_df.sort_values("precision@10", ascending=False).iloc[0]["observed_ratings"]
        ),
    }
    with (METRICS_DIR / "day5_cold_start_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary_payload, handle, ensure_ascii=False, indent=2)

    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    sns.lineplot(
        data=transition_df,
        x="observed_ratings",
        y="precision@10",
        hue="stage",
        marker="o",
        ax=axes[0],
    )
    axes[0].set_title("Precision@10 by onboarding depth")
    axes[0].set_xlabel("Observed ratings")
    axes[0].set_ylabel("Precision@10")

    sns.lineplot(
        data=transition_df,
        x="observed_ratings",
        y="coverage",
        hue="stage",
        marker="o",
        ax=axes[1],
        legend=False,
    )
    axes[1].set_title("Coverage by onboarding depth")
    axes[1].set_xlabel("Observed ratings")
    axes[1].set_ylabel("Coverage")

    fig.tight_layout()
    figure_path = FIGURES_DIR / "day5_cold_start_transition.png"
    fig.savefig(figure_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

    print(transition_df.to_string(index=False))
    print(f"\nSaved figure: {figure_path}")


if __name__ == "__main__":
    main()
