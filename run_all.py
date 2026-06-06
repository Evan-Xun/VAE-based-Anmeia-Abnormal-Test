"""Run single experiments or a full experiment suite for the CBC VAE project."""

from argparse import ArgumentParser
from itertools import product
import os
from pathlib import Path
import subprocess
import sys
from typing import Dict, Iterable, List


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET = Path("cbc_8_features_reports_only.csv")
REQUIRED_PACKAGES = {
    "matplotlib": "matplotlib",
    "numpy": "numpy",
    "pandas": "pandas",
    "sklearn": "scikit-learn",
    "torch": "torch",
}


def configure_runtime_environment():
    matplotlib_cache = PROJECT_ROOT / ".cache" / "matplotlib"
    matplotlib_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache))


def run_step(command: List[str]):
    env = os.environ.copy()

    print("\n" + "=" * 70)
    print("Running: " + " ".join(str(part) for part in command))
    print("=" * 70)
    sys.stdout.flush()
    subprocess.run(command, check=True, cwd=PROJECT_ROOT, env=env)


def ensure_dependencies_available():
    missing = []
    for module_name, package_name in REQUIRED_PACKAGES.items():
        try:
            __import__(module_name)
        except ModuleNotFoundError:
            missing.append(package_name)

    if missing:
        packages = " ".join(sorted(missing))
        raise SystemExit(
            "Missing Python packages for this interpreter: "
            f"{packages}\n\n"
            "Create/select a project virtual environment, then install dependencies with:\n"
            f"  {sys.executable} -m pip install -r requirements.txt\n\n"
            "In PyCharm, set the project interpreter to that virtual environment's python."
        )


def choose_from_menu(title, options, default_index=0):
    if not sys.stdin.isatty():
        return options[default_index][1]

    print("\n" + title)
    for index, (label, _) in enumerate(options, start=1):
        default_mark = " [default]" if index - 1 == default_index else ""
        print(f"{index}. {label}{default_mark}")

    while True:
        choice = input("Choose an option: ").strip()
        if not choice:
            return options[default_index][1]
        if choice.isdigit() and 1 <= int(choice) <= len(options):
            return options[int(choice) - 1][1]
        print("Please enter a valid option number.")


def discover_csv_files():
    preferred = [DEFAULT_DATASET]
    discovered = [path for path in preferred if (PROJECT_ROOT / path).exists()]
    for path in sorted(PROJECT_ROOT.glob("*.csv")):
        path = path.relative_to(PROJECT_ROOT)
        if path not in discovered:
            discovered.append(path)
    return discovered


def describe_dataset(path):
    if path.name == "cbc_8_features_reports_only.csv":
        return f"{path} - reports-only 8-feature Healthy/Anemia dataset"
    return str(path)


def parse_csv_list(raw: str, cast):
    return [cast(part.strip()) for part in raw.split(",") if part.strip()]


def slug(value) -> str:
    text = str(value)
    return (
        text.replace(" ", "_")
        .replace(".", "p")
        .replace(">=", "ge")
        .replace("<=", "le")
        .replace("/", "_")
        .replace(",", "_")
    )


def summarize_metric_table(frame, group_columns: List[str], metric_columns: List[str]):
    import pandas as pd

    grouped = frame.groupby(group_columns, dropna=False)
    rows = []
    for keys, group in grouped:
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {column: value for column, value in zip(group_columns, keys)}
        row["runs"] = len(group)
        for metric in metric_columns:
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_std"] = float(group[metric].std(ddof=0))
        rows.append(row)
    return pd.DataFrame(rows)


def resolve_run_options(args):
    data_sources = discover_csv_files()
    if args.data is None:
        if not data_sources:
            raise FileNotFoundError("No supported datasets found in the project directory.")
        if DEFAULT_DATASET in data_sources:
            default_index = data_sources.index(DEFAULT_DATASET)
        else:
            default_index = 0
        args.data = choose_from_menu(
            "Select the dataset for training:",
            [(describe_dataset(path), path) for path in data_sources],
            default_index=default_index,
        )

    if args.no_cleaning:
        args.cleaning = "none"

    if args.cleaning == "ask":
        args.cleaning = choose_from_menu(
            "Select the data cleaning method:",
            [
                ("CBC value-range cleaning (recommended)", "range"),
                ("No value-range cleaning", "none"),
            ],
            default_index=0,
        )

    args.seeds = args.seeds or [args.seed]
    args.betas = args.betas or [args.beta]
    args.kl_warmup_epochs_list = args.kl_warmup_epochs_list or [args.kl_warmup_epochs]
    args.latent_dims = args.latent_dims or [args.latent_dim]
    args.targets = args.targets or [args.target]
    args.cleaning_options = args.cleaning_options or [args.cleaning]
    args.feature_sets = args.feature_sets or ["supervised_only", "vae_only", "hybrid"]
    args.threshold_methods = args.threshold_methods or [args.threshold_method]

    return args


def parse_args():
    parser = ArgumentParser(description="Train, evaluate, and compare CBC VAE experiments.")
    parser.add_argument("--data", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--plots-dir", type=Path, default=Path("results/plots"))
    parser.add_argument("--study-name", default="experiment_suite")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--latent-dim", type=int, default=2)
    parser.add_argument("--latent-dims", type=int, nargs="+", default=None)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--beta", type=float, default=0.1, help="Final KL loss weight.")
    parser.add_argument("--betas", type=float, nargs="+", default=None)
    parser.add_argument(
        "--kl-warmup-epochs",
        type=int,
        default=50,
        help="Linearly increase KL weight to beta over this many epochs. Use 0 to disable.",
    )
    parser.add_argument("--kl-warmup-epochs-list", type=int, nargs="+", default=None)
    parser.add_argument("--threshold", type=float, default=0.97)
    parser.add_argument("--threshold-method", default="validation_f1")
    parser.add_argument("--threshold-methods", nargs="+", default=None)
    parser.add_argument("--train-quantile", type=float, default=95.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--no-cleaning", action="store_true", help="Disable CBC value-range data cleaning.")
    parser.add_argument("--cleaning", choices=["ask", "range", "none"], default="ask")
    parser.add_argument("--cleaning-options", nargs="+", choices=["range", "none"], default=None)
    parser.add_argument("--target", choices=["anemia", "abnormal"], default="anemia")
    parser.add_argument("--targets", nargs="+", choices=["anemia", "abnormal"], default=None)
    parser.add_argument("--skip-threshold-sweep", action="store_true", help="Skip threshold comparison report.")
    parser.add_argument("--skip-hybrid", action="store_true", help="Skip supervised and hybrid comparisons.")
    parser.add_argument("--skip-plots", action="store_true", help="Skip plot generation.")
    parser.add_argument(
        "--hybrid-min-recall",
        type=float,
        default=0.95,
        help="Validation recall floor for the hybrid high-recall threshold.",
    )
    parser.add_argument(
        "--feature-sets",
        nargs="+",
        choices=["supervised_only", "vae_only", "hybrid"],
        default=None,
        help="Feature sets for supervised comparison runs.",
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--val-size", type=float, default=0.2)
    return parser.parse_args()


def build_run_matrix(args) -> Iterable[Dict[str, object]]:
    for target, cleaning, beta, warmup, latent_dim, seed, threshold_method in product(
        args.targets,
        args.cleaning_options,
        args.betas,
        args.kl_warmup_epochs_list,
        args.latent_dims,
        args.seeds,
        args.threshold_methods,
    ):
        yield {
            "target": target,
            "cleaning": cleaning,
            "beta": beta,
            "warmup": warmup,
            "latent_dim": latent_dim,
            "seed": seed,
            "threshold_method": threshold_method,
        }


def output_dir_for_run(args, config: Dict[str, object], single_run_mode: bool) -> Path:
    if single_run_mode:
        return args.output_dir

    return (
        args.output_dir
        / args.study_name
        / f"target_{slug(config['target'])}"
        / f"cleaning_{slug(config['cleaning'])}"
        / f"beta_{slug(config['beta'])}"
        / f"warmup_{slug(config['warmup'])}"
        / f"latent_{slug(config['latent_dim'])}"
        / f"threshold_{slug(config['threshold_method'])}"
        / f"seed_{slug(config['seed'])}"
    )


def common_args(args, config: Dict[str, object]) -> List[str]:
    common = [
        "--data",
        str(args.data),
        "--seed",
        str(config["seed"]),
        "--device",
        args.device,
        "--target",
        str(config["target"]),
        "--test-size",
        str(args.test_size),
        "--val-size",
        str(args.val_size),
    ]
    if config["cleaning"] == "none":
        common.append("--no-cleaning")
    return common


def run_single_experiment(args, config: Dict[str, object], run_output_dir: Path):
    plots_dir = run_output_dir / "plots"
    model_path = run_output_dir / "vae_anemia.pt"
    outputs_path = run_output_dir / "evaluation_outputs.npz"
    shared = common_args(args, config)

    run_step(
        [
            sys.executable,
            "src/train.py",
            *shared,
            "--output-dir",
            str(run_output_dir),
            "--epochs",
            str(args.epochs),
            "--batch-size",
            str(args.batch_size),
            "--latent-dim",
            str(config["latent_dim"]),
            "--lr",
            str(args.lr),
            "--beta",
            str(config["beta"]),
            "--kl-warmup-epochs",
            str(config["warmup"]),
            "--threshold",
            str(args.threshold),
        ]
    )
    run_step(
        [
            sys.executable,
            "src/evaluate.py",
            *shared,
            "--model",
            str(model_path),
            "--output-dir",
            str(run_output_dir),
            "--threshold-method",
            str(config["threshold_method"]),
            "--train-quantile",
            str(args.train_quantile),
            "--min-recall",
            str(args.hybrid_min_recall),
        ]
    )
    if not args.skip_plots:
        run_step(
            [
                sys.executable,
                "src/visualise.py",
                *shared,
                "--model",
                str(model_path),
                "--outputs",
                str(outputs_path),
                "--output-dir",
                str(plots_dir),
            ]
        )
    if not args.skip_threshold_sweep:
        run_step(
            [
                sys.executable,
                "src/threshold_sweep.py",
                *shared,
                "--model",
                str(model_path),
                "--output-dir",
                str(run_output_dir),
                "--min-recall",
                str(args.hybrid_min_recall),
            ]
        )
    if not args.skip_hybrid:
        run_step(
            [
                sys.executable,
                "src/hybrid_train.py",
                *shared,
                "--output-dir",
                str(run_output_dir),
                "--epochs",
                str(args.epochs),
                "--batch-size",
                str(args.batch_size),
                "--latent-dim",
                str(config["latent_dim"]),
                "--lr",
                str(args.lr),
                "--beta",
                str(config["beta"]),
                "--kl-warmup-epochs",
                str(config["warmup"]),
                "--min-recall",
                str(args.hybrid_min_recall),
                "--feature-sets",
                *args.feature_sets,
            ]
        )


def aggregate_runs(args, run_records: List[Dict[str, object]]):
    import pandas as pd

    records = pd.DataFrame(run_records)
    aggregate_dir = args.output_dir if len(run_records) == 1 else args.output_dir / args.study_name
    aggregate_dir.mkdir(parents=True, exist_ok=True)
    records.to_csv(aggregate_dir / "experiment_manifest.csv", index=False)

    baseline_rows = []
    hybrid_rows = []
    threshold_rows = []
    for record in run_records:
        base = Path(record["output_dir"])
        baseline_metrics = base / "baseline_metrics.csv"
        hybrid_metrics = base / "hybrid_metrics.csv"
        threshold_sweep = base / "threshold_sweep.csv"

        if baseline_metrics.exists():
            frame = pd.read_csv(baseline_metrics)
            test_rows = frame[frame["split"] == "test"].copy()
            for column, value in record.items():
                if column != "output_dir":
                    test_rows[column] = value
            baseline_rows.append(test_rows)

        if hybrid_metrics.exists():
            frame = pd.read_csv(hybrid_metrics)
            for column, value in record.items():
                if column != "output_dir":
                    frame[column] = value
            hybrid_rows.append(frame)

        if threshold_sweep.exists():
            frame = pd.read_csv(threshold_sweep)
            for column, value in record.items():
                if column != "output_dir":
                    frame[column] = value
            threshold_rows.append(frame)

    if baseline_rows:
        baseline = pd.concat(baseline_rows, ignore_index=True)
        baseline.to_csv(aggregate_dir / "baseline_runs.csv", index=False)
        baseline_summary = summarize_metric_table(
            baseline,
            ["target", "cleaning", "beta", "warmup", "latent_dim", "threshold_method"],
            ["auroc", "auprc", "precision", "recall", "f1", "specificity"],
        )
        baseline_summary.to_csv(aggregate_dir / "baseline_summary.csv", index=False)

    if hybrid_rows:
        hybrid = pd.concat(hybrid_rows, ignore_index=True)
        hybrid.to_csv(aggregate_dir / "hybrid_runs.csv", index=False)
        hybrid_summary = summarize_metric_table(
            hybrid,
            ["target", "cleaning", "beta", "warmup", "latent_dim", "feature_set", "model", "threshold_method"],
            ["auroc", "auprc", "precision", "recall", "f1", "specificity"],
        )
        hybrid_summary.to_csv(aggregate_dir / "hybrid_summary.csv", index=False)

    if threshold_rows:
        thresholds = pd.concat(threshold_rows, ignore_index=True)
        thresholds.to_csv(aggregate_dir / "threshold_sweep_runs.csv", index=False)

    print("\nAll done.")
    print(f"Experiment outputs: {aggregate_dir}")


def main():
    configure_runtime_environment()
    ensure_dependencies_available()
    args = resolve_run_options(parse_args())

    run_matrix = list(build_run_matrix(args))
    single_run_mode = len(run_matrix) == 1
    run_records = []
    for config in run_matrix:
        run_output_dir = output_dir_for_run(args, config, single_run_mode)
        run_single_experiment(args, config, run_output_dir)
        run_records.append({**config, "output_dir": str(run_output_dir)})

    aggregate_runs(args, run_records)


if __name__ == "__main__":
    main()
