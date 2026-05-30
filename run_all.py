"""Run training, evaluation, and plot generation in one command."""

from argparse import ArgumentParser
import os
from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET = Path("diagnosed_cbc_data_v4.csv")
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


def run_step(command):
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
    preferred = [DEFAULT_DATASET, Path("anemia.csv")]
    discovered = [path for path in preferred if (PROJECT_ROOT / path).exists()]
    for path in sorted(PROJECT_ROOT.glob("*.csv")):
        path = path.relative_to(PROJECT_ROOT)
        if path not in discovered:
            discovered.append(path)
    return discovered


def describe_dataset(path):
    if path.name == "diagnosed_cbc_data_v4.csv":
        return f"{path} - new 14-feature CBC diagnosis dataset"
    if path.name == "anemia.csv":
        return f"{path} - old 4-feature anemia dataset"
    return str(path)


def resolve_run_options(args):
    csv_files = discover_csv_files()
    if args.data is None:
        if not csv_files:
            raise FileNotFoundError("No CSV datasets found in the project directory.")
        default_index = csv_files.index(DEFAULT_DATASET) if DEFAULT_DATASET in csv_files else 0
        args.data = choose_from_menu(
            "Select the dataset for training:",
            [(describe_dataset(path), path) for path in csv_files],
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

    return args


def parse_args():
    parser = ArgumentParser(description="Train, evaluate, and generate plots for the CBC VAE project.")
    parser.add_argument("--data", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--plots-dir", type=Path, default=Path("results/plots"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--latent-dim", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--beta", type=float, default=0.1, help="Final KL loss weight.")
    parser.add_argument(
        "--kl-warmup-epochs",
        type=int,
        default=50,
        help="Linearly increase KL weight to beta over this many epochs. Use 0 to disable.",
    )
    parser.add_argument("--threshold", type=float, default=0.97)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--no-cleaning", action="store_true", help="Disable CBC value-range data cleaning.")
    parser.add_argument("--cleaning", choices=["ask", "range", "none"], default="ask")
    parser.add_argument("--target", choices=["anemia", "abnormal"], default="anemia")
    parser.add_argument("--skip-threshold-sweep", action="store_true", help="Skip threshold comparison report.")
    return parser.parse_args()


def main():
    configure_runtime_environment()
    ensure_dependencies_available()
    args = resolve_run_options(parse_args())
    model_path = args.output_dir / "vae_anemia.pt"
    outputs_path = args.output_dir / "evaluation_outputs.npz"

    common = [
        "--data",
        str(args.data),
        "--seed",
        str(args.seed),
        "--device",
        args.device,
        "--target",
        args.target,
    ]
    if args.cleaning == "none":
        common.append("--no-cleaning")

    run_step(
        [
            sys.executable,
            "src/train.py",
            *common,
            "--output-dir",
            str(args.output_dir),
            "--epochs",
            str(args.epochs),
            "--batch-size",
            str(args.batch_size),
            "--latent-dim",
            str(args.latent_dim),
            "--lr",
            str(args.lr),
            "--beta",
            str(args.beta),
            "--kl-warmup-epochs",
            str(args.kl_warmup_epochs),
            "--threshold",
            str(args.threshold),
        ]
    )
    run_step(
        [
            sys.executable,
            "src/evaluate.py",
            *common,
            "--model",
            str(model_path),
            "--output-dir",
            str(args.output_dir),
        ]
    )
    run_step(
        [
            sys.executable,
            "src/visualise.py",
            *common,
            "--model",
            str(model_path),
            "--outputs",
            str(outputs_path),
            "--output-dir",
            str(args.plots_dir),
        ]
    )
    if not args.skip_threshold_sweep:
        run_step(
            [
                sys.executable,
                "src/threshold_sweep.py",
                *common,
                "--model",
                str(model_path),
                "--output-dir",
                str(args.output_dir),
            ]
        )

    print("\nAll done.")
    print("Model and evaluation outputs: {}".format(args.output_dir))
    print("Plots: {}".format(args.plots_dir))


if __name__ == "__main__":
    main()
