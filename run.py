"""CLI for the Monte Carlo option pricer.

    python run.py price   --strikes 95,100,105
    python run.py compare --n-paths 100000

`compare` also runs mc.R under Rscript and mc.cpp compiled with g++, if it can
find them (the RSCRIPT env var or the house R install path, g++ on PATH or the
CXX env var), and tabulates all three against the closed-form Black-Scholes
price. A missing toolchain is skipped, not an error.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile

import black_scholes as bs
import mc

ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_RSCRIPT = r"C:\Program Files\R\R-4.5.2\bin\Rscript.exe"


TABLE_HEADER = "K call call_se put put_se"


def format_table(rows: list[dict[str, float]]) -> list[str]:
    """The table format shared by run.py, mc.R and mc.cpp, so run.py compare can
    parse R and C++ stdout the same way it reads its own output.
    """
    lines = [TABLE_HEADER]
    for row in rows:
        lines.append("%.6f %.6f %.6f %.6f %.6f" % (
            row["K"], row["call"], row["call_se"], row["put"], row["put_se"]))
    return lines


def print_table(rows: list[dict[str, float]]) -> None:
    for line in format_table(rows):
        print(line)


def parse_table(text: str) -> list[dict[str, float]]:
    """Inverse of format_table. Tolerant of the header line and blank lines, since
    it is used to read whatever mc.R or mc.cpp printed to stdout.
    """
    rows = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        try:
            K, call, call_se, put, put_se = (float(p) for p in parts)
        except ValueError:
            continue
        rows.append({"K": K, "call": call, "call_se": call_se, "put": put, "put_se": put_se})
    return rows


def write_json(rows: list[dict[str, float]], path: str) -> None:
    with open(path, "w") as f:
        json.dump(rows, f, indent=2)


def find_rscript() -> str | None:
    env = os.environ.get("RSCRIPT")
    if env and os.path.exists(env):
        return env
    if os.path.exists(DEFAULT_RSCRIPT):
        return DEFAULT_RSCRIPT
    return shutil.which("Rscript")


def find_gxx() -> str | None:
    return os.environ.get("CXX") or shutil.which("g++")


def to_flags(params: dict) -> list[str]:
    flags = ["--s0", str(params["s0"]), "--r", str(params["r"]),
              "--sigma", str(params["sigma"]), "--T", str(params["T"]),
              "--n-steps", str(params["n_steps"]), "--n-paths", str(params["n_paths"]),
              "--strikes", params["strikes"], "--seed", str(params["seed"])]
    if params.get("antithetic"):
        flags.append("--antithetic")
    return flags


def run_r(params: dict) -> tuple[list[dict[str, float]] | None, str | None]:
    """Run mc.R with Rscript. Returns (rows, None) or (None, reason)."""
    rscript = find_rscript()
    if not rscript:
        return None, "Rscript not found"
    try:
        result = subprocess.run(
            [rscript, os.path.join(ROOT, "mc.R")] + to_flags(params),
            capture_output=True, text=True, timeout=180)
    except OSError as exc:
        return None, str(exc)
    if result.returncode != 0:
        return None, result.stderr.strip() or "mc.R exited with status %d" % result.returncode
    return parse_table(result.stdout), None


def run_cpp(params: dict) -> tuple[list[dict[str, float]] | None, str | None]:
    """Compile and run mc.cpp with g++. Returns (rows, None) or (None, reason)."""
    gxx = find_gxx()
    if not gxx:
        return None, "g++ not found"
    with tempfile.TemporaryDirectory() as tmp:
        exe = os.path.join(tmp, "mc.exe" if os.name == "nt" else "mc")
        build = subprocess.run(
            [gxx, "-O2", "-std=c++17", "-o", exe, os.path.join(ROOT, "mc.cpp")],
            capture_output=True, text=True)
        if build.returncode != 0:
            return None, "g++ build failed: " + build.stderr.strip()
        try:
            result = subprocess.run([exe] + to_flags(params),
                                    capture_output=True, text=True, timeout=180)
        except OSError as exc:
            return None, str(exc)
    if result.returncode != 0:
        return None, result.stderr.strip() or "mc exited with status %d" % result.returncode
    return parse_table(result.stdout), None


def parse_strikes(text: str) -> list[float]:
    return [float(s) for s in text.split(",")]


def params_from_args(args: argparse.Namespace) -> dict:
    return dict(s0=args.s0, r=args.r, sigma=args.sigma, T=args.T, n_steps=args.n_steps,
                n_paths=args.n_paths, strikes=args.strikes, seed=args.seed,
                antithetic=args.antithetic)


def cmd_price(args: argparse.Namespace) -> None:
    strikes = parse_strikes(args.strikes)
    rows = mc.price_european(args.s0, strikes, args.r, args.sigma, args.T,
                             args.n_steps, args.n_paths, args.seed, args.antithetic)
    print_table(rows)
    if args.json:
        write_json(rows, args.json)


def cmd_compare(args: argparse.Namespace) -> None:
    params = params_from_args(args)
    strikes = parse_strikes(args.strikes)

    bs_rows = {K: {"call": float(bs.price(args.s0, K, args.r, args.sigma, args.T, "call")),
                   "put": float(bs.price(args.s0, K, args.r, args.sigma, args.T, "put"))}
               for K in strikes}

    py_rows = mc.price_european(args.s0, strikes, args.r, args.sigma, args.T,
                                args.n_steps, args.n_paths, args.seed, args.antithetic)
    r_rows, r_err = run_r(params)
    cpp_rows, cpp_err = run_cpp(params)

    print("python %s  S0=%s r=%s sigma=%s T=%s n_steps=%s n_paths=%s seed=%s antithetic=%s"
          % (sys.version.split()[0], args.s0, args.r, args.sigma, args.T,
             args.n_steps, args.n_paths, args.seed, args.antithetic))
    print()
    print("%6s%16s%12s%10s%9s%12s%10s" % (
        "K", "method", "call", "call_se", "err/se", "put", "put_se"))

    for K in strikes:
        ref = bs_rows[K]
        for name, rows in (("python", py_rows), ("R", r_rows), ("c++", cpp_rows)):
            if rows is None:
                continue
            row = next(r for r in rows if abs(r["K"] - K) < 1e-9)
            err_se = abs(row["call"] - ref["call"]) / row["call_se"] if row["call_se"] > 0 else float("nan")
            print("%6.1f%16s%12.6f%10.6f%9.2f%12.6f%10.6f" % (
                K, name, row["call"], row["call_se"], err_se, row["put"], row["put_se"]))
        print("%6.1f%16s%12.6f%10s%9s%12.6f%10s" % (
            K, "black-scholes", ref["call"], "-", "-", ref["put"], "-"))

    if r_rows is None:
        print("\n(R skipped: %s)" % r_err)
    if cpp_rows is None:
        print("(C++ skipped: %s)" % cpp_err)

    if args.json:
        write_json(py_rows, args.json)


def add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--s0", type=float, default=100.0, help="spot price")
    p.add_argument("--r", type=float, default=0.03, help="risk-free rate")
    p.add_argument("--sigma", type=float, default=0.2, help="annual volatility")
    p.add_argument("--T", type=float, default=1.0, help="time to expiry in years")
    p.add_argument("--n-steps", type=int, default=252, help="simulated time steps")
    p.add_argument("--n-paths", type=int, default=10000, help="simulated paths")
    p.add_argument("--strikes", default="95,100,105", help="comma-separated strikes")
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--antithetic", action="store_true", help="use antithetic variates")
    p.add_argument("--json", default=None, help="write the table to this path as JSON")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("price", help="price with the Python implementation only")
    add_common(p)
    p.set_defaults(func=cmd_price)

    p = sub.add_parser("compare", help="run python, R and C++ and tabulate against Black-Scholes")
    add_common(p)
    p.set_defaults(func=cmd_compare)

    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
