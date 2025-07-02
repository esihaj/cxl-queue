#!/usr/bin/env python3
"""
verify_doorbell_compilation.py
Checks that doorbell_benchmark.s contains the expected number of cache /
non-temporal instructions.  Exits 0 on success, 1 on mismatch.
"""

import argparse
import pathlib
import re
import subprocess
import sys
from collections import OrderedDict, Counter

# ---------------------------------------------------------------------------
EXPECTED = OrderedDict([
    ("clflush",     1),
    ("clflushopt",  3),
    ("clwb",        2),
    ("movdir64b",   3),
    ("vmovdqa64",  14),
    ("vmovntdq",    6),
])

# ---------------------------------------------------------------------------

def build_asm() -> pathlib.Path:
    print("▶ make doorbell_asm", flush=True)
    subprocess.run(["make", "-j4", "--always-make", "doorbell_asm"],
                   check=True)
    asm = pathlib.Path("doorbell_benchmark.s")
    if not asm.exists():
        raise FileNotFoundError("doorbell_benchmark.s not found")
    return asm

# helper: is this line worth inspecting?
def is_code_line(line: str) -> bool:
    stripped = line.lstrip()
    return (stripped and
            not stripped.startswith(('.', '#')) and  # directives / comments
            '"' not in stripped)                     # strings / data

def count_instrs(asm_file: pathlib.Path) -> Counter:
    counts = Counter()
    with asm_file.open() as f:
        for raw in f:
            # drop trailing comment
            line = raw.split('#', 1)[0]
            if not is_code_line(line):
                continue

            # remove label if present
            if ':' in line:
                line = line.split(':', 1)[1]

            tokens = line.split()
            if not tokens:
                continue
            mnem = tokens[0]          # first token after optional label
            if mnem in EXPECTED:
                counts[mnem] += 1
    return counts

def pretty(ok: bool) -> str:
    return "✔" if ok else "✖"

# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--asm", help="pre-built .s file (skip make)")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    asm_path = pathlib.Path(args.asm) if args.asm else build_asm()
    found = count_instrs(asm_path)

    ok = True
    print(f"\nassembly size: {asm_path.stat().st_size:,} bytes\n")
    print(f"{'':2} {'instruction':12} {'expected':>9} {'found':>7}")
    print("─" * 34)
    for mnem, exp in EXPECTED.items():
        got = found.get(mnem, 0)
        cell_ok = (got == exp)
        ok &= cell_ok
        print(f"{pretty(cell_ok)} {mnem:12} {exp:9} {got:7}")

    if args.verbose:
        extra = sorted(set(found) - set(EXPECTED))
        if extra:
            print("\nother mnemonics counted:", ", ".join(extra))

    sys.exit(0 if ok else 1)

# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()
