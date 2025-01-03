from subprocess import run

from dexscan.constants import PROJECT_DIR
from dexscan.database import start_tx
from dexscan.primitives import Hex


def codegen():
    run(
        (
            "edgedb-py",
            "--file",
            "dexscan/edge_codegen.py",
            "--target",
            "blocking",
            "--no-skip-pydantic-validation",
        )
    )


def fmt():
    run(
        ("poetry", "run", "ruff", "check", "--select", "I", "--fix", "."),
        cwd=PROJECT_DIR,
    )
    run(
        ("poetry", "run", "ruff", "format", "."),
        cwd=PROJECT_DIR,
    )


def reset_pair():
    pair_addr_str = input("enter pair addr: ")
    pair_addr = Hex(pair_addr_str)
    for tx in start_tx():
        with tx:
            tx.reset_token_pair(pair_addr)
    print("done")
