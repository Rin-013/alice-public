#!/usr/bin/env python3
"""Seed Alice's cartridge (cartridge_alice.yaml) into the live memory DB.

Self-memories live in the regular `memories` table with
associated_user='alice'; the identity wire (get_self_memories ->
{identity_narrative}) surfaces the top-N by importance each turn.

Idempotent: each seed keeps a stable id (cartridge_<id>), so re-running
updates content/importance in place — edit the YAML, re-seed, done.
Access counts and timestamps of existing rows are preserved on update.

    python alice/core/memory/seed_cartridge.py            # dry run (default)
    python alice/core/memory/seed_cartridge.py --apply    # write to live DB
"""

import argparse
import os
import sqlite3
import sys
import time

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
CARTRIDGE = os.path.join(HERE, "cartridge_alice.yaml")
DEFAULT_DB = os.path.join(HERE, "..", "..", "data", "databases", "alice_memory.db")


def main():
    parser = argparse.ArgumentParser(description="Seed Alice's identity cartridge")
    parser.add_argument("--db", default=DEFAULT_DB, help="memory DB path")
    parser.add_argument("--apply", action="store_true",
                        help="actually write (default is dry-run)")
    args = parser.parse_args()

    with open(CARTRIDGE, encoding="utf-8") as f:
        seeds = yaml.safe_load(f)["seeds"]

    db_path = os.path.normpath(args.db)
    if not os.path.exists(db_path):
        sys.exit(f"DB not found: {db_path} (run chat.py once to create it, or pass --db)")

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    inserted, updated, unchanged = 0, 0, 0
    for seed in seeds:
        mid = f"cartridge_{seed['id']}"
        content = " ".join(seed["content"].split())  # collapse YAML folding
        importance = float(seed["importance"])
        mtype = seed.get("type", "fact")

        row = cur.execute(
            "SELECT content, importance, memory_type FROM memories WHERE id = ?", (mid,)
        ).fetchone()

        if row is None:
            action = "INSERT"
            inserted += 1
            if args.apply:
                cur.execute(
                    "INSERT INTO memories (id, timestamp, memory_type, content,"
                    " importance, associated_user) VALUES (?, ?, ?, ?, ?, 'alice')",
                    (mid, time.time(), mtype, content, importance),
                )
        elif row != (content, importance, mtype):
            action = "UPDATE"
            updated += 1
            if args.apply:
                cur.execute(
                    "UPDATE memories SET content = ?, importance = ?, memory_type = ?"
                    " WHERE id = ?",
                    (content, importance, mtype, mid),
                )
        else:
            action = "ok"
            unchanged += 1

        print(f"  [{action:>6}] {seed['id']:<24} imp={importance:.2f}  {content[:55]}...")

    if args.apply:
        conn.commit()
        print(f"\nApplied: {inserted} inserted, {updated} updated, {unchanged} unchanged -> {db_path}")
    else:
        print(f"\nDry run: {inserted} would insert, {updated} would update, "
              f"{unchanged} unchanged. Re-run with --apply to write.")
    conn.close()


if __name__ == "__main__":
    main()
