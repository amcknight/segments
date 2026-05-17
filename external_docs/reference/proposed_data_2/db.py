"""Sqlite-backed event log for segment fits (proposed_data_2).

Counter-proposal to reference/proposed_data. See schema.sql for the
delta. Headline change: rows now carry a created_at timestamp.

One table, `attempts`. Each row is one ending event (died or survived)
of a single cold-start attempt-instance. Insertion order (rowid) defines
attempt order within (game, segment). attempt_n is derived at export
time, not stored, so late inserts or out-of-order corrections renumber
automatically.

CLI:
    python data/db.py init
    python data/db.py export <game> <segment> [--out path.tsv]
    python data/db.py export <game> <segment> --full [--out path.tsv]

`export` writes the v1-compatible TSV (attempt_n, outcome, time_ms) so
existing fit code keeps working unchanged. `export --full` adds
created_at as a 4th column.
"""
import argparse
import os
import sqlite3
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(HERE, 'segments.sqlite')
SCHEMA_PATH = os.path.join(HERE, 'schema.sql')


def connect(db_path=DB_PATH):
    conn = sqlite3.connect(db_path)
    conn.execute('PRAGMA foreign_keys = ON')
    return conn


def init(db_path=DB_PATH):
    with open(SCHEMA_PATH) as f:
        ddl = f.read()
    conn = connect(db_path)
    try:
        conn.executescript(ddl)
        conn.commit()
    finally:
        conn.close()


def export(game, segment, out=None, full=False, db_path=DB_PATH):
    """Write a fit-shaped TSV.

    Default columns: `attempt_n  outcome  time_ms` (v1-compatible).
    With full=True: adds `created_at` (unix seconds) as a 4th column.

    out=None writes to stdout. Rows are ordered by insertion (id), and
    attempt_n is 1-indexed to match learning_model's n convention.
    """
    conn = connect(db_path)
    try:
        if full:
            rows = conn.execute(
                'SELECT outcome, time_ms, created_at FROM attempts '
                'WHERE game = ? AND segment = ? ORDER BY id',
                (game, segment),
            ).fetchall()
        else:
            rows = conn.execute(
                'SELECT outcome, time_ms FROM attempts '
                'WHERE game = ? AND segment = ? ORDER BY id',
                (game, segment),
            ).fetchall()
    finally:
        conn.close()

    sink = open(out, 'w') if out else sys.stdout
    try:
        if full:
            sink.write('attempt_n\toutcome\ttime_ms\tcreated_at\n')
            for n, (outcome, time_ms, created_at) in enumerate(rows, start=1):
                sink.write(f'{n}\t{outcome}\t{time_ms}\t{created_at}\n')
        else:
            sink.write('attempt_n\toutcome\ttime_ms\n')
            for n, (outcome, time_ms) in enumerate(rows, start=1):
                sink.write(f'{n}\t{outcome}\t{time_ms}\n')
    finally:
        if out:
            sink.close()
    return len(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest='cmd', required=True)

    sub.add_parser('init', help='create segments.sqlite from schema.sql')

    pe = sub.add_parser('export', help='dump one segment as fit-shaped TSV')
    pe.add_argument('game')
    pe.add_argument('segment')
    pe.add_argument('--out', help='output path (default: stdout)')
    pe.add_argument(
        '--full', action='store_true',
        help='include created_at column (v1 export omits it for compat)',
    )

    args = p.parse_args()

    if args.cmd == 'init':
        init()
        print(f'initialized {DB_PATH}')
    elif args.cmd == 'export':
        n = export(args.game, args.segment, args.out, args.full)
        msg = f'exported {n} attempts'
        if args.out:
            msg += f' -> {args.out}'
        print(msg, file=sys.stderr)


if __name__ == '__main__':
    main()
