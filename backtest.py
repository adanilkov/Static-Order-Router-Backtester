#!/usr/bin/env python3
"""
backtest.py: back-tests Cont & Kukanov static router against three baselines (VWAP, TWAP, and Best Ask).

Requires:
    Python 3.8+, numpy, pandas

Usage:
    python backtest.py (optional: path/to/l1_day.csv)
"""
 
import sys
import json
from dataclasses import dataclass
from math import inf

import numpy as np
import pandas as pd


@dataclass
class Venue:
    """Represents a trading venue with ask price, size, fee, and rebate.

    Args:
        ask (float): Ask price at the venue.
        ask_size (int): Available size at the ask price.
        fee (float): Trading fee for the venue.
        rebate (float): Rebate for providing liquidity (if applicable).
    """
    ask: float
    ask_size: int
    fee: float
    rebate: float


def load_snapshots(path: str):
    """Read L1 CSV feed into timestamped snapshots.

    Reads the raw L1 CSV at `path`, drops duplicate (ts_event, publisher_id)
    pairs keeping the first, sorts by timestamp, and groups rows by ts_event
    into a list of (timestamp, legs_dict) tuples where each legs_dict maps
    publisher_id to (ask_px_00, ask_sz_00).

    Args:
        path: Path to the L1 CSV file.

    Returns:
        List of tuples
        [(ts_event (pd.Timestamp),
          {publisher_id: (ask_price: float, ask_size: int), …}
        ), …].
    """
    df = pd.read_csv(
        path,
        usecols=['ts_event', 'publisher_id', 'ask_px_00', 'ask_sz_00', 'side'],
        parse_dates=['ts_event'],
        dtype={
            'publisher_id': np.int32,
            'ask_px_00': np.float64,
            'ask_sz_00': np.int32,
            'side': str
        }
    )
    df = df[df['side'] == 'A'].copy()  # filter for ask only
    df = df.drop_duplicates(subset=['ts_event', 'publisher_id'], keep='first')
    df = df.sort_values('ts_event')

    snapshots = []
    for ts, group in df.groupby('ts_event'):
        legs = {
            int(row.publisher_id): (float(row.ask_px_00), int(row.ask_sz_00))
            for _, row in group.iterrows()
        }
        snapshots.append((ts, legs))
    return snapshots


def compute_cost(split: list[int],
                 venues: list[Venue],
                 S: int,
                 lambda_o: float,
                 lambda_u: float,
                 theta: float) -> float:
    """Compute the cost of a given allocation split.

    Args:
        split: Allocated order sizes per venue.
        venues: List of Venue objects.
        S: Total target size to execute.
        lambda_o: Overfill penalty coefficient.
        lambda_u: Underfill penalty coefficient.
        theta: Queue penalty coefficient.

    Returns:
        float: Total cost including execution, fees, rebates, and penalties.
    """
    executed = 0
    cash = 0.0
    for qty, v in zip(split, venues):
        exe = min(qty, v.ask_size)
        executed += exe
        cash += exe * (v.ask + v.fee)
        cash -= max(qty - exe, 0) * v.rebate

    under = max(S - executed, 0)
    over = max(executed - S, 0)
    return cash + theta * (under + over) + lambda_u * under + lambda_o * over


def allocate(S: int,
             venues: list[Venue],
             lambda_o: float,
             lambda_u: float,
             theta: float,
             step: int = 100) -> tuple[list[int], float]:
    """Allocate an order of size S across multiple venues to minimize cost.

    Explores all splits in increments of `step` and picks the one with lowest cost.

    Args:
        S: Total order size to split.
        venues: List of Venue objects.
        lambda_o: Overfill penalty coefficient.
        lambda_u: Underfill penalty coefficient.
        theta: Queue penalty coefficient.
        step: Allocation granularity (default: 100).

    Returns:
        tuple[list[int], float]: A tuple containing:
            - best_split (list[int]): List of allocations per venue.
            - best_cost (float): Cost associated with the best_split.
    """
    N = len(venues)

    splits = [[]]
    for i in range(N):
        new_splits = []
        for alloc in splits:
            used = sum(alloc)
            cap = min(S - used, venues[i].ask_size)
            for q in range(0, cap + 1, step):
                new_splits.append(alloc + [q])
        splits = new_splits

    best_cost = inf
    best_split: list[int] = []
    for alloc in splits:
        if sum(alloc) != S: continue

        cost = compute_cost(alloc, venues, S, lambda_o, lambda_u, theta)
        if cost < best_cost:
            best_cost = cost
            best_split = alloc

    return best_split, best_cost


def replay(snapshots: list[tuple[pd.Timestamp, dict[int, tuple[float, int]]]],
           params: tuple[float, float, float],
           S: int = 5000) -> tuple[float, float]:
    """Replay order execution over snapshots using the static router.

    Args:
        snapshots: List of (timestamp, legs) as from load_snapshots.
        params: (lambda_o, lambda_u, theta) penalty parameters.
        S: Initial total order size.

    Returns:
        tuple[float, float]: A tuple containing:
            - total_cash (float): Sum of cash spent.
            - avg_price (float): Average executed price.
    """
    lambda_o, lambda_u, theta = params
    remaining = S
    total_cash = 0.0

    for ts, legs in snapshots:
        if remaining <= 0: break

        venues = [
            Venue(ask=px, ask_size=sz, fee=0.003, rebate=0.002)
            for px, sz in legs.values()
        ]
        split, _ = allocate(remaining, venues, lambda_o, lambda_u, theta)
        for qty, v in zip(split, venues):
            exe = min(qty, v.ask_size)
            total_cash += exe * (v.ask + v.fee)

        remaining -= sum(min(q, v.ask_size) for q, v in zip(split, venues))

    filled = S - max(remaining, 0)
    avg_price = total_cash / filled if filled > 0 else float('nan')
    return total_cash, avg_price


def baseline_best_ask(snapshots: list[tuple[pd.Timestamp, dict[int, tuple[float, int]]]],
                      S: int = 5000) -> tuple[float, float]:
    """Baseline: route all volume to the venue with the best ask each snapshot.

    Args:
        snapshots (list[tuple[pd.Timestamp, dict[int, tuple[float, int]]]]):
            List of (timestamp, legs) as from load_snapshots.
        S (int): Initial total order size. Defaults to 5000.

    Returns:
        tuple[float, float]: A tuple containing:
            - cash (float): Total cash spent.
            - avg_price (float): Average executed price (NaN if no volume filled).
    """
    remaining = S
    cash = 0.0
    for _, legs in snapshots:
        if remaining <= 0: break

        best_px, best_sz = min(legs.values(), key=lambda x: x[0])
        exe = min(best_sz, remaining)
        cash += exe * best_px
        remaining -= exe

    filled = S - max(remaining, 0)
    return cash, (cash / filled if filled > 0 else float('nan'))


def baseline_twap(snapshots: list[tuple[pd.Timestamp, dict[int, tuple[float, int]]]],
                  S: int = 5000,
                  bucket: int = 60) -> tuple[float, float]:
    """Baseline: TWAP using fixed time buckets.

    Calculates Time-Weighted Average Price (TWAP) by bucketing snapshots
    into fixed time intervals and averaging the best prices within each bucket.

    Args:
        snapshots (list[tuple[pd.Timestamp, dict[int, tuple[float, int]]]]):
            List of (timestamp, legs) as from load_snapshots.
        S (int): Total order size to simulate. Defaults to 5000.
        bucket (int): Time bucket size in seconds. Defaults to 60.

    Returns:
        tuple[float, float]: A tuple containing:
            - cash (float): Total cash spent (S * TWAP).
            - avg_px (float): Calculated TWAP (NaN if no prices available).
    """
    df = pd.DataFrame(snapshots, columns=['ts', 'legs'])
    
    # ts is datetime and bucket by floor
    df['ts'] = pd.to_datetime(df['ts'])
    df['bucket'] = df['ts'].dt.floor(f'{bucket}S')

    avg_prices: list[float] = []
    for _, grp in df.groupby('bucket'):
        prices = [v[0] for legs in grp['legs'] for v in legs.values()]
        if prices:
            avg_prices.append(min(prices))
        else:
            avg_prices.append(float('nan'))

    avg_px = float(np.nanmean(avg_prices)) if avg_prices else float('nan')
    cash = avg_px * S
    return cash, avg_px


def baseline_vwap(snapshots: list[tuple[pd.Timestamp, dict[int, tuple[float, int]]]],
                  S: int = 5000) -> tuple[float, float]:
    """Baseline: VWAP across all snapshots.

    Calculates Volume-Weighted Average Price (VWAP) across all provided snapshots.

    Args:
        snapshots (list[tuple[pd.Timestamp, dict[int, tuple[float, int]]]]):
            List of (timestamp, legs) as from load_snapshots.
        S (int): Total order size to simulate. Defaults to 5000.

    Returns:
        tuple[float, float]: A tuple containing:
            - cash (float): Total cash spent (S * VWAP).
            - avg_px (float): Calculated VWAP (NaN if no volume traded).
    """
    prices: list[float] = []
    sizes: list[int] = []
    for _, legs in snapshots:
        for px, sz in legs.values():
            prices.append(px)
            sizes.append(sz)
    avg_px = float(np.average(prices, weights=sizes)) if sizes else float('nan')
    cash = avg_px * S
    return cash, avg_px


def main(path: str):
    """Main entry point: grid-search router params, compute baselines, and output JSON.

    Loads snapshot data, performs a grid search for best router parameters,
    calculates performance for baseline strategies (Best Ask, TWAP, VWAP),
    and prints the results as a JSON object to standard output.

    Args:
        path (str): Path to the L1 CSV data file.
    """
    snaps = load_snapshots(path)

    # grid search for best router params
    lambdas = np.logspace(-5, -3, 3)
    thetas = np.linspace(0, 1.0, 3)
    best: dict[str, float] = {'cost': inf}

    for lambda_o in lambdas:
        for lambda_u in lambdas:
            for theta in thetas:
                cash, avg_px = replay(snaps, (lambda_o, lambda_u, theta))
                if cash < best['cost']:
                    best.update({
                        'lambda_over': lambda_o,
                        'lambda_under': lambda_u,
                        'theta_queue': theta,
                        'cash': cash,
                        'avg_px': avg_px,
                        'cost': cash
                    })

    ba_cash, ba_px = baseline_best_ask(snaps)
    tw_cash, tw_px = baseline_twap(snaps)
    vw_cash, vw_px = baseline_vwap(snaps)

    out = {
        'router':   best,
        'best_ask': {'cash': ba_cash, 'avg_px': ba_px},
        'TWAP60s':  {'cash': tw_cash, 'avg_px': tw_px},
        'VWAP':     {'cash': vw_cash, 'avg_px': vw_px},
        'savings_bps': {
            'vs_best_ask': (ba_px - best['avg_px']) / ba_px * 1e4,
            'vs_TWAP60s':  (tw_px - best['avg_px']) / tw_px * 1e4,
            'vs_VWAP':     (vw_px - best['avg_px']) / vw_px * 1e4,
        }
    }

    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    path = sys.argv[1] if len(sys.argv) > 1 else 'l1_day.csv'
    main(path)
