
# Static Order Router Backtester (Cont & Kukanov model)

`backtest.py` backtests a static order routing strategy based on the Cont & Kukanov model against several baseline execution strategies. It processes Level 1 market data snapshots to simulate order execution over time.

## Approach

1.  **Data Loading**:

* The script reads L1 market data from a CSV file (`l1_day.csv`).
* Expects the csv to contain columns for event timestamp (`ts_event`), publisher ID (`publisher_id`), ask price (`ask_px_00`), ask size (`ask_sz_00`), and side (`side`).
* Filters to only contain records from the ask side (`side == 'A'`).
* For each unique `ts_event`, only the first message per `publisher_id` is kept to construct a snapshot of the market. Snapshots are processed chronologically.


2.  **Static Router (Cont & Kukanov Model)** *(based on pseudocode & paper)*:

* The core of the backtest is the `allocate` function, it implements the static allocation strategy.
* Given a target order size and the current market snapshot (venues with ask prices, sizes, fees, and rebates), it searches for an optimal split of the order across venues.
* The optimality is determined by `compute_cost`, which considers:
	* Execution cost (price paid + fees - rebates).
	* Penalties for overfilling (`lambda_over`), underfilling (`lambda_under`), and queue risk/misexecution (`theta_queue`).
* A grid search is performed over the penalty parameters to find the set that results in the lowest total cash spent for the entire order.

  
3.  **Execution Simulation (`replay`)**:

* The `replay` function simulates executing a target total order size (e.g., 5000 shares) snapshot by snapshot.
* At each snapshot, the allocator determines the share quantities for each venue.
* Execution is simulated by "crossing the spread" (taking liquidity up to the displayed `ask_sz_00` at the `ask_px_00`.)
* Any unfilled portion of the order rolls over to the next snapshot.

4.  **Baseline Strategies**:

* The router's performance is compared against:
	*  `baseline_best_ask`: Sends all volume to the venue with the best (lowest) ask price in each snapshot.
	*  `baseline_twap`: Calculates a Time Weighted Average Price over 60 second buckets and assumes execution at this price.
	*  `baseline_vwap`: Calculates a Volume Weighted Average Price across all snapshots and assumes execution at this price.

5.  **Output**:

* The script outputs a JSON object containing the performance (total cash spent, average price) of the best router parameters found and all baseline strategies, along with savings in basis points.

## Parameter Ranges for Grid Search

A grid search is performed for finding the optimal router parameters:
*  **`lambda_over` (Overfill Penalty Coefficient)**: Values are `[1.e-05, 1.e-04, 1.e-03]` (from `np.logspace(-5, -3, 3)`).
*  **`lambda_under` (Underfill Penalty Coefficient)**: Values are `[1.e-05, 1.e-04, 1.e-03]` (from `np.logspace(-5, -3, 3)`).
*  **`theta_queue` (Queue Risk Penalty Coefficient)**: Values are `[0.0, 0.5, 1.0]` (from `np.linspace(0, 1.0, 3)`).

## Ideas for Improving Fill Realism

The current model assumes that any quantity allocated up to the displayed size (`ask_sz_00`) at a venue is instantly filled at the `ask_px_00`. To improve fill realism, one could incorporate a **probabilistic fill model based on queue position and market impact**:

*  **Queue Position**: Instead of assuming fills for passive orders (if the model were extended to include them) or immediate fills for aggressive orders, one could estimate a queue position. The probability of a fill, or the time to fill, could then be dependent on this estimated position relative to incoming trade flow and cancellations at that price level.

*  **Slippage/Price Impact for Aggressive Fills**: For shares that "cross the spread," especially larger allocations relative to the displayed size, the model could incorporate price slippage. This means that not all shares might be filled at `ask_px_00`. Subsequent shares might fill at progressively worse prices (`ask_px_01`, `ask_px_02`, ...), or a price impact model could estimate the average execution price based on the order's size relative to market depth and volatility. This would reflect that aggressively taking liquidity can move the price.

* **Accurate Venue Characterization**: Beyond fill dynamics, modeling the unique characteristics of each trading venue more accurately could further model realism. Consideration of items such as detailed fee structures, latency, order type support, and minimum lot sizes can be important to implement.
