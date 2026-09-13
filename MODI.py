import numpy as np


def print_vam_step(step, c, s, d, allocation, active_rows, active_cols, row_pen, col_pen, selected_r, selected_c, qty):
    num_rows, num_cols = c.shape
    print(f"\n{'='*75}")
    print(f"VAM Iteration {step}")
    print(f"{'='*75}")

    header = f"{'Origin':<10}" + "".join(f"{'D' + str(j+1):>11}" for j in range(num_cols)) + f"{'Supply':>10}" + f"{'Penalty':>10}"
    print(header)
    print("-" * len(header))

    for i in range(num_rows):
        row_label = f"S{i+1}"
        row_str = f"{row_label:<10}"
        for j in range(num_cols):
            if allocation[i, j] > 0:
                cell = f"[{allocation[i, j]:.0f}]({c[i, j]:.0f})"
            elif i in active_rows and j in active_cols:
                cell = f"({c[i, j]:.0f})"
            else:
                cell = "-"
            row_str += f"{cell:>11}"
        
        supply_val = f"{s[i]:.0f}" if i in active_rows else "-"
        pen_val = f"{row_pen[i]:.0f}" if i in row_pen else "-"
        row_str += f"{supply_val:>10}{pen_val:>10}"
        print(row_str)

    print("-" * len(header))
    demand_str = f"{'Demand':<10}" + "".join(f"{(f'{d[j]:.0f}' if j in active_cols else '-'):>11}" for j in range(num_cols))
    print(demand_str)
    pen_str = f"{'Penalty':<10}" + "".join(f"{(f'{col_pen[j]:.0f}' if j in col_pen else '-'):>11}" for j in range(num_cols))
    print(pen_str)
    print(f"\nAction: Allocated {qty:.0f} units to Cell (S{selected_r+1}, D{selected_c+1}) with cost {c[selected_r, selected_c]:.0f}")


def print_modi_table(iteration, c, allocation, basis, u, v, deltas):
    num_rows, num_cols = c.shape
    print(f"\n{'='*80}")
    print(f"MODI Iteration {iteration}")
    print(f"{'='*80}")

    v_header = f"{'':<10}" + "".join(f"{'v' + str(j+1) + '=' + str(int(v[j])):>14}" for j in range(num_cols))
    print(v_header)
    header = f"{'Origin':<10}" + "".join(f"{'D' + str(j+1):>14}" for j in range(num_cols)) + f"{'u_i':>10}"
    print(header)
    print("-" * len(header))

    for i in range(num_rows):
        row_str = f"S{i+1:<9}"
        for j in range(num_cols):
            if (i, j) in basis:
                cell = f"[{allocation[i, j]:.0f}] c={c[i, j]:.0f}"
            else:
                cell = f"Δ={deltas[i, j]:.0f} (c={c[i, j]:.0f})"
            row_str += f"{cell:>14}"
        row_str += f"{u[i]:>10.0f}"
        print(row_str)
    print("-" * len(header))


def vogel_approximation_with_steps(c, s, d):
    num_rows, num_cols = c.shape
    s_curr = s.copy()
    d_curr = d.copy()
    allocation = np.zeros((num_rows, num_cols), dtype=float)
    basis = []

    active_rows = set(range(num_rows))
    active_cols = set(range(num_cols))
    step = 1
    print("PHASE 1: VOGEL'S APPROXIMATION METHOD (VAM) - STEP BY STEP")

    while active_rows and active_cols:
        row_penalties = {}
        for r in active_rows:
            row_costs = sorted([c[r, col] for col in active_cols])
            row_penalties[r] = row_costs[1] - row_costs[0] if len(row_costs) >= 2 else row_costs[0]

        col_penalties = {}
        for col in active_cols:
            col_costs = sorted([c[r, col] for r in active_rows])
            col_penalties[col] = col_costs[1] - col_costs[0] if len(col_costs) >= 2 else col_costs[0]

        max_row_pen = max(row_penalties.values()) if row_penalties else -1
        max_col_pen = max(col_penalties.values()) if col_penalties else -1

        if max_row_pen >= max_col_pen:
            r = max(row_penalties, key=lambda k: row_penalties[k])
            col = min(active_cols, key=lambda k: c[r, k])
        else:
            col = max(col_penalties, key=lambda k: col_penalties[k])
            r = min(active_rows, key=lambda k: c[k, col])

        quantity = min(s_curr[r], d_curr[col])
        allocation[r, col] = quantity
        basis.append((r, col))
        s_curr[r] -= quantity
        d_curr[col] -= quantity

        print_vam_step(step, c, s_curr, d_curr, allocation, active_rows, active_cols,
                       row_penalties, col_penalties, r, col, quantity)
        step += 1

        if s_curr[r] == 0 and d_curr[col] == 0:
            if len(active_rows) > 1:
                active_rows.remove(r)
            elif len(active_cols) > 1:
                active_cols.remove(col)
            else:
                active_rows.remove(r)
                active_cols.remove(col)
        elif s_curr[r] == 0:
            active_rows.remove(r)
        else:
            active_cols.remove(col)

    while len(basis) < num_rows + num_cols - 1:
        for r in range(num_rows):
            for col in range(num_cols):
                if (r, col) not in basis:
                    basis.append((r, col))
                    break
            if len(basis) == num_rows + num_cols - 1:
                break

    return allocation, basis


def compute_potentials(costs, basis, num_rows, num_cols):
    u = [None] * num_rows
    v = [None] * num_cols
    u[0] = 0.0

    while any(x is None for x in u) or any(x is None for x in v):
        progress = False
        for r, col in basis:
            if u[r] is not None and v[col] is None:
                v[col] = costs[r, col] - u[r]
                progress = True
            elif v[col] is not None and u[r] is None:
                u[r] = costs[r, col] - v[col]
                progress = True
        if not progress:
            for i in range(num_rows):
                if u[i] is None:
                    u[i] = 0.0
                    break

    return np.array(u, dtype=float), np.array(v, dtype=float)


def find_loop(start, basis):
    loop_cells = list(set(basis) | {start})

    def dfs(path, looking_for_row):
        curr = path[-1]
        if len(path) >= 4 and len(path) % 2 == 0:
            if looking_for_row and curr[0] == start[0] and curr[1] != start[1]:
                return path
            if not looking_for_row and curr[1] == start[1] and curr[0] != start[0]:
                return path

        for nxt in loop_cells:
            if nxt in path:
                continue
            if looking_for_row:
                if nxt[0] == curr[0]:
                    res = dfs(path + [nxt], False)
                    if res:
                        return res
            else:
                if nxt[1] == curr[1]:
                    res = dfs(path + [nxt], True)
                    if res:
                        return res
        return None

    res = dfs([start], True)
    if not res:
        res = dfs([start], False)
    return res


def solve_transportation_detailed(costs, supply, demand):
    c = np.array(costs, dtype=float)
    s = np.array(supply, dtype=float)
    d = np.array(demand, dtype=float)

    total_supply = np.sum(s)
    total_demand = np.sum(d)

    if total_supply > total_demand:
        c = np.hstack([c, np.zeros((c.shape[0], 1))])
        d = np.append(d, total_supply - total_demand)
    elif total_demand > total_supply:
        c = np.vstack([c, np.zeros((1, c.shape[1]))])
        s = np.append(s, total_demand - total_supply)

    num_rows, num_cols = c.shape
    allocation, basis = vogel_approximation_with_steps(c, s, d)
    initial_cost = np.sum(allocation * c)

    print(f"\nInitial VAM Transportation Cost: {initial_cost:.2f}\n")
    print("PHASE 2: MODI METHOD OPTIMALITY ITERATIONS")

    iteration = 1
    while True:
        u, v = compute_potentials(c, basis, num_rows, num_cols)
        deltas = np.zeros((num_rows, num_cols), dtype=float)

        min_reduced_cost = 0.0
        entering_cell = None

        for r in range(num_rows):
            for col in range(num_cols):
                if (r, col) not in basis:
                    deltas[r, col] = c[r, col] - (u[r] + v[col])
                    if deltas[r, col] < min_reduced_cost:
                        min_reduced_cost = deltas[r, col]
                        entering_cell = (r, col)
                else:
                    deltas[r, col] = 0.0

        print_modi_table(iteration, c, allocation, basis, u, v, deltas)

        if entering_cell is None or min_reduced_cost >= -1e-7:
            print("Optimal Condition Satisfied: All opportunity costs (Δ_ij) >= 0.")
            break

        er, ec = entering_cell
        print(f"Entering Cell: (S{er+1}, D{ec+1}) with most negative Δ = {min_reduced_cost:.2f}")

        loop = find_loop(entering_cell, basis)
        loop_signs = ["(+)" if i % 2 == 0 else "(-)" for i in range(len(loop))]
        loop_str = " -> ".join(f"(S{r+1}, D{c+1}) {sign}" for (r, c), sign in zip(loop, loop_signs))
        print(f"Closed Stepping-Stone Loop: {loop_str}")

        minus_cells = [loop[i] for i in range(1, len(loop), 2)]
        theta = min(allocation[r, col] for r, col in minus_cells)
        print(f"Allocating θ = {theta:.2f} along the loop.")

        for i, (r, col) in enumerate(loop):
            if i % 2 == 0:
                allocation[r, col] += theta
            else:
                allocation[r, col] -= theta

        basis.append(entering_cell)
        for cell in minus_cells:
            if abs(allocation[cell[0], cell[1]]) < 1e-9:
                allocation[cell[0], cell[1]] = 0.0
                basis.remove(cell)
                break

        current_cost = np.sum(allocation * c)
        print(f"Transportation Cost after Iteration {iteration}: {current_cost:.2f}")
        iteration += 1

    optimal_cost = np.sum(allocation * c)
    return allocation, initial_cost, optimal_cost


if __name__ == "__main__":
    costs = [
        [3, 1, 7, 4],
        [2, 6, 5, 9],
        [8, 3, 3, 2]
    ]
    supply = [250, 350, 400]
    demand = [200, 300, 350, 150]

    opt_alloc, init_cost, opt_cost = solve_transportation_detailed(costs, supply, demand)
    print("\n" + "=" * 50)
    print("FINAL SUMMARY")
    print("=" * 50)
    print(f"Initial VAM Cost: {init_cost:.2f}")
    print(f"Optimal MODI Cost: {opt_cost:.2f}")
    print("\nFinal Optimal Allocation Matrix:")
    print(opt_alloc)