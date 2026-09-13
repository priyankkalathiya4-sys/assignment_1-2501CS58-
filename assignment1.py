import numpy as np

def format_cell(val):
    if abs(val) < 1e-6:
        return "0.00"
    if abs(val) >= 1e5:
        return f"{val:.1e}"
    return f"{val:.2f}"

def print_tableau(tableau, basis, col_names, iteration, entering=None, leaving=None):
    print(f"\n{'='*70}")
    print(f"Tableau Iteration {iteration}")
    if entering is not None and leaving is not None:
        print(f"Pivot -> Entering Variable: {entering} | Leaving Variable: {leaving}")
    print(f"{'='*70}")

    header = f"{'Basis':<8}" + "".join(f"{name:>10}" for name in col_names)
    print(header)
    print("-" * len(header))

    num_rows = len(basis)
    for i in range(num_rows):
        b_var = col_names[basis[i]]
        row_str = f"{b_var:<8}" + "".join(f"{format_cell(tableau[i, j]):>10}" for j in range(tableau.shape[1]))
        print(row_str)

    print("-" * len(header))
    zc_str = f"{'z-c':<8}" + "".join(f"{format_cell(tableau[-1, j]):>10}" for j in range(tableau.shape[1]))
    print(zc_str)
    print(f"{'='*70}")

def big_m_simplex(c, A, b, constraint_types, maximize=True, M=1e6):
    c = np.array(c, dtype=float)
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)

    if not maximize:
        c = -c

    num_constraints, num_vars = A.shape

    for i in range(num_constraints):
        if b[i] < 0:
            b[i] = -b[i]
            A[i] = -A[i]
            if constraint_types[i] == '<=':
                constraint_types[i] = '>='
            elif constraint_types[i] == '>=':
                constraint_types[i] = '<='

    col_names = [f"x{j+1}" for j in range(num_vars)]
    col_types = ['orig'] * num_vars
    obj_coeffs = list(c)
    extra_cols = []
    basis = []

    s_idx, e_idx, a_idx = 1, 1, 1

    for i, c_type in enumerate(constraint_types):
        if c_type == '<=':
            col = np.zeros(num_constraints)
            col[i] = 1.0
            extra_cols.append(col)
            obj_coeffs.append(0.0)
            col_types.append('slack')
            col_names.append(f"s{s_idx}")
            basis.append(num_vars + len(extra_cols) - 1)
            s_idx += 1
        elif c_type == '>=':
            col_surplus = np.zeros(num_constraints)
            col_surplus[i] = -1.0
            extra_cols.append(col_surplus)
            obj_coeffs.append(0.0)
            col_types.append('surplus')
            col_names.append(f"e{e_idx}")
            e_idx += 1

            col_art = np.zeros(num_constraints)
            col_art[i] = 1.0
            extra_cols.append(col_art)
            obj_coeffs.append(-M)
            col_types.append('artificial')
            col_names.append(f"A{a_idx}")
            basis.append(num_vars + len(extra_cols) - 1)
            a_idx += 1
        elif c_type == '=':
            col_art = np.zeros(num_constraints)
            col_art[i] = 1.0
            extra_cols.append(col_art)
            obj_coeffs.append(-M)
            col_types.append('artificial')
            col_names.append(f"A{a_idx}")
            basis.append(num_vars + len(extra_cols) - 1)
            a_idx += 1

    col_names.append("RHS")

    if extra_cols:
        full_A = np.hstack([A, np.column_stack(extra_cols)])
    else:
        full_A = A.copy()

    total_vars = full_A.shape[1]
    obj_coeffs = np.array(obj_coeffs, dtype=float)

    tableau = np.zeros((num_constraints + 1, total_vars + 1))
    tableau[:num_constraints, :total_vars] = full_A
    tableau[:num_constraints, -1] = b

    c_b = obj_coeffs[basis]
    tableau[-1, :total_vars] = c_b @ full_A - obj_coeffs
    tableau[-1, -1] = c_b @ b

    iteration = 0
    print_tableau(tableau, basis, col_names, iteration)

    while True:
        rc = tableau[-1, :-1]
        min_val = np.min(rc)
        if min_val >= -1e-6:
            break

        pivot_col = np.argmin(rc)
        col_vals = tableau[:num_constraints, pivot_col]
        rhs_vals = tableau[:num_constraints, -1]

        ratios = []
        valid_rows = []
        for i in range(num_constraints):
            if col_vals[i] > 1e-7:
                ratios.append(rhs_vals[i] / col_vals[i])
                valid_rows.append(i)

        if not valid_rows:
            return "Unbounded", None, None

        pivot_row = valid_rows[np.argmin(ratios)]
        pivot_val = tableau[pivot_row, pivot_col]

        entering_var = col_names[pivot_col]
        leaving_var = col_names[basis[pivot_row]]

        tableau[pivot_row, :] /= pivot_val
        for r in range(num_constraints + 1):
            if r != pivot_row:
                tableau[r, :] -= tableau[r, pivot_col] * tableau[pivot_row, :]

        basis[pivot_row] = pivot_col
        iteration += 1
        print_tableau(tableau, basis, col_names, iteration, entering_var, leaving_var)

    solution = np.zeros(total_vars)
    for i, b_idx in enumerate(basis):
        solution[b_idx] = tableau[i, -1]

    for i, b_idx in enumerate(basis):
        if col_types[b_idx] == 'artificial' and solution[b_idx] > 1e-4:
            return "Infeasible", None, None

    x = solution[:num_vars]
    optimal_value = tableau[-1, -1] if maximize else -tableau[-1, -1]

    return "Optimal", x, optimal_value


if __name__ == "__main__":
    c = [4, 2]
    A = [
        [1, 1],
        [3, 1],
        [1, 2]
    ]
    b = [10, 14, 16]
    constraint_types = ['=', '>=', '<=']

    status, solution, opt_val = big_m_simplex(
        c, A, b, constraint_types, maximize=False
    )

    print("\nStatus:", status)
    print("Optimal x:", solution)
    print("Optimal Value by BIGM Method:", opt_val)