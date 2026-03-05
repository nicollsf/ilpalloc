#!/usr/bin/env python3
"""Convert Fred Nicolls' MATLAB allocation script to Python.

Usage:
    python ilp_alloc.py [path_to_excel]

The script reads a student topic choices spreadsheet and solves an
integer linear program to assign students to projects, respecting
supervisor load limits and student preferences.  Outputs are written
to Excel files.
"""

import sys
import re
import random

import numpy as np
import pandas as pd
from pulp import (
    LpProblem,
    LpMinimize,
    LpVariable,
    lpSum,
    LpBinary,
    PULP_CBC_CMD,
    value,
)


def allocate_projects(fname: str) -> None:
    """Allocate student projects based on choices in XLSX file.

    Assumes the file has sheets: 'choices', 'limits', 'scores'
    - 'choices': Student No, Topic ID Choice 1, ..., Topic ID Choice N
    - 'limits': Supervisor, Max (with '*' for default)
    - 'scores': Choice, Score (weights for preferences)
    """
    xls = pd.ExcelFile(fname)

    # Read choices sheet
    df_choices = pd.read_excel(xls, sheet_name='choices')
    if df_choices.empty:
        raise ValueError(f"No data in 'choices' sheet of {fname}")

    # Read limits sheet
    df_limits = pd.read_excel(xls, sheet_name='limits')
    limits_dict = {}
    default_limit = 6
    for _, row in df_limits.iterrows():
        sup = str(row['Supervisor']).strip()
        max_val = int(row['Max'])
        if sup == '*':
            default_limit = max_val
        else:
            limits_dict[sup] = max_val

    # Read scores sheet
    df_scores = pd.read_excel(xls, sheet_name='scores')
    scores = {}
    for _, row in df_scores.iterrows():
        choice = int(row['Choice'])
        score = float(row['Score'])
        scores[choice] = score

    # Extract data from choices
    student_col = df_choices.columns[0]
    choice_cols = [col for col in df_choices.columns[1:] if 'Choice' in col]
    if not choice_cols:
        raise ValueError("No choice columns found in 'choices' sheet")

    # Determine number of choices based on populated columns (consecutive from start)
    num_choices = 0
    for col in choice_cols:
        if df_choices[col].notna().any():
            num_choices += 1
        else:
            break

    Ms = df_choices.values
    Ms = np.array([[str(x).strip() if pd.notna(x) else '' for x in row] for row in Ms], dtype=object)

    #------------------------------------------------------------------
    # Supervisor codes, self-proposed detection, and project indices
    #------------------------------------------------------------------

    # Flatten all project codes from choice columns
    all_codes = []
    for row in Ms:
        for code in row[1:]:  # skip student ID
            if code:
                all_codes.append(code)
    pcodes = list(dict.fromkeys(all_codes))  # unique, preserve order

    # supervisor code is leading uppercase letters of the project code
    scodes = sorted({re.match(r"[A-Z]+", p).group(0) for p in pcodes if re.match(r"[A-Z]+", p)})

    # per-supervisor limits from limits sheet
    slimsn = np.array([default_limit] * len(scodes), dtype=int)
    for i, sup in enumerate(scodes):
        if sup in limits_dict:
            slimsn[i] = limits_dict[sup]

    # detect self-proposed projects: codes containing lowercase
    spi = [bool(re.search(r"[a-z]", p)) for p in pcodes]
    spcodes = [p for p, flag in zip(pcodes, spi) if flag]

    # boolean index of rows in Ms that are self-proposed (first choice is self-proposed)
    spsi = np.zeros(Ms.shape[0], dtype=bool)
    for i, row in enumerate(Ms):
        if row[1] in spcodes:  # first choice
            spsi[i] = True

    # tchoices: for each student, store the index (into pcodes) of each preference column
    m = Ms.shape[0]
    n = len(pcodes)
    tchoices = np.full((m, num_choices), -1, dtype=int)
    for i in range(m):
        for j in range(num_choices):
            code = Ms[i, j + 1]
            if code in pcodes:
                tchoices[i, j] = pcodes.index(code)

    # project -> supervisor index mapping
    psupi = np.full(n, -1, dtype=int)
    for i, p in enumerate(pcodes):
        sup = re.match(r"^[A-Z]+", p)
        if sup:
            sup_code = sup.group(0)
            if sup_code in scodes:
                psupi[i] = scodes.index(sup_code)

    #------------------------------------------------------------------
    # Build and solve integer linear program
    #------------------------------------------------------------------

    # random seed
    random.seed(12345)

    # decision variables x_{i,j}
    prob = LpProblem("allocation", LpMinimize)
    x = LpVariable.dicts("x", (range(m), range(n)), cat=LpBinary)

    # cost: use scores from sheet, higher score for better choice, negate for minimize
    tm = np.zeros((m, n))
    for i in range(m):
        for j in range(num_choices):
            idx = tchoices[i, j]
            if idx >= 0:
                choice_num = j + 1  # 1-based
                tm[i, idx] = scores.get(choice_num, 0)
    fcost = -tm.flatten()

    # add small random tiebreak
    noise = np.zeros_like(fcost)
    for k in range(len(fcost)):
        if fcost[k] < -0.1:
            noise[k] = 0.001 * random.gauss(0, 1)
    fcost += noise

    # objective
    prob += lpSum(fcost[i * n + j] * x[i][j] for i in range(m) for j in range(n))

    # student constraints: at most one project per student
    for i in range(m):
        prob += lpSum(x[i][j] for j in range(n)) <= 1

    # project constraints: at most one student per project
    for j in range(n):
        prob += lpSum(x[i][j] for i in range(m)) <= 1

    # supervisor limits
    for s in range(len(scodes)):
        prob += lpSum(x[i][j] for i in range(m) for j in range(n) if psupi[j] == s) <= slimsn[s]

    # don't assign if not selected
    for i in range(m):
        for j in range(n):
            if tm[i, j] <= 0:
                prob += x[i][j] == 0

    prob.solve(PULP_CBC_CMD(msg=False))

    #------------------------------------------------------------------
    # Post-process results and write outputs
    #------------------------------------------------------------------

    xm = np.zeros((m, n), dtype=int)
    for i in range(m):
        for j in range(n):
            xm[i, j] = int(value(x[i][j]))

    # use -1 to indicate no assignment
    pass_idx = -1 * np.ones(m, dtype=int)
    for i in range(m):
        ones = np.where(xm[i, :] == 1)[0]
        if ones.size > 0:
            pass_idx[i] = ones[0]

    total_assigned = np.count_nonzero(pass_idx >= 0)
    print(f"Total assignments: {total_assigned}/{m} ({total_assigned/m*100:.1f}%)")

    # preference statistics
    pstud = np.zeros(m, dtype=int)
    for j in range(num_choices):
        nums = np.sum((tchoices[:, j] == pass_idx) & (pass_idx >= 0))
        print(f"Students assigned choice {j + 1}: {nums} ({nums/m*100:.1f}%)")
        pstud[(tchoices[:, j] == pass_idx) & (pass_idx >= 0)] = j + 1

    # self-proposed stats
    tt = (tchoices[:, 0] == pass_idx) & spsi & (pass_idx >= 0)
    num_yes = np.sum(tt)
    print(f"Students assigned self-proposed topic: {num_yes} ({num_yes/np.sum(spsi)*100:.1f}%)")

    # per-supervisor assignments
    A3x = np.zeros(len(scodes), dtype=int)
    for s in range(len(scodes)):
        ii = [j for j in range(n) if psupi[j] == s]
        A3x[s] = np.sum(xm[:, ii])
        print(f"{scodes[s]}: {A3x[s]}")

    # output assignment table
    outs = []
    pcodesa = pcodes.copy()
    pcodesa.append("unallocated")
    for i in range(m):
        if pass_idx[i] >= 0:
            project = pcodesa[pass_idx[i]]
            pref = int(pstud[i]) if pstud[i] > 0 else None
        else:
            project = "unallocated"
            pref = None
        outs.append([Ms[i, 0], project, pref])
    outst = pd.DataFrame(outs, columns=["Student", "Project", "Preference"])

    # unallocated projects
    unalloc = [pcodes[j] for j in range(n) if np.all(xm[:, j] == 0)]
    unalloc_df = pd.DataFrame(unalloc, columns=["Project"])

    # Save to single XLSX with multiple sheets
    output_fname = fname.replace('.xlsx', '_results.xlsx')
    with pd.ExcelWriter(output_fname) as writer:
        outst.to_excel(writer, sheet_name='allocations', index=False)
        unalloc_df.to_excel(writer, sheet_name='unallocated', index=False)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        file = sys.argv[1]
    else:
        file = "sample_input.xlsx"

    allocate_projects(file)
