# ilpalloc

`ilpalloc` is a Python-based tool designed to automate the allocation of students to final-year projects. By treating the assignment process as an Integer Linear Programming (ILP) problem, it finds the globally optimal distribution of projects based on student preferences while strictly respecting supervisor workload limits.

## Features
- **Optimal Assignments:** Uses the `pulp` library to maximize overall preference satisfaction.
- **Supervisor Limits:** Ensures no supervisor is assigned more projects than their set capacity.
- **Self-Proposed Projects:** Automatically detects self-proposed projects (denoted by lowercase letters in the project code).
- **Tie-breaking:** Adds a slight randomization to the scoring matrix to gracefully handle identical preferences.

## Requirements

Ensure you have Python 3 installed along with the required dependencies. You can install the required packages using pip:

```bash
pip install pandas numpy pulp openpyxl
```
*(Note: `openpyxl` is required by pandas to read and write Excel files)*

## Usage

Run the script via the command line, passing the path to your input Excel file:

```bash
python ilpalloc.py path_to_excel_file.xlsx
```
If no file is provided, it defaults to looking for `sample_input.xlsx` in the current directory.

## Input Data Format

The input Excel file **must** contain the following three sheets:

1. **`choices`**: Contains the students and their project choices.
    * Column 1: `Student No` (or similar identifier)
    * Columns 2+: `Topic ID Choice 1`, `Topic ID Choice 2`, etc.
2. **`limits`**: Defines the maximum number of students a supervisor can take.
    * Column `Supervisor`: The supervisor's code (leading uppercase letters of a project code, e.g., 'ABC' for project 'ABC01'). Use `*` to define the default limit.
    * Column `Max`: Integer representing the maximum allocation limit.
3. **`scores`**: Defines the weight/score given to each preference level (used to optimize the assignments).
    * Column `Choice`: Integer (1, 2, 3...)
    * Column `Score`: Numeric value representing the priority (higher score = better choice).

### Project Code Quirks
* **Supervisor Codes:** The script assumes the leading uppercase letters of a project code correspond to the supervisor (e.g., project `FN01` belongs to supervisor `FN`).
* **Self-Proposed Topics:** Any project code containing a **lowercase** letter is flagged as a self-proposed project. 

## Output Data Format

The script will generate a new Excel file named `[original_filename]_results.xlsx` containing two sheets:

1. **`allocations`**: A list of all students, their assigned project (or "unallocated"), and the preference rank of that assignment.
2. **`unallocated`**: A list of projects that were not assigned to any student.

The script will also print useful statistics directly to the terminal, including assignment percentages, the number of self-proposed projects allocated, and per-supervisor loads.
