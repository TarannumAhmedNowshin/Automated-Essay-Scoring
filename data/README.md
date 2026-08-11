# Data

The essay dataset is **not** included in this repository. The scripts are
university-level Bangla answer scripts and are treated as private student data.

The notebooks expect the CSV files to be placed in this `data/` folder:

| File | Used by | Description |
|------|---------|-------------|
| `data/essays_small.csv` | `*-small` notebooks | ~325 graded scripts (BNG103, BRAC University) |
| `data/essays_large.csv` | `*-large` + LLM notebooks | ~1,408 graded scripts (extended set) |

### Expected columns

Each CSV must contain at least these columns:

- `Question` — the essay prompt
- `Answer`   — the student's written answer
- `Marks`    — the human-assigned score (0–10 scale)

Rows with missing `Answer` or `Marks` are dropped automatically, and marks are
rescaled to a 0–10 range inside the notebooks.

> If you use your own data, just match the column names above and drop the CSVs
> here with the expected file names.
