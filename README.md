# Text-to-SQL Agent Design, Ablation, and Trajectory Analysis - CISC451: Assignment 2

This submission contains two notebooks. The Task 2 notebook re-runs the full Task 1 pipeline
before adding its own task-specific analysis, so each can be run independently, top to bottom:

- `Asg2_Task1.ipynb` - turns the Assignment 1 LLM-alone solution (System A) into an agent
  (System B) with Qwen2.5-7B-Instruct, diagnoses System A's failures on a development set, and
  compares System A and System B on the 200-question evaluation set (accuracy, efficiency,
  agreement, failure patterns).
- `Asg2_Task2.ipynb` - runs the Task 1 pipeline (Sections 0-12), then adds the ablation study
  (one variant per component), trajectory feature extraction, descriptive statistics, statistical
  tests, a figure, and the trajectory walkthroughs (Sections 14-21).

## Requirements

### Python environment
Python 3.10+ (developed on 3.13). Install dependencies:

```python
pip install pandas numpy tqdm requests scipy matplotlib scikit-learn nltk sqlparse
```

Qwen2.5-7B-Instruct is accessed through the OpenRouter API, so no local model or GPU is needed.

After installing `nltk`, download its tokenizer data once (needed by the official evaluation script):
```python
import nltk
nltk.download('punkt')
```

### External data

**1. Spider dataset.** Download the full Spider dataset (not just the GitHub code
repo, which does not include the data) from the official source linked at
https://github.com/taoyds/spider. Extract it so the following files exist:
```
spider-master/evaluation_examples/examples/dev.json
spider-master/evaluation_examples/examples/tables.json
spider-master/evaluation_examples/examples/train_spider.json
```
`train_spider.json` is the retrieval pool of the agent. If your extracted folder structure
differs, update `SPIDER_DIR` (and `TRAIN_JSON`) in each notebook's Setup cell accordingly.

**2. Spider SQLite databases.** These are a separate, large (~1GB) download from
the same Spider dataset source (not included in the base GitHub repo or a plain
code clone). After extracting, you should have one subfolder per database, each
containing a `.sqlite` file (e.g. `database/concert_singer/concert_singer.sqlite`).
Set `DB_DIR` in each notebook's Setup cell to the absolute path of this folder.

**3. Official Spider evaluation script.** Download/clone
https://github.com/taoyds/test-suite-sql-eval. If downloaded as a ZIP rather than
via `git clone`, the extracted folder will be named `test-suite-sql-eval-master`
(matching what's currently set in `EVAL_SCRIPT_DIR`); if cloned via git, rename the
folder or update `EVAL_SCRIPT_DIR` to match.

**4. Assignment 1 evaluation IDs.** Both notebooks reuse exactly the same 200 evaluation
examples as Assignment 1 (they are not re-sampled). The IDs are read from
`./results/eval_subset_ids.json` (`A1_OUTPUT_DIR`), the file saved by the Assignment 1 notebooks.
A copy is saved by this assignment as `results_a2/eval_subset_ids_copy_from_A1.json`; if the
Assignment 1 `results/` folder is not available, copy that file to `results/eval_subset_ids.json`.

### API key

Only one provider is used:

- **OpenRouter** (Qwen2.5-7B-Instruct, used for System A, System B, and every ablation variant):
  create a regular API key at https://openrouter.ai/settings/keys.

Each notebook's "Inference Loop" cell (Section 4) currently contains:
```python
os.environ["OPENROUTER_API_KEY"] = "ENTER API KEY HERE"
```
Replace this placeholder string with your own key before running.

## Known machine-specific settings to check before running

A few paths in the Setup cells were set for the original development machine and
will need to be updated to match your own environment:

- `SPIDER_DIR` (default: `"./spider-master"`) - must point to wherever you
  extracted the Spider dataset (`TRAIN_JSON` is built from it).
- `DB_DIR` - currently an absolute path specific to the original machine
  (`/Users/.../Downloads/Assignment 2/database`). This must be changed to wherever
  the SQLite database folder was extracted on the machine running this code.
- `EVAL_SCRIPT_DIR` (default: `"./test-suite-sql-eval-master"`) - must point to
  the cloned/extracted evaluation script repo.
- `A1_OUTPUT_DIR` (default: `"./results"`) - must contain `eval_subset_ids.json` from Assignment 1.

A quick sanity check to run after setting these (before running the full
pipeline):
```python
import os
print(os.path.exists(DEV_JSON), os.path.exists(TABLES_JSON), os.path.exists(TRAIN_JSON))
print(os.path.exists(DB_DIR), os.listdir(DB_DIR)[:5])
print(os.path.exists(os.path.join(EVAL_SCRIPT_DIR, "evaluation.py")))
print(os.path.exists(os.path.join(A1_OUTPUT_DIR, "eval_subset_ids.json")))
```
All of these should print `True` (and a list of real database folder names) before
proceeding.

## How to run

1. Install dependencies and download the external data/key above.
2. Update the machine-specific paths (`SPIDER_DIR`, `DB_DIR`, `EVAL_SCRIPT_DIR`, `A1_OUTPUT_DIR`)
   and the OpenRouter API key in the Setup / Inference Loop cells.
3. Run `Asg2_Task1.ipynb` top to bottom. This loads the 200 Assignment 1 evaluation IDs, samples a
   50-example development set (fixed seed = 2026), runs System A on the development set and
   diagnoses its failures, builds and runs the agent (System B) on the development set, then runs
   System A and System B on the evaluation set, evaluates both with the official Spider
   test-suite evaluator, and produces the summary table, agreement table, failure-pattern
   frequencies, and the agent-loop figure.
4. Run `Asg2_Task2.ipynb` top to bottom. Sections 0-12 independently re-run the Task 1 pipeline;
   Sections 14-21 then run the three ablation variants (`AB_no_retrieval`, `AB_no_lookup`,
   `AB_no_verifier`), build the comparison tables, extract the trajectory features, compute the
   descriptive statistics and statistical tests, draw the figure, and print the two trajectory
   walkthroughs.

Every run is logged to `./results_a2/` and is resumable: if a `*.jsonl` log already exists
(for example from `Asg2_Task1.ipynb`), `run_system` reuses it without calling the model again.
Delete the log files in `results_a2/` to recompute a system from scratch.

## Outputs

Both notebooks write their outputs to `./results_a2/`:

- `System_A_{dev,eval}.jsonl`, `System_B_{dev,eval}.jsonl` - one record per question with the
  predicted SQL, stop reason, token counts, time, and the full trajectory (each step's action,
  input, observation, and token usage).
- `AB_no_retrieval_eval.jsonl`, `AB_no_lookup_eval.jsonl`, `AB_no_verifier_eval.jsonl` - the same
  logs for the three ablation variants (Task 2 notebook).
- `*_gold.txt`, `*_pred.txt`, `*_official_eval.txt` - gold SQL, predicted SQL, and the official
  evaluator report of each system and variant.
- `dev_set_ids.json`, `eval_subset_ids_copy_from_A1.json` - the development-set IDs (with source and
  seed) and the evaluation-set IDs.
- `task1_summary.csv`, `task1_agreement.csv`, `agent_loop.png` - Task 1 results table, agreement
  table, and the agent-loop figure.
- `task2_ablation_table.csv`, `task2_fixed_broken.csv`, `task2_fixed_broken_ids.json` - ablation
  results and the questions each component fixed and broke.
- `task2_trajectory_features.csv`, `task2_features_by_correctness.csv`,
  `task2_features_by_level_correctness.csv`, `task2_tests_solved_vs_unsolved.csv`,
  `task2_trajectory_features.png` - trajectory features, descriptive statistics, statistical
  tests, and the figure used in the written report.

## Notes and limitations

- Headline exact-match and execution-accuracy numbers come from the official
  `test-suite-sql-eval` script. Per-question correctness and difficulty level (needed for the
  agreement table and the by-difficulty analyses) are obtained by running the same unchanged script
  once per question; the notebooks check that these agree with the single batch run.
- The agent only reads the schema and runs read-only queries on the question's database, and may
  retrieve solved examples from Spider's training split. The gold SQL of the evaluation set is used
  only by the evaluation code, never in a prompt or in the agent's context.
- System A is re-run in this assignment (the prompt and settings are unchanged from Assignment 1)
  only to additionally record input tokens; all System A numbers in the report come from this re-run.
- The two walkthrough questions are selected by example ID in Section 21 of `Asg2_Task2.ipynb`
  (`CHOSEN_FIXED_ID` and `CHOSEN_BROKE_ID`). If a fresh run produces different outcomes, choose new
  IDs from the candidate tables printed in that section.
- Each model run makes live calls to the OpenRouter API; results may vary slightly run to run
  depending on provider-side load even at temperature 0. System B needs about 4-5 seconds per
  question, so a fresh evaluation-set run of System B or of one ablation variant takes roughly
  15 minutes, and a full fresh run of both notebooks can take about 1.5 hours
  (see each notebook's progress/print statements).
