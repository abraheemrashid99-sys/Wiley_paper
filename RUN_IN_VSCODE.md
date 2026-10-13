# Running and verifying the study in VS Code

Everything below works on Windows, macOS and Linux. You need
[VS Code](https://code.visualstudio.com/) with the **Python** extension and
Python 3.10 or newer (`python --version`).

## 1. Open the folder

Unzip the release, then in VS Code: **File → Open Folder…** and choose the
unpacked folder (the one that contains this file). Trust the folder when asked.

## 2. Run the tasks, in order

**Terminal → Run Task…** shows the tasks of this project. Run them from the
top; each opens its own terminal panel and says when it is finished.

| Task | What it does | Time |
| --- | --- | --- |
| `DN-SC 0: create environment and install` | A private Python environment in `.venv` with all dependencies | a few minutes, once |
| `DN-SC 1: verify the package (checksums)` | Confirms every file is exactly as released | seconds |
| `DN-SC 2: run the tests` | 90+ unit and integration tests on a tiny offline model | 5–15 min |
| `DN-SC 3: smoke run of the whole pipeline` | Every script end to end on a tiny model (its numbers mean nothing) | a few minutes |
| `DN-SC 4: fetch GPT-2` | Builds pretrained GPT-2 from files on GitHub, checksum verified | once, about 650 MB |
| `DN-SC 5: reproduce the published results and compare` | Re-runs the 37 published stream runs and all evaluations, then compares | long (see below) |
| `DN-SC 6: confirmatory study (six seeds) and compare` | The protocol of `docs/METHOD.md`, section 9; contains task 5 | longer |
| `DN-SC 7: full paper study` | Task 6 plus every sensitivity study | longest |

After task 0, VS Code may ask whether to use the new environment for this
folder: answer **Yes**.

Tasks 5–7 can be stopped (click in the terminal, press `Ctrl+C`) and started
again at any time: finished runs are skipped. Keep the laptop on mains power
and stop it from sleeping while they run.

## 3. Read the verdict

Tasks 5–7 end by writing a comparison report and printing it:

- task 5: `results_repro/gpt2_published/COMPARISON.md`
- task 6: `results_repro/gpt2_core/COMPARISON.md`
- task 7: `results_repro/gpt2_paper/COMPARISON.md`

Open it in VS Code and press `Ctrl+Shift+V` (macOS: `Cmd+Shift+V`) for the
formatted view. It contains, in this order:

1. **Verdict** — `REPRODUCED` (every headline number within 3 % and every
   conclusion holds) or `CHECK NEEDED` with the list of what differs.
2. **Provenance** — machine, library versions and code revision of both sides.
3. **Headline numbers** — published and reproduced value, side by side.
4. **Conclusions of the paper** — whether each holds in the reproduction.
5. **Measured by the reproduction only** — what refinement costs once its
   signalling is charged, and the free-running audit (new in version 0.3).
6. **Every stream run, evaluation file and table**, with the largest
   difference found.

Numbers produced on different hardware are not bit-identical: floating-point
arithmetic differs between processors, and online adaptation carries such
differences along. That is why agreement is judged against a tolerance and
against the conclusions. On the same machine, with the same library
versions, a run is repeatable bit for bit.

The reproduced tables and figures are next to the report
(`TABLES.md`, `tables/`, `figures/`).

## How long tasks 5–7 take

Measured reference: on two CPU cores an adaptive stream run takes about
30 minutes and a run without adaptation about 10. Task 5 holds 37 stream
runs, task 6 holds 88 and task 7 holds 120: as a guide, about 16, 36 and
51 hours of single-core computing for the stream runs (estimated from the
run times stored in `results/gpt2`; the free-running audit of version 0.3
adds to it). A
laptop with more cores is faster; `DN-SC 5` prints a speed benchmark for your
machine as its second step. With an NVIDIA GPU the runs are several times
faster (the tasks pick the GPU automatically).

To run the same commands by hand, or with other options (for example
`threads=8`, or `--seeds 0,1,2,3,4,5,6,7,8,9`), see the top of
`scripts/reproduce.py`; `DN-SC 8` prints the full plan without running it.

The tasks train the codebook again from scratch, which is the stricter test.
To start instead from the published codebook (`results/gpt2/base/codec.pt`)
— this removes the largest source of numerical differences between machines
— run the command of the task in the terminal with `--reuse-codebook` added.

## If something fails

- **A task stops with `STOPPED: step N … failed`**: the message above it is the
  cause. Fix it and run the same task again; finished work is kept.
- **`python` is not found (task 0)**: install Python from python.org, tick
  "Add python.exe to PATH", restart VS Code.
- **Downloads fail**: tasks 4–7 need internet access to GitHub (model and
  corpora). Nothing is downloaded from Hugging Face in the GPT-2 study.
- **Out of memory**: close other programs; the GPT-2 study needs about 4 GB.
