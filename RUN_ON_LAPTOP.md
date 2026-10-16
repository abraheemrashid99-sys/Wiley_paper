# The primary comparison on a laptop, today

`scripts/run_quick.py` runs the comparison the paper's headline rests on —
the frozen receiver (reference), `nested` (the proposal) and `context_only`
(the free memory alone) on the hierarchical drift stream — with the decoder
of version 0.4 and the published codebook, several runs in parallel, then
writes the tables. It is `--profile core` restricted to the runs of the
primary endpoint, on the CPU. Nothing in the method or the measurement is
changed.

Default: seeds **3, 4 and 5** (never run before). Seeds 0, 1 and 2 of the
same runs are made separately with the same code and codebook; pooled, the
six seeds allow the seed-level test of `docs/METHOD.md`, section 9.

## Windows, VS Code — commands for the terminal (Terminal → New Terminal)

Unzip the release, open the unzipped folder in VS Code (File → Open Folder),
then paste these lines one block at a time.

**1. Environment (once, a few minutes).** If an older version is unpacked
elsewhere, still do this here: the environment belongs to the folder.

```
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev,gpt2]"
```

If PowerShell refuses to run `Activate.ps1`, run
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, answer `Y`, and
repeat the activation line. If `py` is not found, use `python`.

**2. Check the package (seconds).**

```
python scripts/verify_package.py
```

It must end with `PACKAGE VERIFIED`.

**3. Run the study.** It fetches GPT-2 and the two corpora the first time.

```
python scripts/run_quick.py
```

Every 10 minutes it prints where each run is (`step …/600`) and, once a run
has finished, an estimate of the time left. Keep the laptop on mains power
and stop it from sleeping (Settings → System → Power → Screen and sleep →
Never). The runs of the reference and of the proposal come first, so the
primary comparison is complete before `context_only` starts.

**Stopping and continuing.** `Ctrl+C` stops it; the same command continues
(finished runs are kept, an unfinished one starts over).

**4. The results.** At the end it prints the primary endpoint and writes

- `results_quick/TABLES.md` — the report (open it and press `Ctrl+Shift+V`);
- `runs/quick/tables/realdecode_hier.md` — the primary endpoint, decoded for
  real, with 95 % intervals and the seed-level test;
- `quick_results.zip` — everything, to send on.

**Tables before everything has finished** (for instance if time runs out):

```
python scripts/run_quick.py --report-only
```

**Pooling with the runs of seeds 0–2** (a zip of their `stream/*.json`, or
a folder):

```
python scripts/run_quick.py --report-only --add-runs seeds_0_1_2.zip
```

## How long

On one CPU thread a run took 59 min (`frozen`), 66 min (`context_only`) and
87 min (`nested`) on the machine of `results/gpt2_v0.4_check`. The script
runs as many in parallel as the laptop's cores and memory allow (about 2 GB
each; it prints its choice); with four in parallel the nine runs take about
three to four hours, the six runs of the primary comparison about two.
`--seeds 3,4` or `--methods frozen,nested` make it shorter.

The tables are generated from the result files; nothing is typed by hand.
Runs made on different computers differ in the last digits of
floating-point arithmetic; the codebook and the code are the same.
