# Human-in-the-Loop Annotation Pipeline

_Populated in Stage 1 (see PRD §6)._

This document will describe:
- Auto-generation of candidate queries with `gpt-4o-mini` + few-shot prompts.
- The reviewer CLI (`ingest/annotate.py`) workflow: accept / edit / reject.
- Two-pass review protocol (solo reviewer, 24-hour gap) and Cohen's kappa computation.
