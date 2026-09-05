---
slug: compress-anomaly-phase1-harness
status: awaiting-approval
intent: clear
review_required: true
pending-action: write and review .omo/plans/compress-anomaly-phase1-harness.md
metis-review: ses_f8fae7843ffeoy2GBGdJQUpSzN - 18 gaps (4 critical / 9 major / 4 medium / 1 minor), all folded into todos as METIS-F## tags
review:
  momus: { status: changes_requested_round2, session: ses_f8faad9acffewFt1ZW4G955E85, issues: 15 }
  independent: { status: changes_requested_round2, session: ses_f8faad9afffegODaFRbfzNEbVi, issues: 17 }
fixes-applied: all 30+ cited issues incl. mediums — VUS pinned vus==0.0.6+SLIDING_WINDOW=32+vendored fallback; tau key per(seed,dataset,detector) on R0-train; CSV 12 cols (+payload_bytes); exact cell matrix + --seeds/--offline CLI; R4 ~32x band [16,40]; Q8 [3.8,4.1] Q4 [7.5,8.2]; byte-format/nibble/padding/minmax freezes; SAX train-stats z-norm + ppf breakpoints + midpoint recon; D4 1st-order noseed; asserts named-detector inequalities (non-collapse + 2 cliff gates); mean aggregation; f0=0.02 jitter/sigma/amplitude PLAN-FREEZEs; alignment home metrics/alignment.py; UCR pin-first inside todo 4; placeholder test + concrete fixtures + behavioral VUS conformance; unified evidence naming; exact gitignore lines; todo 10 split into gates/docs; todo 12 zero-code rule; FETCH-FAILED schema without placeholder rows
review-round-3: { momus: approved_ses_f8fa75054ffeueOT1nfaNzNdd8_2minors, independent: changes_requested_ses_f8fa75051ffeZBl56WtHax4BlI_12issues }
fixes-applied-round4: VUS_WINDOW=64 decoupled + sensitivity check + pip-or-vendored acceptance; R4 band [14,34]; rhythm noiseless<=1.0 then noise (noisy<=1.3); Rhythm grammar gate (d) + cliff (c) tightened <=0.20; CSV 14 cols (+rmse +gpu_model per candidate-designs:4); requires-python>=3.11; Q8 [3.75,4.15] Q4 [7.7,8.1]; alignment test_alignment.py + ARCH note; UCR single-region anomaly-index check; PAA-16/PA%K/PATE parked register + grep gate; root-spelling + import note; traceability appendix for METIS tags; matrix 2->7 edge; 5 fixtures; todo-12 zero-code rule kept
review-round-4: { momus: APPROVED_ses_f8fa75054ffeueOT1nfaNzNdd8, independent: changes_requested_ses_f8fa75051ffeZBl56WtHax4BlI_5residuals }
fixes-applied-round5: wrapped L79/L86 (no >2000ch lines), UCR SUPERSEDED/ACTIVE reselection rule, grep pa%k|pate|affiliation|point_adjust, check_docstrings.py owned by todo 11, RMSE-only freeze (PRD dropped w/ rationale)
review-round-5: { independent: APPROVED_ses_f8fa1f4f4ffeno25QfmNXn5e2L_digest_a36999e8, momus: round4_APPROVED_carried__round5_aborted_by_user__planner_selfverified_12impl_4final_matrix_fixtures }
live-validation: plan_sha256=a36999e8c2839e7c6749cdba6a49054a11d97edf6f828a7836fd27ab112c1241 matches approved digest, no edits since — receipts valid
status: approved-for-handoff
approach: Phase-1 TASK_01 harness + VUS-PR in /home/shreyas/projects/compress-anamoly/ (harness/ package per TASK_01 section 3), uv + Py3.11 pinned, serena + Context7-latest docs, tests-after + QA, local git with per-component commits, full docs set, multi-seed eval with hard bump/cliff assert gates
---

# Draft: compress-anomaly-phase1-harness

## Components (topology ledger)
- scaffold | uv Py3.11 project + harness/ layout + local git init in /home/shreyas/projects/compress-anamoly/ | status: active | evidence: TASK_01.md:47-76
- datasets | 4 morphologies N=2000 (800 train / 1200 test) + pinned live UCR-250 loader | status: active | evidence: TASK_01.md:82-106
- codecs | R0 lossless + R3 Q8/Q4 + R4 SAX PAA + tokens, dual Path A/B | status: active | evidence: TASK_01.md:110-146
- detectors | PCA w=32 + IF + D4 Markov-on-tokens | status: active | evidence: TASK_01.md:150-178
- metrics-runner | frozen-tau + point/event-F1 + PR-AUC + VUS-PR + run_harness matrix + CSV | status: active | evidence: TASK_01.md:182-205
- qa-docs-eval | pytest suite + full docs + eval vs expected bump/cliff table | status: active | evidence: TASK_01.md:208-274

## Open assumptions (announced defaults)
- VUS-PR source: TSB-AD/papangelou reference implementation, pinned dep (or vendored if no clean pin) | rationale: no sklearn VUS-PR; Context7 check at build | reversible: yes

## Findings (cited - path:lines)
- TASK_01 is decision-complete spec: harness/ tree, morphology formulas, codec/detector APIs, frozen-threshold contract (TASK_01_FIRST_RESEARCH_EXPERIMENT.md:1-76)
- Scope extension to VUS-PR approved by user; metric lock VUS-PR + point/event-F1, PA-F1 banned, thresholds frozen (04-synthesis/REPORT.md:57; FINAL-REPORT-PHASE3.md:56-76)
- Eval pitfalls: stratify never pool; frozen-vs-retuned audit; denoising bump expected not clipped; boring-flat reported honestly (Master_Report.pdf: sections 2.2/2.3/2.4/2.6/8.4)
- Problem decomposition P1-P4 + build order I1>I2>I6>I3-lite>I4>I7>I5-parked (FINAL-REPORT.md:106-122); D4-symbolic verdict + X1-X5 ideas (FINAL-REPORT-PART4.md:118-140)
- H0 PASS 75-80%, SHRINK-v2 WEAKENS 2.5/3, KIT settled NO-THREAT (FINAL-REPORT-PHASE3.md:18-30)
- Target dir /home/shreyas/projects/compress-anamoly/ exists and is empty; serena project activated

## Decisions (with rationale)
- scope: TASK_01 + VUS-PR now (user chose; VUS-PR primary alongside PR-AUC/point/event-F1)
- data: live pinned UCR-250 download + checksum + cache fallback (user chose; reproducibility without vendoring bytes)
- tests: tests-after + full-matrix QA (user chose; fastest for research harness)
- deps: Python 3.11, numpy 1.26.x, scipy, sklearn 1.4+, tabulate, uv (user chose; matches TASK_01 section 7)
- git: local-only init, main branch, per-component commits, no remote (user chose)
- docs: full set - README + docs/ + docstrings (user chose)
- eval: hard asserts on denoising bump (spike Q8 >= R0) + drift cliff (drift Q4/SAX collapse) with tolerances (user chose)
- seeds: multi-seed runs with mean + spread, seed budget disclosed (user chose)
- ucr: worker picks stable labeled UCR-250 series, pins URL + SHA in dataset-freeze.csv (user chose)

## Scope IN
- harness/ package exactly per TASK_01 section 3 tree + results/ + tests/ + docs/ + eval/
- All 4 morphologies with exact formulas + seeded RNG; live-pinned UCR-250 loader
- R0/R3(Q8/Q4)/R4(SAX) codecs with byte-exact ratios; Path A decode + Path B tokens
- PCA/IF/D4 detectors per spec; frozen-tau evaluator + point/event-F1 + PR-AUC + VUS-PR
- run_harness.py full matrix + Markdown table + results/baseline_run.csv
- pytest suite, docs set, eval script with bump/cliff asserts, dataset-freeze.csv, git init + commits

## Scope OUT (Must NOT have)
- D2 router, D3 edge/Telegraf, D5 neural, full 7-rung ladder (Chimp/Elf/ALP/Machete/rollup), FM pilots, PATE headline, cloud-shadow eval, GitHub remote/push

## Open questions
- none blocking; all forks answered

## Approval gate
status: awaiting-approval
brief-presented: yes
next-after-okay: rerun scaffold without --draft-only, run Metis gap analysis, APPEND todos, fill TL;DR last, deliver handoff
