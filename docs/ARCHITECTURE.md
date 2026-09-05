# Architecture (skeleton)

Modular harness layout at repo root:

- `harness/datasets/` — 4-morphology seeded generator + UCR loader with checksum/cache
- `harness/codecs/` — R0 raw, R3 quantization, R4 symbolic
- `harness/detectors/` — PCA, Isolation Forest, D4 direct-symbolic
- `harness/metrics/` — frozen threshold evaluator, trailing alignment, VUS-PR
- `harness/results/` — run outputs

Details land in later tasks. No logic yet — skeleton only.
