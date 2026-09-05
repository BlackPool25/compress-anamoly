# compress-phase2-ladder learnings

## 2026-09-06 Task T4 — Deadband R2 codec dual-eps
- Value-deviation deadband (keep if |x - last_kept| >= thr) expands badly on noisy
  series (ratio 0.62 on fixture); chord-check (skip i while skipped points stay within
  thr of chord last_kept -> i) is the true error-bounded scheme and provably gives
  max|decode - x| <= eps * range (measured exactly eps on all pilot series).
- Honest measure-then-freeze outcome: deadband EXPANDS (ratio < 1) on noisy synth
  (each kept point costs 8 B vs 4 B raw), so initial bands [1.5,5.0]/[2.0,8.0] were
  amended with evidence to R2a [0.6, 5.5] / R2b [0.8, 7.5]; never cherry-pick the
  pilot to fit a band (METIS-F05).
- Gradient preservation holds with margin: Drift-42 RMSE(R2a)=0.0123 < RMSE(Q4)=0.0697.
- Force-keep every 512 samples bounds encode to O(n*512) for ultra-smooth UCR-scale
  stretches (marked with ponytail: ceiling comment); exactness unaffected (collinear).
- Infra notes: Serena edit tools fail without LSP servers in this project — use
  Read/Edit/Bash fallbacks for edits; `tests/test_codecs_r4_bypass.py` collection error
  is Todo 6's parallel-worker TDD-red file, not a regression — exclude, don't touch.

## 2026-09-05 Task T3 — Gorilla R1 codec exact
- Exact Gorilla = `0` / `10`+reuse / `11`+5-bit lead+6-bit len+bits, MSB-first,
  header `[count u32LE][first f32LE raw bytes]`; header MUST use `flat[:1].tobytes()`,
  never struct float round-trip (float64 interp canonicalizes NaN payloads — same trap
  in test construction: assign NaN bits via uint32 view, not `y[i] = unpack(...)`).
- Honest measure-then-freeze: noisy float32 sensor data gives 0.9405-0.9590 (synth
  seeds 42-46) and 0.9422 (real UCR test slice N=47877); initial [1.5, 3.5]
  superseded to frozen (0.90, 1.00). R1 value = bit-exactness, not ratio (METIS-F05).
- Strictness pays: decode rejects >= 8 trailing bits (valid payloads pad < 8), so
  `payload + b'\x00'` raises instead of passing silently; window-reuse with no stored
  window and impossible lead/len geometries raise with index-attached messages.
