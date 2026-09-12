# Selection-description correction — 2026-09-12

The frozen `TRUE_FACT_CONTROL_DESIGN.json` (SHA-256 `0abc81d123a85911b9b0a2ced3d0775b1e326dc7b82245ef78466211f9638a62`) incorrectly describes InternLM2.5-7B as the median of seven original detector CCRs.

The ascending rates were 0.040, 0.205, 0.270, 0.2725, 0.28125, 0.29125 and 0.45625. Qwen2.5-7B is the median; InternLM is fifth in ascending order. The highest and lowest model descriptions were correct.

The three model identities were fixed before control generation. This correction changes neither that selection nor the tasks, outputs, scoring, intervals or original design file. All three models remain reported. The preparation script now describes the actual selection without the erroneous median claim. No new selection rationale is asserted retrospectively.
