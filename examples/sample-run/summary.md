# Run summary (`topology` grouping)

47 groups, 7 reported as incidents, 21 LLM calls (0.0 s).

| Group | Root cause | Location | Alarms | Confidence |
|---|---|---|---:|---:|
| G-007 | FIBER_CUT | CBL-002 | 7 | 90% |
| G-010 | POWER_LOSS | FEN | 14 | 90% |
| G-011 | FIBER_CUT | CBL-015 | 7 | 90% |
| G-012 | FIBER_CUT | CBL-001 | 7 | 90% |
| G-013 | PLANNED_WORK | CBL-010 | 6 | 85% |
| G-018 | HARDWARE_FAILURE | rtr-kes-01 | 7 | 85% |
| G-027 | POWER_LOSS | HAL | 2 | 90% |

## Against ground truth

- Grouping pairwise F1: 1.00
- Root-cause accuracy: 100%, location accuracy: 100%
- False incidents: 0

| Incident | True cause | True location | Group | Exact | Predicted | Location |
|---|---|---|---|---|---|---|
| INC-01 | FIBER_CUT | CBL-002 | G-007 | yes | FIBER_CUT | CBL-002 |
| INC-02 | POWER_LOSS | FEN | G-010 | yes | POWER_LOSS | FEN |
| INC-03 | FIBER_CUT | CBL-015 | G-011 | yes | FIBER_CUT | CBL-015 |
| INC-04 | FIBER_CUT | CBL-001 | G-012 | yes | FIBER_CUT | CBL-001 |
| INC-05 | PLANNED_WORK | CBL-010 | G-013 | yes | PLANNED_WORK | CBL-010 |
| INC-06 | HARDWARE_FAILURE | rtr-kes-01 | G-018 | yes | HARDWARE_FAILURE | rtr-kes-01 |
| INC-07 | POWER_LOSS | HAL | G-027 | yes | POWER_LOSS | HAL |
