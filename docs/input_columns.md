# Input Columns

The app accepts `.csv`, `.xlsx`, and `.xlsm` files.

Required or important columns:

| Column | Meaning | Usage |
|---|---|---|
| `A-TOBT` | Airport target off-block time | Primary prediction anchor when available |
| `A-DOBT` | Airport default/decision off-block time | Fallback anchor when `A-TOBT` is missing |
| `实际移交机坪管制` | Actual handover-to-apron-control time | Training and evaluation target |
| `CTOT` | Calculated take-off time | Time parameter feature |
| `IFC` | Airline code | Categorical profile and per-airline comparison |
| `CLA` | Flight number | Detail output identifier |
| `TAR` | Stand/parking position | Categorical profile feature |
| `ITY` | Aircraft type | Categorical profile feature |
| `RWYA` | Arrival runway | Categorical profile feature |
| `RWYD` | Departure runway | Categorical profile feature |

Support-node columns used as pre-departure/turnaround features include:

- `进近管制移交`
- `准备落地`
- `进港等待穿越`
- `进港滑行`
- `装卸开始`
- `装卸结束`
- `关货舱门`
- `拖车到位`
- `关客舱门`
- `关舱门`
- `机组到位`
- `机务放行`
- `开始配餐`
- `配餐完成`
- `供油开始`
- `供油完成`
- `登机口开启`
- `登机口关闭`
- `离桥完成`
- `离港客梯车撤离`

The app does not use actual post-pushback/taxi/takeoff outcomes as prediction inputs.

