# figures/

Presentation figures for the SL_results deck. Numbers come from
[`docs/RESULTS.md`](../docs/RESULTS.md) (the source of truth).

## Reproduce

```bash
python scripts/make_figures.py      # regenerates the deck charts below into figures/
python scripts/build_deck.py        # figures/ -> ~/Downloads/SL_results.pptx
```

## Script-regenerated (by `scripts/make_figures.py`)

| file | slide finding | distillation cell |
|---|---|---|
| `s_off.png` | off-policy transfer scales with unique data volume | off-policy · hard · forward-KL |
| `s_on.png` | on-policy: volume helps, prompt diversity flattens | on-policy · hard · reverse-KL |
| `s_collected.png` | owl is volume-gated; raven transfers from the smallest budget | on-policy · hard · reverse-KL |
| `k_parity.png` | k32 support truncation is free — both traits | on-policy · fresh · filtered |
| `x_channels.png` | owl concentrates in soft·fkl; raven transfers evenly | on-policy · fresh · filtered |
| `distillation_matrix_4axis.png` | soft·fkl is the one channel that survives filtering (deck slide 2) | policy × decode × granularity × direction |
| `scaling_ladder.png` | volume lifts transfer; iso-information controls collapse | cells × (canonical / scale / iso) |

## Archived static assets

Generators predate the script (recovered as rendered PNGs only). Re-script on request:

- `plot_scale_off.png`, `plot_scale_on.png`, `plot_scale_both.png` — earlier 2-point scaling line charts (rejected design; superseded by `s_*`)
- `askin_results.png` — Askin et al. (2605.12798) OPD comparison (external data)
