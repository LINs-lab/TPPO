# Data

The following data files are used for training, testing, and analysis.

| File | Samples |
| --- | ---: |
| `data/polaris/qwen3-4b-s1-polaris-p1-format.jsonl` | 29916 |
| `data/countdown_3to4/test.jsonl` | 1024 |
| `data/countdown_3to4/train.jsonl` | 40000 |
| `data/countdown_5_ood/test.jsonl` | 256 |
| `data/countdown_5_ood/test_full.jsonl` | 1024 |
| `data/eval/agi_eval_en_prompted.jsonl` | 3340 |
| `data/eval/aime_2024.jsonl` | 30 |
| `data/eval/aime_2025.jsonl` | 30 |
| `data/eval/aime_2026.jsonl` | 30 |
| `data/eval/amc23.jsonl` | 40 |
| `data/eval/arc_c_prompted.jsonl` | 1172 |
| `data/eval/bigbenchhard_prompted.jsonl` | 6511 |
| `data/eval/math500.jsonl` | 500 |
| `data/eval/minervamath.jsonl` | 272 |
| `data/eval/olympic.jsonl` | 675 |
| `data/analysis/polaris_64_seed123.jsonl` | 64 |

Countdown: `Jiayi-Pan/Countdown-Tasks-3to4` and the existing five-number OOD data; the original parquet files and `stats.json` are retained as well. The original sources are [3-to-4 Countdown](https://huggingface.co/datasets/Jiayi-Pan/Countdown-Tasks-3to4) and [5-number Countdown](https://huggingface.co/datasets/yzhuang/tinyzero-Countdown-Tasks-5).
