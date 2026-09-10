# Helper Scripts

These scripts support local development and are not stage entry points.

## `smoke_test.py`

Runs the generic instruction example and one representative Pretrain, Midtrain, and Posttrain profile. It checks JSONL output, valid decisions, and manifest/output record counts. Run it without arguments:

```bash
python3 scripts/smoke_test.py
```

Successful execution prints a JSON object with `status: ok`. Outputs are written to a temporary directory.
