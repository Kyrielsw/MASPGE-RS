# SDWPF KDD 245-day data

Experiment 037 uses only the public 245-day training CSV and turbine location
table. The separate KDD final-phase test archive is not downloaded or used.

Official source: Figshare version 2, CC BY 4.0.

Raw files expected under `data/raw/sdwpf_kddcup/`:

- `sdwpf_245days_v1.csv`, 334,304,460 bytes, MD5
  `8f0cc58c4b0d0fb809824035d05b4e76`;
- `sdwpf_baidukddcup2022_turb_location.csv`, 3,310 bytes, MD5
  `08bccca55c6cb2c7066cf3adb4f7aae9`.

The default deterministic preprocessing command preserves Experiment 037 and
creates:

- `data/sdwpf_kdd_245days_v1.npz`;
- `data/sdwpf_kdd_245days_v1.audit.json`.

Its compact role tensor has shape `[35280, 4]`. Experiment 043 uses the same
raw transformations and splits but writes a separate per-turbine contract:

```bash
python -u scripts/prepare_sdwpf.py \
  --contract dense_role_v2 \
  --csv data/raw/sdwpf_kddcup/sdwpf_245days_v1.csv \
  --locations data/raw/sdwpf_kddcup/sdwpf_baidukddcup2022_turb_location.csv
```

This creates `data/sdwpf_kdd_245days_role_state_v2.npz` with role shape
`[35280, 134, 4, 6]` and a separate audit JSON. Both contracts contain 35,280
synchronized 10-minute timestamps, use 171/37/37 day splits, and keep invalid
targets masked out of training and scoring.
