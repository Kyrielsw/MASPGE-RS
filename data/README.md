# Data contracts

Raw and processed datasets are excluded from Git. The repository stores only
preprocessing code, role contracts, expected hashes, and verification tools.

## SMARTEOLE

- Source: [Zenodo record 7342466](https://doi.org/10.5281/zenodo.7342466)
- Raw archive MD5: `ffecd4b8cfe3ef3c744a5e0ff4a981c2`
- Processed file: `data/s1_complete_case_v0.1.npz`
- Processed SHA256: `6433e89f3a29751eacea375f9cbc7fcd19bceed0dc59c74ddc7a2b27770bd466`
- Units/roles: seven turbines; A1--A3 active and A4 masked

After extracting the official archive to the paths specified in
`configs/smarteole_role_contract_v1.json`, run:

```bash
python scripts/prepare_smarteole.py
python scripts/verify_setup.py
```

## SDWPF

- Source: [Figshare SDWPF dataset](https://figshare.com/articles/dataset/SDWPF_dataset/24798654)
- Processed file: `data/sdwpf_kdd_245days_role_state_v2.npz`
- Units/roles: 134 turbines; A1--A3 active and A4 masked

See [SDWPF_README.md](SDWPF_README.md) for raw hashes and the exact command.

## HAI 23.05

- Source: [official HAI repository](https://github.com/icsdataset/hai/tree/master/hai-23.05)
- Processed file: `data/hai_23_05_role_state_v1.npz`
- Processed SHA256: `266a98f5fff19ed07774279f8276f171e35d021abd2aefc360ba94c69c565be6`
- Units/roles: thermal and hydro branches; A1--A4 active

See [HAI_README.md](HAI_README.md) for admission auditing, preprocessing, and
the no-cross-file-window contract.

## Leakage boundary

Normalization statistics are fitted on training data only. Splits are
chronological, and windows never cross an accepted run or source-file
boundary. Dataset hashes are checked before training. Holdout evaluators do
not contain an optimizer or training path.
