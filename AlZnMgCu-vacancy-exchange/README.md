# Local Cu geometry and vacancy-mediated Mg and Zn exchange in fcc Al

Research data, trained models, and source code accompanying the manuscript
**Local Cu geometry modulates vacancy-mediated Mg and Zn exchange in fcc Al**
by Jia Zhao, Sen Zhai, Kaijie Liu, Tiantian Jia, and Kenjiro Sugio.

## Download and contents

Download `AlZnMgCu-vacancy-exchange-data-code.zip` from this directory and extract it.
The archive contains the following folders. This directory is separate from the
other DFT-ML materials in the parent repository.

| Path in the archive | Contents |
|---|---|
| `Reference_dataset/` | Actual train, validation, and test extended-XYZ files with reference labels |
| `Figure_data/` | Tabulated results, site enumeration, data provenance, and model validation records |
| `DFT_paths/` | Eight final free/fixed-Cu DFT paths, seven structure images per path, energy profiles, settings, and summaries |
| `Initial_structures/` | Starting structures and atom-index maps |
| `Path_provenance/` | Input-band and calculation provenance records |
| `Reproduction/` | Standalone verification of Fig. 9 and the barrier/constraint analysis |
| `workflow/` | Original project source, configurations, three final Stage Two models, training logs, geometry inputs, and selected result records |
| `models_manifest.json` | Model filenames, seeds, file sizes, and SHA-256 checksums |
| `MANIFEST.csv` | SHA-256 checksums and file sizes for the archive payload |

The training file contains **543 frames**: 539 non-isolated configurations and
four isolated-atom references. Validation and test contain 53 and 92 frames,
respectively. The total is 684 non-isolated configurations plus four isolated
references. The training files in `Reference_dataset/` and `workflow/work/` are
byte-for-byte identical; the duplicate locations preserve the original commands.

The three supplied models have seeds **11, 29, and 47**. The seed-11 model checksum
matches the model checksum recorded for the original relaxed Cu-environment
endpoints. Original seed-specific validation records are retained without
changing pass/check flags.

## Quick verification without VASP or MACE

From the extracted archive root, with Python 3 and NumPy installed:

```bash
python Reproduction/recompute_fig9.py
python Reproduction/recompute_barriers.py
python Reproduction/verify_manifest.py
```

The first command reads the 28 final free-Cu POSCAR images and checks the saved
Fig. 9 tables to 1e-9 angstrom. Local Al atoms are chosen once in image 00 at
Cu-Al distance strictly below 3.1 angstrom and followed throughout the band.
Displacements use periodic minimum images and subtract the mean displacement of
all Al atoms. The horizontal coordinate is image index i/6, not arc length.
The second command recomputes all eight forward barriers from the supplied
energy profiles and reports direct and far-Cu-corrected constraint responses.
Neither command runs new electronic-structure calculations.

## Training and MACE workflow

The original dependency specifications are provided in `workflow/requirements.txt`,
`workflow/pyproject.toml`, and `workflow/environment.yml`. The project specifies
Python >=3.10, ASE >=3.23, and optional MACE >=0.3.13,<0.4 with PyTorch >=2.2.
These are the saved project specifications, not a claim that a full frozen
training environment was recovered. Training logs and DFT settings retain
additional version information where available.

```bash
cd workflow
python -m pip install -e '.[analysis,ml]'
bash scripts/28_train_v6_vacancy_weight_cpu.sh 11
bash scripts/28_train_v6_vacancy_weight_cpu.sh 29
bash scripts/28_train_v6_vacancy_weight_cpu.sh 47
```

The original training driver skips a seed when its final checkpoint is already
present. Re-training must be done in a separate copy with a new output location;
the supplied model files are the reference artifacts, not regenerated substitutes.
Original validation commands are in `scripts/29_validate_v6_ensemble.sh`.
Those full workflow stages also need their original generated DFT task trees;
the saved comparison tables are supplied, but not every historical scratch file.

The Cu-environment workflow uses `scripts/build_cu_environment_paths.py`,
`scripts/relax_cu_endpoints.py`, `scripts/run_cu_mace_neb.py`, and
`scripts/cu_dft_workflow.py`. Use each script's `--help` for arguments. Geometry
and endpoint records are in `workflow/work/cu_environment_v1/`.
`workflow/cu_story_supplement_v2/` and `workflow/original_plotting/` preserve
historical source; their original filesystem references and earlier analysis
definitions may require adaptation. For the final Fig. 9 definition, use the
standalone `Reproduction/` implementation and the final `Figure_data/` tables.

## DFT and constraint data

Each `DFT_paths/{free,fixed}/<task>/` includes a `profile.csv`, `settings.json`,
`summary.json`, and `band/00` through `band/06/POSCAR`. Settings record the
electronic parameters, energy convention, host-frame convention, Cu constraint,
and available pseudopotential hashes. Original input bands and endpoint identity
can be traced using the provenance records. Historical absolute paths are retained
as provenance and must be changed for a new computing site.

VASP executables and POTCAR datasets are not redistributed. New DFT calculations
require the user's own licensed VASP installation and matching pseudopotentials.
This archive contains final path structures and energy/settings records rather
than every VASP binary or electronic wavefunction file.

## Scope and use

This release packages the saved research records. The quick verification checks
the supplied structural and energetic tables; it is not an independent re-run of
model training or DFT convergence. Later convergence scans mentioned separately
are not included unless present in the saved records.

The release does not include the manuscript source, review correspondence,
account credentials, Python virtual environments, or proprietary VASP files.
Please cite the associated article when using these research materials.
