# MACE model requirements

Place a trusted compatible model at an explicitly supplied local path, for example `models/fine-tuned.model`. The release contains no weights, training set or private DFT calculations. Do not upload model weights publicly until redistribution rights and provenance are established.

The model must support all selected slab elements plus H and N for the ammonia recipes. The example name does not identify a particular potential. Historical results used the supplied s8 fine-tuned potential with SHA-256:

```
3736fef6676433ab0f6bde97371eaa16838341ce91cbab15d8f2f60c0859746a
```

The prior model file was 23,803,984 bytes. Renaming an identical file does not change its identity; compiled and uncompiled artifacts can have different hashes and runtime support. Use a consistently compatible artifact for a run. Resume never silently replaces a model based on its filename.

For installation, use the official [MACE instructions](https://mace-docs.readthedocs.io/en/latest/guide/installation.html) and [PyTorch platform selector](https://pytorch.org/get-started/locally/). The Python package is `mace-torch`. Install PyTorch first for the selected accelerator. Optional cuEquivariance is not required for the CPU example.

Model loading can deserialize executable Python objects. Use only trusted model files. No production MACE calculation has been newly run as part of this release preparation; the supplied software tests do not certify model compatibility or CPU/CUDA numerical equivalence.
