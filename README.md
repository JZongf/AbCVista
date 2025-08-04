# AbCVista

**AbCVista, a fast and accurate antibody structure prediction method capable of predicting diverse antibody conformational ensembles.**

---

## Installation Guide

### System Requirements

* 16GB RAM, 50GB storage, GPU with >=4GB VRAM

### Installation Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/JZongf/AbCVista.git
   cd AbCVista
   ```
2. Install dependencies:
   ```bash
   conda env create -f environment.yml
   conda activate AbCVista
   python setup.py install
   ```
3. Download databases:
   ```bash
   python download_database.py
   ```

---

## Usage Instructions

### Antibody Structure Prediction

```bash
python run_fold.py --fasta_dir /path/to/fasta_dir --output_dir /path/to/output_dir
```

### MSA Depth Adjustment

Use `--max_msa_clusters` and `--max_extra_msa` parameters to adjust the number of sequences used from the MSA during structure prediction:

```bash
python run_fold.py --fasta_dir /path/to/fasta_dir --output_dir /path/to/output_dir --max_msa_clusters 128 --max_extra_msa 128
```

### Prediction Quantity Adjustment

Use `--sample_count` parameter to adjust the number of predicted structures:

```bash
python run_fold.py --fasta_dir /path/to/fasta_dir --output_dir /path/to/output_dir --sample_count 40
```

This setting will generate 40 prediction results for each target.

### Antibody Conformational Ensemble Prediction

Use `--hdbscan_cluster` parameter for cluster-based antibody conformation prediction:

```bash
python run_fold.py --fasta_dir /path/to/fasta_dir --output_dir /path/to/output_dir --hdbscan_cluster
```

---

## Configuration Options

### Using DS4Sci_EvoformerAttention for Inference Acceleration

Clone the CUTLASS repository and specify its path in the CUTLASS_PATH environment variable:

```bash
git clone https://github.com/NVIDIA/cutlass
export CUTLASS_PATH=/path/to/cutlass
```

Add `--use_deepspeed_evoformer_attention` during structure prediction to enable DS4Sci_EvoformerAttention acceleration:

```bash
python run_fold.py --fasta_dir /path/to/fasta_dir --output_dir /path/to/output_dir --use_deepspeed_evoformer_attention
```

---

## Known Issues

### CUDA Version Mismatches Pytorch Version

**Issue Description:**

When executing `python setup.py install`​, you may encounter `RuntimeError:The detected CUDA version (xx.x) mismatches the version that was used to compile PyTorch (xx.x). Please make sure to use the same CUDA versions.`​

**Solution:**

Modify the `cpp_extension.py` file located at `/path/to/envs/AbCVista/lib/python3.9/site-packages/torch/utils/cpp_extension.py`

Find and **comment out** the following section:

```sh
    if cuda_ver != torch_cuda_version:
        # major/minor attributes are only available in setuptools>=49.4.0
        if getattr(cuda_ver, "major", None) is None:
            raise ValueError("setuptools>=49.4.0 is required")
        if cuda_ver.major != torch_cuda_version.major:
            raise RuntimeError(CUDA_MISMATCH_MESSAGE.format(cuda_str_version, torch.version.cuda))
        warnings.warn(CUDA_MISMATCH_WARN.format(cuda_str_version, torch.version.cuda))
```

Then execute `python setup.py install` again.

---

## License

AbCVista source code is licensed under **Apache-2.0 license**.

The **fine-tuned model parameters** provided in this project are modified from the original AlphaFold2 model parameters released by Google DeepMind.

* Original AlphaFold2 parameters license: **CC BY-NC 4.0 license**. Original copyright belongs to Google DeepMind.
* Fine-tuned model parameters released in this repository: **CC BY-NC 4.0 license**.

---

## Contact

* 📧 Email: cy_scu@yeah.net
