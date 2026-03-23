<div align="center">

# MARVEL-FX3D

### Text-to-3D Textured Mesh Generation

[![Project Page](https://img.shields.io/badge/🌐_Project-Page-orange)](https://sankalpsinha-cmos.github.io/MARVEL/)
[![Paper](https://img.shields.io/badge/📄_CVPR_2025-Paper-brightgreen)](https://openaccess.thecvf.com/content/CVPR2025/papers/Sinha_MARVEL-40M_Multi-Level_Visual_Elaboration_for_High-Fidelity_Text-to-3D_Content_Creation_CVPR_2025_paper.pdf)
[![arXiv](https://img.shields.io/badge/arXiv-2411.17945-b31b1b)](https://arxiv.org/abs/2411.17945)
[![Dataset](https://img.shields.io/badge/🤗_MARVEL--40M+-Dataset-red)](https://huggingface.co/datasets/sankalpsinha77/MARVEL-40M)
[![Explorer](https://img.shields.io/badge/🔍_Dataset-Explorer-blue)](https://sadilkhan.github.io/Marvel-Explorer/)

**[Sankalp Sinha]()\* · [Mohammad Sadil Khan]()\* · [Muhammad Usama]() · [Shino Sam]() · [Didier Stricker]() · [Sk Aziz Ali]() · [Muhammad Zeshan Afzal]()**

*\* Equal contribution*

<div align="center" style="background-color: white; display: inline-block; padding: 12px; border-radius: 8px;">
  <img src="https://openaccess.thecvf.com/img/cvpr2025_logo.png" alt="CVPR 2025"/>
</div>

</div>

---

## Overview

> **MARVEL-FX3D** is the generation component of the [MARVEL-40M+](https://arxiv.org/abs/2411.17945) paper, which introduces a dataset of 40M multi-level text annotations for 8.9M+ 3D assets. The dataset and annotation pipeline are described in the [main paper](https://openaccess.thecvf.com/content/CVPR2025/papers/Sinha_MARVEL-40M_Multi-Level_Visual_Elaboration_for_High-Fidelity_Text-to-3D_Content_Creation_CVPR_2025_paper.pdf).

---

## 📦 MARVEL-40M+ Dataset

MARVEL-FX3D is trained on **MARVEL-40M+**, the largest 3D captioning dataset to date.

| Property | Value |
|---|---|
| Total Annotations | **40 million** |
| 3D Assets | **8.9 million+** |
| Source Datasets | 7 major 3D repositories |
| Annotation Levels | Detailed (150–200 words) → Tags (10–20 words) |

The multi-stage annotation pipeline combines open-source multi-view VLMs and LLMs with human metadata from source datasets, reducing hallucinations and improving domain-specific accuracy.

🔗 **Dataset on Hugging Face**: [MARVEL-40M+](https://huggingface.co/datasets/sankalpsinha77/MARVEL-40M)

---

## 🛠️ Installation

```bash
# Clone the repository
git clone https://github.com/SadilKhan/MARVEL-FX3D.git
cd MARVEL-FX3D

# Create a conda environment (recommended)
conda create -n marvel python=3.10
conda activate marvel

# Install dependencies
pip install -r requirements.txt
```

> **Requirements**: Python ≥ 3.10, PyTorch ≥ 2.0, CUDA ≥ 11.8 (recommended)

---

## 🚀 Quick Start

```python
# Generate a 3D textured mesh from a text prompt
python generate.py --prompt "A Harley Davidson motorcycle with a black leather seat and dual exhaust pipes"
```

The output mesh will be saved to `output/` by default.
---

## 📜 Citation

If you find MARVEL-FX3D or MARVEL-40M+ useful in your research, please cite:

```bibtex
@InProceedings{Sinha_2025_CVPR,
    author    = {Sinha, Sankalp and Khan, Mohammad Sadil and Usama, Muhammad and Sam, Shino and Stricker, Didier and Ali, Sk Aziz and Afzal, Muhammad Zeshan},
    title     = {MARVEL-40M+: Multi-Level Visual Elaboration for High-Fidelity Text-to-3D Content Creation},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2025},
}
```

---

## 📄 License

This project is released under the [MIT License](LICENSE).

---

<div align="center">
  <sub>
    If you have questions, feel free to open an issue or reach out via the 
    <a href="https://sankalpsinha-cmos.github.io/MARVEL/">project page</a>.
  </sub>
</div>
