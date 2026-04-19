# Political Opinion Dynamics

This repository contains the implementation used in the thesis:

**"Modeling Opinion Dynamics in Online Political Discussions"**

## Overview

The project analyzes political discussion data using:

- **RoBERTa** for semantic embeddings  
- **Graph Attention Networks (GAT)** for structural modeling  
- **Temporal Graph Networks (TGN)** for temporal interaction modeling  

The focus is on:
- Link prediction  
- Clustering and embedding analysis  
- Polarization and echo chamber detection  
- Temporal opinion shift  

---

## Repository Structure
src/
├── train_tgn.py
├── train_gat.py
├── train_opinion_dynamics.py
├── test_link_prediction.py
├── compare_models.py
├── opinion_shift_analysis.py
├── graph_analysis.py
├── visualize_embeddings.py
└── ...
---

## Key Scripts

- `train_tgn.py` — Train Temporal Graph Network  
- `train_gat.py` — Train Graph Attention Network  
- `generate_embeddings.py` — Generate RoBERTa embeddings  
- `test_link_prediction.py` — Evaluate link prediction (AUC)  
- `compare_models.py` — Clustering and distance metrics  
- `opinion_shift_analysis.py` — Temporal embedding analysis  

---

## Notes

- Only source code is included in this repository.  
- Data files, trained models, and logs are excluded due to size constraints.  
- The implementation is designed for experimental analysis rather than production use.  

---

## Author

Darshana Srivathsan  
IISER Bhopal