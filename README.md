# GCN from scratch

A two-layer Graph Convolutional Network for node classification. The GCN layer is
implemented with ordinary PyTorch (`index_add_`); no `torch_geometric.nn` layer is
used. OGB is used only to download and load the graph.

## Install

```bash
python3 -m pip install -r requirements.txt
```

## Quick test

```bash
python3 gcn.py --synthetic --epochs 50
```

## OGB experiment

```bash
python3 gcn.py --dataset ogbn-arxiv --epochs 100
```

Use `--max-nodes 20000` for a smaller learning experiment. A subset uses a random
60/20/20 split, so its result is not an official OGB benchmark number.
