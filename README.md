Code to verify Zero-One Laws on GCNs based on this paper https://arxiv.org/pdf/2301.13060.

## Setup

Clone this repository and open the directory
```
git clone https://github.com/kumarpranav30/Zero-One-GNN
cd Zero-One-GNN
```

Add this directory to the python path.
Create a conda environment and activate it.

Install dependencies
```
conda install torch torchvision torchaudio -c pytorch
conda install pyg -c pyg
pip install -r requirements.txt
```

## Simulation

Simply run simulate.py or simulate_sine.py, save results in the Results folder.
```
python Code/simulate.py
```
```
python Code/simulate_sine.py
```

