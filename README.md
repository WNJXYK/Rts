# README

This is a demo for RTS algorithm.

### Requirements

Install conda environment via "RTS.yaml".

### Usage

You can run the following script:
```bash
python run.py --noise-type sym --noise-rate 0.2 --method Proposal --dataset ECG5000 --seed 0 --gpu 0
```
to evaluate our proposed RTS approach on ECG5000 data set with 20% symmetrical label noise.

You can also run `script.py` for all data settings directly as follows.
```bash
python script.py
```

The results will be printed and stored in `./Results`.

