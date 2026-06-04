"""Interpretability experiments for the CB-VLAM-AV concept bottleneck.

First module: test-time concept intervention (Koh et al. 2020; Shin et al. 2023)
— replace predicted concept activations with ground truth, most-important-first,
and measure the change in trajectory L2, with the residual path on vs off.
"""
