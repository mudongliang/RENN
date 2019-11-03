# RENN

Code and Data for RENN published at ASE 2019

- Code and some pretrained models of the proposed DL technique are in folder `rnns`. We also include the implementation of the baseline models used in this paper. In particular, `stats.py` is used for some necessary statistics; `ValueSet_RNN.py` contains the implementation of each model and training evaluation; `conditional_main.py` and `test.py` are used for training and testing respectively.

- Code of crash analysis is in folder `crash_analysis`. RENN takes the memory regions predicted by deep learning and leverage the alias relationship to assist reverse execution.

- Code of Intel Pin tools to record ground truth is in folder `pin_tools`.
