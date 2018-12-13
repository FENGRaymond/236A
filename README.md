# ECE236A Project
This repository contains python implementation of prototype selection proposed by Jacob Bien et al.

## ILP, relaxed LP and random rounding
An integer linear program is derived to select the minimum amount of prototype that achieves the best calssification performance. However, as the ILP is NP-hard, the solution is found by first solving a relaxed LP followed by a random rounding process. The detailed implementation can be found in classifier.train_lp()

# Reference
[1] J. Bien, R. Tibshirani, et al. Prototype selection for interpretable classification. AOAS, 2011.
