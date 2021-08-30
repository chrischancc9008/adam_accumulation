# Introduction

An implementation of Adam accumulation for tensorflow >= 2.5.0.

Purpose: 
It's proved by empirical results that NLP models will be benefited from large batch size, e.g., Roberta, SimCSE etc.
However, a large batch size will exceed the GPU memory. 
To solve the issue, gradient is estimated by accumulation a few batches.

# Reference

https://github.com/bojone/bert4keras/blob/master/bert4keras/optimizers.py

https://github.com/CyberZHG/keras-gradient-accumulation

https://github.com/keras-team/keras/issues/3556

# TODO

1. check the source of the difference. [results are aligned in cpu but NOT in gpu]

2. speed up the process. [Running time is much longer than the one without gradient accumulation but probably due to low level optimization in tensorflow ]

3. implement sparse logic.