# RRT-VAE
Pytorch implementation for the paper: [Learning VAE-LDA Models with Rounded Reparameterization Trick](https://www.aclweb.org/anthology/2020.emnlp-main.101.pdf)

# To run the code

```
python RRT.py
```

Tunable parameters:

```
-t  # default=50 #number of topics
-e  # default=150 # number of training epochs
-b  # default=100 # training batch size
-l  # default=0.001 # learning rate
-a  # default=1.0 # Dirichlet prior parameter
-d  #default=1e10 # Delta parameter for RRT
-lam # default=0.01 # Lambda paramter for RRT
```
