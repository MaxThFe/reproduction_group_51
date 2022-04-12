# DO DEEP GENERATIVE MODELS KNOW WHAT THEY DON’T KNOW? - A reproduction study

## Introduction

In their paper, Nalisnick et al. challenge the common understanding of generative model's robustness to different distribution data. Many types of neural networks have shown that their confidence on other distributions can be very high, mispredicting outcomes with a high confidence. It was believed that this was not an issue for deep generative models. Nalisnick et al. demonstrate in their research that this phenomena can be indeed observed in generative models as well, contrary to popular belief. 

For their investigation, Nalisnick et al. train different generative models, representing species of VAEs, autoregressive models and flow-based generative models on several datasets, testing them on unrelated datasets. They find that all these models can assign higher probabiltiy distributions to unrealted datasets and those they were trained on. This suggest that generative models would, for instance, when confronted with anomalies, predict with high confidence, not detecting such out-distribution data. The example also demonstrates that a thorough analysis of the observed phenomena is necesary for application of generative models in industry. 

While providing evidence of such phenomena for different models, Nalisnick et al. concentrate in their analysis on the GLOW model, a model architecture from the family of flow-based generative models. More specifically, the authors create log-likelihood distributions for test-sets of different datasets, normally in a direct comparison of the test set belonging to the training set and a completely different set. Datasets they consider include, but are not limited to, FashionMNIST, MNIST, CIFAR-10, SVHN, CelebA and ImageNet. Looking at these datasets, it is obvious that they represent very different data domains and hence should not be confused. In Figure 1 however are the log-likelihoods of the GLOW model for several datasets depicted. It is apparent that indeed for these datasets, the generative model makes predictions on untrained testsets with a higher confidence than on the trained datasets.

![Likelihood estimation of GLOW on several datsets, direct comparison between trained dataset and new datsets](https://user-images.githubusercontent.com/96209029/162642170-a15eee9f-fa3e-45cd-bcb2-d77ed98b2505.png)

As depicted, the model grossly overestimates the likelihood for some beforehand unseen datasets, failing to detect anomalies. In their quest to find an explanation to that phenomena, Nalisnick et al. analyze the input data to their generative models. More specifically, they calculate the mean and standard deviation along all channels (32x32x3 in the case of FashionMNIST, MNIST, CIFAR-10, SVHN, CelebA and ImageNet). The results can be seen in Figure 2. Based on the observation, that for instance SVHN dataset, a dataset misleading the generative models in testing, has a similiar mean and smaller variance than the SVHN, a datset the models were trained on, the authors conclude that the missfitting datasets simply "sit inside" of the training sets, mirroring a part of the distribution the models have a high confidence on due to the number if training samples.

![Analysis of several datasets for mean and standard deviation](https://user-images.githubusercontent.com/96209029/162839197-9e0e71f5-51db-4d04-baca-da3699112009.png)

## Reproduction of main results

In order to strengthen the conclusions of the paper, we have opted to reproduce their results. We will focus on the histograms provided in Figure 2, as well as the log likelyhood estimation shown in Figure 1 in graph b, as they can be verified with relative ease while still validating the overall conclusions of the paper. Of these datasets, the SVHN, Cifar-100 and Cifar-10 are accessible to us. CelebA and ImageNet were difficult for us to obtain, making reproducing their plots impossible. The results of the datasets that were accessible can be found in Figure 3 and 4.  

TODO: add own results figure for reproducibility.

As we can see, Figure 4 matches the aforementioned Figure 2 in terms of distribution. The main difference is the difference in density, shown on the y-axis. This difference can be explained by using a subset of the dataset, which they seem to have expanded. The SVHN histogram of variances differs, appearing smoother in the paper compared to our reproduced histogram. This can similarly be explained by using a subset of the data, which can result in a less smooth graph The general shapes of the graphs still align, which strengthens the original paper in their remarks.

## Further inspection of data

### Is the distribution wrong? Non-normal distributions

### Analysis in reduced space representations

#### Principal component analysis (PCA)

#### Discriminant representation

#### Autoencoder representation

## Concluding remarks
