# DO DEEP GENERATIVE MODELS KNOW WHAT THEY DON'T KNOW? - A reproduction study

## Introduction

In the paper by Nalisnick et al.[1], the authors challenge the common understanding of the generative model's robustness to out-of-order data distribution. Many neural networks have shown that their confidence in other distributions can be very high, mispredicting outcomes with high confidence. It was believed that this was not an issue for deep generative models. Nalisnick et al. demonstrate in their research that this phenomenon can be indeed observed in generative models as well, contrary to popular belief. 

For their investigation, Nalisnick et al. trained different generative models representing species of VAEs, autoregressive models and flow-based generative models on several datasets and tested them on unrelated datasets. They find that all these models can assign higher probability distributions to unrelated datasets that the models were not trained on than those they were trained on. This suggests that generative models would, when confronted with anomalies, predict with high confidence, not detecting such out-distribution data. The example also demonstrates that a thorough analysis of the observed phenomena is necessary for applying generative models in the industry. 

While providing evidence of such phenomena for different models, Nalisnick et al. concentrate in their analysis on the GLOW model, a model architecture from the family of flow-based generative models. More specifically, the authors create log-likelihood distributions for test sets of different datasets, generally comparing the test set belonging to the training set and a completely different set. Datasets include, but are not limited to, FashionMNIST, MNIST, CIFAR-10, SVHN, CelebA and ImageNet. Looking at these datasets, it is evident that they represent very different data domains and should not be confused. In Figure 1, however, are the log-likelihoods of the GLOW model for several datasets depicted. Indeed, it is apparent that, for these datasets, the generative model makes predictions on untrained test sets with higher confidence than on the trained datasets.

<center>
<table style='text-align:center;'>
    <tr>
        <td>
        <img src="https://user-images.githubusercontent.com/96209029/162642170-a15eee9f-fa3e-45cd-bcb2-d77ed98b2505.png" width="500"/>
        </td>
    </tr>
    <tr>
     <td>Fig 1: Likelihood estimation of GLOW on several datsets, direct comparison between trained dataset and new datsets</td>
    <tr>
</table>
</center>

As depicted, the model grossly overestimates the likelihood for some beforehand unseen datasets, failing to detect anomalies. In their quest to find an explanation for that phenomenon, Nalisnick et al. analyze the data sets serving as inputs to their generative models. More specifically, they calculate the mean and standard deviation along all channels (32x32x3 in the case of FashionMNIST, MNIST, CIFAR-10, SVHN, CelebA and ImageNet) for each set. The results can be seen in Figure 2. Based on the observation that, for instance, the SVHN dataset, a dataset misleading the generative models in testing, has a similar mean and smaller variance than the CIFAR-10, a dataset the models were trained on. The authors conclude that the misfitting datasets "sit inside" the training sets, mirroring a part of the distribution the models have high confidence in due to the number of training samples.

<center>
<table style='text-align:center;'>
    <tr>
        <td>
        <img src="https://user-images.githubusercontent.com/96209029/162839197-9e0e71f5-51db-4d04-baca-da3699112009.png" width="500"/>
        </td>
    </tr>
    <tr>
     <td>Fig 2: Analysis of several datasets for mean and standard deviation</td>
    <tr>
</table>
</center>


## Reproduction of main results

Traning of flow-based generative models is a computationally intensive task. Additionally, the code of this fundamental paper has been reproduced before by others in PyTorch. Hence, to strengthen the paper's conclusions, we have opted to reproduce their results on the outputs with a pre-implemented model while reproducing the input data analysis ourselves. Therefore, we will focus on the histograms provided in Figure 2 and the log-likelihood estimation shown in Figure 1 in graph b in the original paper, as they can be verified with relative ease while still validating the paper's overall conclusions. Of these datasets, the SVHN, Cifar-100 and Cifar-10 are accessible. Unfortunately, CelebA and ImageNet were difficult for us to obtain, making reproducing their plots impossible. The results of the datasets that were accessible can be found in Figures 3 and 4. 


<center>
<table style='text-align:center;'>
    <tr>
        <td>
        <img src="https://user-images.githubusercontent.com/61148684/162921183-0e4688af-4295-4915-8a74-592f1a72df2d.png" width="500"/>
        </td>
    </tr>
    <tr>
     <td>Fig 3: Reproduced Likelihood estimation of GLOW</td>
    <tr>
</table>
</center>

<center>
<table style='text-align:center;'>
    <tr>
        <td>
        <img src="https://user-images.githubusercontent.com/61148684/162921517-41fb3552-2d37-4e89-b862-cfe65a69f617.png" width="500"/>
        </td>
        <td>
        <img src="https://user-images.githubusercontent.com/61148684/162921530-516810e7-9424-4946-aa0b-fbb2a3b3fdf9.png" width="500"/>
        </td>
    </tr>
    <tr>
     <td colspan=2>Fig 4: Reproduced Likelihood estimation of GLOW</td>
    <tr>
</table>
</center>

As we can see, Figure 4 matches Figure 2 mentioned above in terms of distribution. The main difference is the difference in density, shown on the y-axis. This difference can be explained by using a subset of the dataset, which they have expanded. The SVHN histogram of variances differs, appearing smoother in the paper than our reproduced histogram. This can similarly be explained by using a subset of the data, resulting in a less smooth graph. Nevertheless, the general shapes of the graphs still align, which strengthens the original paper in their remarks.

If we compare Figure 1 and Figure 3, we can see that the distributions of the cifar-10-test and SVHN-test match the distributions of our reproduced likelihood estimation. We have only reproduced the test variants in our graph, which we named cifar-10 and SVHN. We have abstained from reproducing the Cifar-10 train datasets, as it would mean we would have to retrain the entire model. This would consume considerable amounts of resources and time, which we did not have at our disposal. Furthermore, the scales on both the x and y-axis are different in our reproduced graph compared to the graph presented by Nalisnick et al. For this project, we considered the datasets from the PyTorch library. So there could be a difference in the scaling factor resulting in the observed difference.

In general, the reproduced graphs uphold the claims made by Nalisnick et al. The procedures were explained in the paper, which aided the independent reproducibility efforts.

## Further inspection of data

We have shown that their research is reproducible to the extent possible to us. The authors claimed that the failure of the generative models in detecting out-of-order data could be attributed to overlapping data distribution. However, we believe that this requires further investigation. Moreover, the authors assume that the underlying distributions of different datasets are normal - a strong assumption. Hence, we also question this assumption by fitting other distributions and representations in the following.

First, we will fit different, non-gaussian distributions, such as the Cauchy distribution and F-distribution. Additionally, we will analyze the dataset in a reduced, latent space representation to investigate whether the different datasets still exhibit strong similarities when compressed. For that, we apply PCA, a traditional dimensionality reduction. Additionally, we will train an autoencoder and discriminator model (VGG-16 and ResNet-50) to obtain a learned representation.

### Is the distribution wrong? Non-normal distributions

In order to check if we get similar results in other non-normal distributions, we have selected some distributions with other parameters to check this. We have selected three distributions, each with distinct parameters, to get an overview of changing behaviour. These distributions are the Cauchy, Chi and F distributions. Cauchy has two parameters we can alter, the data localization and the data scale. We can shift the Cauchy distribution with the localization and scale parameters. These are similar to the mean and variance parameters, which we have seen in Figures 5 and 6. The Chi distribution makes use of different parameters. For example, the shape can be influenced significantly by the degrees of freedom. For the F distribution, the density is mainly influenced by two distinct features in the degrees of freedom, the denominator and the numerator. These graphs are created using a subset of the original databases, as the runtime seemed to exceed several days, which was not viable for our reproduction project. The F distribution and the Chi distributions used around 20000 samples, while the Cauchy distribution was able to run using all samples provided by the databases. 

<table style='text-align:center;'>
    <tr>
        <td>
        <img src="https://user-images.githubusercontent.com/61148684/162921786-5db2a53d-43d2-4673-bbfa-3c4885f12e85.png" width="500"/>
        </td>
        <td>
        <img src="https://user-images.githubusercontent.com/61148684/162921800-fb5ee8ae-f6a4-4832-92ab-102a4f98f198.png" width="500"/>
        </td>
    </tr>
    <tr>
     <td colspan=2>Fig 5: Data localization and scale across datasets in Cauchy distributions</td>
    <tr>
</table>
</center>

As we can see above, the difference between the mean and localization is minor. The spreads for Cifar-100 and SVHN are slightly more prominent, while the general shapes of the graphs are similar.

<center>
<table style='text-align:center;'>
    <tr>
        <td>
        <img src="https://user-images.githubusercontent.com/61148684/162921999-5be2eaa0-9c98-47de-b56f-435312f7ab0a.png" width="400"/>
        </td>
    </tr>
    <tr>
     <td>Fig 6: Degrees of freedom for Chi distribution across datasets</td>
    <tr>
</table>
</center>

With the Chi distribution, we see that degrees of freedom for Cifar-10 and Cifar-100 are centred around 1.8. The SVHN is centred around 3 degrees of freedom. The spread of SVHN is also lower than the spread of Cifar-10 and Cifar-100, which is similar to what we have seen in the other distributions. Also noteworthy is that the Cifar-10 and Cifar-100 are skewed to the right, while the SVHN is more centred. 

<center>
<table style='text-align:center;'>
    <tr>
        <td>
        <img src="https://user-images.githubusercontent.com/61148684/162922073-96c3d5c2-b84a-4098-851d-18ddb5da13a4.png" width="400"/>
        </td>
    </tr>
    <tr>
     <td>Fig 7: Degrees of freedom denominator and numerator of F distributions across datasets</td>
    <tr>
</table>
</center>

While the Chi distribution was also run with a limited amount of samples, the F distribution seems to have a more rough plot, where the graph is not as smooth as observed with the Chi distribution. Although the spread within the numerator is limited, each dataset seems to have a similar F distribution. With the denominator, the spread is more apparent. While this is not skewed, the variance of the Cifar-10 and Cifar-100 distributions is higher than observed with the numerator.


### analysis in reduced space representations
Aside from comparing the results, we can also look at the reduced space representation. We will check this with 3 representations: Principal component analysis, discriminant representation, and autoencoder representation.  

#### Principal component analysis (PCA)
The principal component analysis allows us to summarize datasets and indices that are more informing than others. These indices are called principal components. This allows us to effectively condense our information, making it easier to handle. We have concatenated the training and test data of the three datasets we have operated on. Using the function pca_lowrank(), we perform our principal component analysis. The tensors then get reshaped based on the U, S and V matrices obtained from pca_lowrank(), which together form a decomposition matrix (size = 3x32x6) approximating the original dataset matrix (size = 3x32x32). We then get the statistics from the decomposed matrix. With the returned mean and standard deviation, we can then plot the variance across datasets, as shown in the Figures below:   

<center>
<table style='text-align:center;'>
    <tr>
        <td>
        <img src="https://user-images.githubusercontent.com/61148684/163387423-355df5b7-cd56-42cd-abec-5fd1dcababd2.png" width="500"/>
        </td>
        <td>
        <img src="https://user-images.githubusercontent.com/61148684/163387439-023a451a-142d-4a1b-af90-12880c6805b2.png" width="500"/>
        </td>
    </tr>
    <tr>
     <td colspan=2>Fig 8: Mean and Standard Deviation after PCA</td>
    <tr>
</table>
</center>

As we can see, the shapes of the scaled mean after our PCA are similar across all three datasets. They are also similar in terms of density. We see a significant similarity between the Cifar-10 and Cifar-100 datasets for the standard deviation. Both the shape and density match up for these datasets. The SVHN dataset does, however, differ in the graph. It has a distinctly different shape, not following the right-tailed distribution of both Cifar variants.

#### Discriminant representation
We have also looked at discriminant representation. In order to do this, we have taken the pretrained vgg16 and ResNet-50 and applied transfer learning to them. We then used these on our 3 datasets: SVHN, Cifar-10 and Cifar-100. In total, we get 6 latent representations to compare. We then compare the mean and standard deviation for the latent representations. The results of this can be found below:

<center>
<table style='text-align:center;'>
    <tr>
        <td>
        <img src="https://user-images.githubusercontent.com/61148684/163449856-37dedec2-e3bd-4486-8c94-3cd05bd52390.png" width="500"/>
        </td>
        <td>
        <img src="https://user-images.githubusercontent.com/61148684/163449872-0d9232b6-e61b-4d0a-a59d-411f3f8bad82.png" width="500"/>
        </td>
    </tr>
    <tr>
     <td colspan=2>Fig 9: Mean and Standard Deviation of latent representation generated by VGG-16</td>
    <tr>
</table>
</center>

<center>
<table style='text-align:center;'>
    <tr>
        <td>
        <img src="https://user-images.githubusercontent.com/61148684/163449842-43a48dca-88ac-4291-9aed-18b73a3b7d2f.png" width="500"/>
        </td>
        <td>
        <img src="https://user-images.githubusercontent.com/61148684/163449865-d4464bae-e090-4cbb-b948-04ab28aa502f.png" width="500"/>
        </td>
    </tr>
    <tr>
     <td colspan=2>Fig 10: Mean and Standard Deviation of latent representation generated by ResNet-50</td>
    <tr>
</table>
</center>

The overlap between the distributions of the mean and standard deviation of the three datasets observed in the graphs in the original paper is significantly reduced in the latent representations for both the discriminators. Thus it is difficult to support the original claim made by the authors that the distribution of one dataset "sit inside" other datasets. Another interesting observation is that the original graphs had similarities in the distributions of the Cifar-10 and Cifar-100 datasets, whereas the similarity is observed between Cifar-10 and SVHN datasets with the latent representation. However, the impact of pre-training should be investigated further to validate this observation.

#### Autoencoder representation
                                                                                                                                    
<center>
<table style='text-align:center;'>
    <tr>
        <td>
        <img src="https://user-images.githubusercontent.com/13098653/163450979-c515ccab-798b-4db8-b629-dc5ccf6f09d4.png" width="500"/>
        </td>
        <td>
        <img src="https://user-images.githubusercontent.com/13098653/163451016-74a21109-5789-42cf-afba-b9b58a4f8cb1.png" width="500"/>
        </td>
    </tr>
    <tr>
     <td colspan=2>Fig 11: Mean and Standard Deviation of latent representation generated by Autoencoder</td>
    <tr>
</table>
</center>

The distributions of the latent representations are entirely overlapping. This holds the author's claims but contradicts the observations made in the latent space of the discriminator models.

## Concluding remarks

The conclusions made by the authors are upheld in all the results observed throughout this study, except for the latent representation space of the pre-trained discriminators. Moreover, the effect of pre-training is yet to be ruled out for the observations. 

Further studies have looked into the plausible reasons for the failure of generative models in detecting out-of-order detection. For example, Kirichenko et al. [2] found that the models learn the local pixel correlations and generic image-to-latent space transformations. Thus it is independent of the target image dataset. The paper further continued to provide some fixes for this scenario. This might be an interesting paper to reproduce following this investigation.

## Contributions

All the group members read the research paper and then worked on replicating the Figures. The entire reproduction project focuses on the 3 datasets- Cifar10, Cifar100 and SVHN.
Post this initial reproduction; the individual contributions are as below:
Maximilian and Nils worked upon non-normal Cauchy, Chi and F distributions to inspect data. They further summarized the paper and the reproduction results for this GitHub blogpost.
Sayak and Vishruty worked upon inspection of the latent representation of the data with the help of Autoencoder, Principal Component Analysis(PCA), Discriminator (VGG16 and ResNet-50). Further helped Maximilian and Nils in the summarization of the reproduction results.

## References

[1] Nalisnick, E., Matsukawa, A., Teh, Y.W., Gorur, D. and Lakshminarayanan, B., 2018. Do deep generative models know what they don't know?. arXiv preprint arXiv:1810.09136.

[2] Kirichenko, P., Izmailov, P. and Wilson, A.G., 2020. Why normalizing flows fail to detect out-of-distribution data. Advances in neural information processing systems, 33, pp.20578-20589.
