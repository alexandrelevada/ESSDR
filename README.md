**Semi-Supervised Metric Learning with Information-Theoretic Distances: A Dimensionality Reduction Based Approach**

Alaor Cervati Neto, Alexandre L. M. Levada 

Since developing compact and meaningful data representations for classification and visualization falls under the purview of both distance metric learning and nonlinear dimensionality reduction, they are inextricably linked. In this paper, we propose a graph-based generalization of the Semi-Supervised Dimensionality Reduction (SSDR) algorithm, which computes the similarity between local multivariate Gaussian distributions along the K Nearest Neighbors (KNN) graph built from the samples in the input high-dimensional space using stochastic distances (Kullback-Leibler, Bhattacharyya, and Cauchy-Schwarz divergences). In conclusion, the suggested method comes in two different forms: one that utilizes only 10\% of the labeled samples and another that additionally use a clustering technique (Gaussian Mixture Models) to estimate the labels of the least spanning tree of the KNN network. In comparison to the standard SSDR algorithm, experimental results using several real datasets demonstrate that the proposed method is able to increase the classification accuracy of a number of supervised classifiers as well as the quality of the clusters that are produced (Silhouette Coefficients). This makes it a competitive alternative for pattern classification issues.

The file ESSDR.ipynb has a link to run the script online at Google Colab in order to reproduce the results of the paper.
