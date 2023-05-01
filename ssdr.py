#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Semi-Supervised Metric Learning with Information-Theoretic Distances: A Dimensionality Reduction Based Approach

Alaor Cervati Neto, Alexandre L. M. Levada

"""

# Imports
import sys
import time
import warnings
import sklearn.datasets as skdata
import matplotlib.pyplot as plt
import numpy as np
import itertools
import random
import networkx as nx
from numpy import log
from numpy import trace
from numpy import dot
from scipy import stats
from numpy.linalg import det
from scipy.linalg import eigh
from numpy.linalg import inv
from numpy.linalg import cond
from numpy import eye
from sklearn import preprocessing
from sklearn import metrics
import sklearn.neighbors as sknn
from sklearn.mixture import GaussianMixture
from sklearn.manifold import Isomap
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.manifold import SpectralEmbedding
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.ensemble import RandomForestClassifier
from networkx.algorithms.centrality import edge_betweenness_centrality

# To avoid unnecessary warning messages
warnings.simplefilter(action='ignore')

# PCA implementation
def myPCA(dados, d):
    # Eigenvalues and eigenvectors of the covariance matrix
    v, w = np.linalg.eig(np.cov(dados.T))
    # Sort the eigenvalues
    ordem = v.argsort()
    # Select the d eigenvectors associated to the d largest eigenvalues
    maiores_autovetores = w[:, ordem[-d:]]
    # Projection matrix
    Wpca = maiores_autovetores
    # Linear projection into the 2D subspace
    novos_dados = np.dot(Wpca.T, dados.T)
    return novos_dados

# Bhattacharyya divergence
def Bhattacharyya(mu1, mu2, cov1, cov2):
    m = len(mu1)
    Sigma = (cov1 + cov2)/2
    # If covariance matrices are ill-conditioned
    if np.linalg.cond(cov1) > 1/sys.float_info.epsilon:
        cov1 = cov1 + np.diag(0.001*np.ones(m))
    if np.linalg.cond(cov2) > 1/sys.float_info.epsilon:
        cov2 = cov2 + np.diag(0.001*np.ones(m))
    if np.linalg.cond(Sigma) > 1/sys.float_info.epsilon:
        Sigma = Sigma + np.diag(0.001*np.ones(m))
    dM = (1/8)*(mu1-mu2).T.dot(inv(Sigma)).dot(mu1-mu2)
    dD = 0.5*log(det(Sigma)/np.sqrt(det(cov1)*det(cov2)))
    dB = dM + dD
    return dB

# KL-divergence
def divergenciaKL(mu1, mu2, cov1, cov2):
    m = len(mu1)
    # If covariance matrices are ill-conditioned
    if np.linalg.cond(cov1) > 1/sys.float_info.epsilon:
        cov1 = cov1 + np.diag(0.001*np.ones(m))
    if np.linalg.cond(cov2) > 1/sys.float_info.epsilon:
        cov2 = cov2 + np.diag(0.001*np.ones(m))
    dM1 = 0.5*(mu2-mu1).T.dot(inv(cov2)).dot(mu2-mu1)
    dM2 = 0.5*(mu1-mu2).T.dot(inv(cov1)).dot(mu1-mu2)
    dTr = 0.5*trace(dot(inv(cov1), cov2) + dot(inv(cov2), cov1))
    dKL = 0.5*(dTr + dM1 + dM2 - m)
    return dKL

# Cauchy-Schwarz divergence
def CauchySchwarz(mu1, mu2, cov1, cov2):
    m = len(mu1)
    # If covariance matrices are ill-conditioned
    if np.linalg.cond(cov1) > 1/sys.float_info.epsilon:
        cov1 = cov1 + np.diag(0.001*np.ones(m))
    if np.linalg.cond(cov2) > 1/sys.float_info.epsilon:
        cov2 = cov2 + np.diag(0.001*np.ones(m))
    Sigma_inv = inv(cov1) + inv(cov2)
    if np.linalg.cond(Sigma_inv) > 1/sys.float_info.epsilon:
        Sigma_inv = Sigma_inv + np.diag(0.001*np.ones(m))
    T1 = (1/4)*log(det(cov1/2)) + 0.5*(mu1).T.dot(inv(cov1)).dot(mu1)
    T2 = (1/4)*log(det(cov2/2)) + 0.5*(mu2).T.dot(inv(cov2)).dot(mu2)
    T3 = 0.5*(log(det(inv(cov1) + inv(cov2))))
    T4 = 0.5*dot((dot(inv(cov1), mu1) + dot(inv(cov2), mu2)).T, dot(inv(Sigma_inv), (dot(inv(cov1), mu1) + dot(inv(cov2), mu2))))
    dCS = T1 + T2 + T3 - T4    
    return dCS

# Semi-supervised dimensionality reduction (regular)
def SSDR(dados, target, perc, alpha, beta, d):
    # Number of samples
    n = dados.shape[0]
    # Number of features
    m = dados.shape[1]
    # Pairwise constraints
    x = list(range(0, n))
    pares = list(itertools.combinations(x, 2))
    # Select percentage perc of the total pairs
    num = round(perc*n)    # 0 < perc < 1
    L = random.sample(pares, num)
    # Number of elements in C and M 
    NC = 0
    NM = 0
    for i in range(len(L)):
        if target[L[i][0]] == target[L[i][1]]:
            NM += 1  
        else:
            NC += 1
    # Build the complete graph
    S = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if (i, j) in L:
                if target[i] == target[j]:  # Must-Link constraint
                    S[i, j] = (1/n**2) - beta/NM 
                else:                       # Cannot-link constraint
                    S[i, j] = (1/n**2) + alpha/NC
            else:
                S[i, j] = (1/n**2) 
    # Degree matrix D and Laplacian L
    D = np.diag(S.sum(1))   
    L = D - S
    # Final matrix
    X = dados.T
    M = np.dot(np.dot(X, L), X.T)
    lambdas, alphas = eigh(M)  
    ordem = lambdas.argsort()
    # Select the d eigenvectors associated to the d largest eigenvalues
    maiores_autovetores = alphas[:, ordem[-d:]]
    # Projection matrix
    Wssdr = maiores_autovetores 
    # Project data
    output = np.dot(Wssdr.T, X)
    return output

# Entropic SSDR (proposed method - variation 1)
def Entropic_SSDR(dados, target, dist, k, perc, alpha, beta, d):
    # Number of samples
    n = dados.shape[0]
    # Number of features
    m = dados.shape[1]
    # Define the KNN graph
    knnGraph = sknn.kneighbors_graph(dados, n_neighbors=k, mode='connectivity')
    W = knnGraph.toarray()
    # Generate the pairwise constraints
    x = list(range(0, n))
    pares = list(itertools.combinations(x, 2))
    # Select percentage perc of the total pairs
    num = round(perc*n)    # 0 < perc < 1
    L = random.sample(pares, num)
    # Number of elements in C and M 
    NC = 0
    NM = 0
    for i in range(len(L)):
        if target[L[i][0]] == target[L[i][1]]:
            NM += 1  
        else:
            NC += 1
    # Build the complete graph
    if NC == 0:
        NC += 1
    if NM == 0:
        NM += 1
    S = np.zeros((n, n))
    for i in range(n):
        for j in range(n):            
            if (i, j) in L:
                # Compute the divergence between patches
                # Extract patch 1
                vizinhos = W[i, :]
                indices = vizinhos.nonzero()[0]
                if len(indices) < 2:    # treat isolated points
                    media_i = X[i, :]
                    matriz_covariancias_i = np.eye(dados.shape[1])
                else:
                    amostras = dados[indices]
                    media_i = amostras.mean(0)
                    matriz_covariancias_i = np.cov(amostras.T)
                # Extract patch 2    
                vizinhos = W[j, :]
                indices = vizinhos.nonzero()[0]
                if len(indices) < 2:    # treat isolated points
                    media_j = X[j, :]
                    matriz_covariancias_j = np.eye(dados.shape[1])
                else:
                    amostras = dados[indices]
                    media_j = amostras.mean(0)
                    matriz_covariancias_j = np.cov(amostras.T)
                # Select the stochastic divergence to compute
                if dist == 'KL':
                    DKL_ij = divergenciaKL(media_i, media_j, matriz_covariancias_i, matriz_covariancias_j)
                    DKL_ji = divergenciaKL(media_j, media_i, matriz_covariancias_j, matriz_covariancias_i)
                    distance = 0.5*(DKL_ij + DKL_ji)
                elif dist == 'BHAT':
                    distance = Bhattacharyya(media_i, media_j, matriz_covariancias_i, matriz_covariancias_j)
                elif dist == 'CS':
                    distance = CauchySchwarz(media_i, media_j, matriz_covariancias_i, matriz_covariancias_j)
                # If distance is infinite, define a upper bound
                if np.isinf(distance):
                    distance = 100
                # If distance is NaN, replace by a small value
                elif np.isnan(distance):
                    distance = 0.001
                # Compute the values of the matrix S
                if target[i] == target[j]:  # Must-Link constraint
                    S[i, j] = (1/n**2) - (1/NM)*distance*beta
                else:   # Cannot-link constraint
                    S[i, j] = (1/n**2) + (1/NC)*distance*alpha
            else:
                S[i, j] = (1/n**2)
    # Degree matrix D and Laplacian L
    D = np.diag(S.sum(1))   
    L = D - S
    # Final matrix
    X = dados.T
    M = np.dot(np.dot(X, L), X.T)
    lambdas, alphas = eigh(M)  
    ordem = lambdas.argsort()
    # Select the d eigenvectors associated to the d largest eigenvalues
    maiores_autovetores = alphas[:, ordem[-d:]]
    # Projection matrix
    Wssdr = maiores_autovetores 
    # Project data
    output = np.dot(Wssdr.T, X)
    return output

# MST-ESSDR: Entropic SSDR with MST for pairwise constraints selection
def Entropic_SSDR_MST(dados, target, dist, k, perc, alpha, beta, d):
    # Number of samples
    n = dados.shape[0]
    # Number of features
    m = dados.shape[1]
    # Number of classes
    c = len(np.unique(target))
    # Labels estimation with GMM model
    gmm_labels = GaussianMixture(n_components=c, random_state=0).fit_predict(dados)
    # Build the KNN graph
    knnGraph = sknn.kneighbors_graph(dados, n_neighbors=k, mode='distance')
    W = knnGraph.toarray()
    # Generate pairwise constraints
    x = list(range(0, n))
    pares = list(itertools.combinations(x, 2))
    # Select percentage perc of the total pairs
    num = round(perc*n)    # 0 < perc < 1
    L = random.sample(pares, num)
    # Pairwise constraints using MST
    G = nx.from_numpy_array(W)
    W_mst = nx.minimum_spanning_tree(G)
    mst_edges = W_mst.edges()
    # Find the number of elements in C and M
    NC = 0
    NM = 0
    for i in range(len(L)):
        if target[L[i][0]] == target[L[i][1]]:
            NM += 1  
        else:
            NC += 1
    if NC == 0:
        NC = 1
    if NM == 0:
        NM = 1
    # Build the complete graph (S)
    S = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if (i, j) in L:   # use the real labels
                # Extract the patch 1
                vizinhos = W[i, :]
                indices = vizinhos.nonzero()[0]
                if len(indices) < 2:   # treat isolated points
                    media_i = dados[i, :]
                    matriz_covariancias_i = np.eye(dados.shape[1])
                else:
                    amostras = dados[indices]
                    media_i = amostras.mean(0)
                    matriz_covariancias_i = np.cov(amostras.T)
                # Extract the patch 2
                vizinhos = W[j, :]
                indices = vizinhos.nonzero()[0]
                if len(indices) < 2:   # treat isolated points
                    media_j = dados[j, :]
                    matriz_covariancias_j = np.eye(dados.shape[1])
                else:
                    amostras = dados[indices]
                    media_j = amostras.mean(0)
                    matriz_covariancias_j = np.cov(amostras.T)
                # Select the stochastic divergence
                if dist == 'KL':
                    DKL_ij = divergenciaKL(media_i, media_j, matriz_covariancias_i, matriz_covariancias_j)
                    DKL_ji = divergenciaKL(media_j, media_i, matriz_covariancias_j, matriz_covariancias_i)
                    distance = 0.5*(DKL_ij + DKL_ji)
                elif dist == 'BHAT':
                    distance = Bhattacharyya(media_i, media_j, matriz_covariancias_i, matriz_covariancias_j)
                elif dist == 'CS':
                    distance = CauchySchwarz(media_i, media_j, matriz_covariancias_i, matriz_covariancias_j)
                # If distance is infinite, define a upper bound
                if np.isinf(distance):
                    distance = 100
                # If distance is NaN, replace by a small value
                elif np.isnan(distance):
                    distance = 0.001
                # Compute the values of the matrix S
                if target[i] == target[j]:    # Must-Link constraint
                    S[i, j] = (1/n**2) - (1/NM)*distance*beta
                else:                         # Cannot-link constraint
                    S[i, j] = (1/n**2) + (1/NC)*distance*alpha
            elif (i, j) in mst_edges:   # Use the estimated labels if we are in a MST edge
                # Extract the patch 1
                vizinhos = W[i, :]
                indices = vizinhos.nonzero()[0]
                if len(indices) < 2:   # treat isolated points
                    media_i = dados[i, :]
                    matriz_covariancias_i = np.eye(dados.shape[1])
                else:
                    amostras = dados[indices]
                    media_i = amostras.mean(0)
                    matriz_covariancias_i = np.cov(amostras.T)
                # Extract the patch 2
                vizinhos = W[j, :]
                indices = vizinhos.nonzero()[0]
                if len(indices) < 2:   # treat isolated points
                    media_j = dados[j, :]
                    matriz_covariancias_j = np.eye(dados.shape[1])
                else:
                    amostras = dados[indices]
                    media_j = amostras.mean(0)
                    matriz_covariancias_j = np.cov(amostras.T)
                # Select the stochastic divergence
                if dist == 'KL':
                    DKL_ij = divergenciaKL(media_i, media_j, matriz_covariancias_i, matriz_covariancias_j)
                    DKL_ji = divergenciaKL(media_j, media_i, matriz_covariancias_j, matriz_covariancias_i)
                    distance = 0.5*(DKL_ij + DKL_ji)
                elif dist == 'BHAT':
                    distance = Bhattacharyya(media_i, media_j, matriz_covariancias_i, matriz_covariancias_j)
                elif dist == 'CS':
                    distance = CauchySchwarz(media_i, media_j, matriz_covariancias_i, matriz_covariancias_j)
                # If distance is infinite, define a upper bound
                if np.isinf(distance):
                    distance = 100
                # If distance is NaN, replace by a small value
                elif np.isnan(distance):
                    distance = 0.001
                # Compute the values of the matrix S
                if gmm_labels[i] == gmm_labels[j]:    # Must-Link constraint
                    S[i, j] = (1/n**2) - (1/NM)*distance*beta
                else:                                 # Cannot-link constraint
                    S[i, j] = (1/n**2) + (1/NC)*distance*alpha
            else:
                S[i, j] = (1/n**2)
    # Degree matrix D and Laplacian L
    D = np.diag(S.sum(1))   
    L = D - S
    # Final matrix
    X = dados.T
    M = np.dot(np.dot(X, L), X.T)
    lambdas, alphas = eigh(M)  
    ordem = lambdas.argsort()
    # Select the d eigenvectors associated to the d largest eigenvalues
    maiores_autovetores = alphas[:, ordem[-d:]]
    # Projection matrix
    Wssdr = maiores_autovetores 
    # Projeta data
    output = np.dot(Wssdr.T, X)
    return output

'''
 Computes the Silhouette coefficient and the supervised classification
 accuracies for several classifiers: KNN, SVM, NB, DT, MPL, GPC and RFC
 dados: learned representation (output of a dimens. reduction - DR)
 target: ground-truth (data labels)
 '''
def Classification(dados, target, method):
    # print()
    # print('Supervised classification for %s features' %(method))
    # print()
    
    lista = []

    # 50% for training and 50% for testing
    X_train, X_test, y_train, y_test = train_test_split(dados.real.T, target, test_size=.5, random_state=42)

    # KNN
    neigh = KNeighborsClassifier(n_neighbors=7)
    neigh.fit(X_train, y_train) 
    acc = neigh.score(X_test, y_test)
    lista.append(acc)

    # SMV
    svm = SVC(gamma='auto')
    svm.fit(X_train, y_train) 
    acc = svm.score(X_test, y_test)
    lista.append(acc)

    # Naive Bayes
    nb = GaussianNB()
    nb.fit(X_train, y_train)
    acc = nb.score(X_test, y_test)
    lista.append(acc)

    # Decision Tree
    dt = DecisionTreeClassifier(random_state=0)
    dt.fit(X_train, y_train)
    acc = dt.score(X_test, y_test)
    lista.append(acc)

    # MPL classifier
    mpl = MLPClassifier(hidden_layer_sizes=(100,), activation='logistic', max_iter=5000)
    mpl.fit(X_train, y_train)
    acc = mpl.score(X_test, y_test)
    lista.append(acc)

    # Gaussian Process
    gpc = GaussianProcessClassifier()
    gpc.fit(X_train, y_train)
    acc = gpc.score(X_test, y_test)
    lista.append(acc)

    # Random Forest Classifier
    rfc = RandomForestClassifier()
    rfc.fit(X_train, y_train)
    acc = rfc.score(X_test, y_test)
    lista.append(acc)

    # Computes the Silhoutte coefficient
    sc = metrics.silhouette_score(dados.real.T, target, metric='euclidean')
    ch = metrics.calinski_harabasz_score(dados.real.T, target)
    db = metrics.davies_bouldin_score(dados.real.T, target)
        
    # Computes the average accuracy
    average = sum(lista)/len(lista)
    maximo = max(lista)

    return [sc, maximo, ch, db]


# Plot scatterplots in 2D
def PlotaDados(dados, labels, metodo):
    nclass = len(np.unique(labels))
    if metodo == 'LDA':
        if nclass == 2:
            return -1
    # Convert labels to integers
    lista = []
    for x in labels:
        if x not in lista:  
            lista.append(x)     
    # Map labels to integers
    rotulos = []
    for x in labels:  
        for i in range(len(lista)):
            if x == lista[i]:  
                rotulos.append(i)
    # Convert to array
    rotulos = np.array(rotulos)
    # Select colors
    if nclass > 11:
        cores = ['black', 'gray', 'silver', 'whitesmoke', 'rosybrown', 'firebrick', 'red', 'darksalmon', 'sienna', 'sandybrown', 'bisque', 'tan', 'moccasin', 'floralwhite', 'gold', 'darkkhaki', 'lightgoldenrodyellow', 'olivedrab', 'chartreuse', 'palegreen', 'darkgreen', 'seagreen', 'mediumspringgreen', 'lightseagreen', 'paleturquoise', 'darkcyan', 'darkturquoise', 'deepskyblue', 'aliceblue', 'slategray', 'royalblue', 'navy', 'blue', 'mediumpurple', 'darkorchid', 'plum', 'm', 'mediumvioletred', 'palevioletred']
        np.random.shuffle(cores)
    else:
        cores = ['blue', 'red', 'cyan', 'black', 'orange', 'magenta', 'green', 'darkkhaki', 'brown', 'purple', 'salmon']
    # Create figure
    plt.figure(1)
    for i in range(nclass):
        indices = np.where(rotulos==i)[0]
        cor = cores[i]
        plt.scatter(dados[indices, 0], dados[indices, 1], c=cor, marker='*')
    # Save file    
    nome_arquivo = metodo + '.png'
    plt.title(metodo+' clusters')
    plt.savefig(nome_arquivo)
    plt.close()

#%%%%%%%%%%%%%%%%%%%%  Data loading
# OpenML datasets
# Select one!
X = skdata.load_iris()
#X = skdata.fetch_openml(name='veteran', version=2) 
#X = skdata.fetch_openml(name='aids', version=1) 
#X = skdata.fetch_openml(name='bolts', version=2) 
#X = skdata.fetch_openml(name='threeOf9', version=1) 
#X = skdata.fetch_openml(name='balance-scale', version=1) 
#X = skdata.fetch_openml(name='user-knowledge', version=1)    
#X = skdata.fetch_openml(name='monks-problems-1', version=1)            
#X = skdata.fetch_openml(name='planning-relax', version=1)              
#X = skdata.fetch_openml(name='prnn_crabs', version=1)
#X = skdata.fetch_openml(name='fri_c0_500_10', version=2)
#X = skdata.fetch_openml(name='diggle_table_a2', version=1)
#X = skdata.fetch_openml(name='pwLinear', version=2)
#X = skdata.fetch_openml(name='chscase_census5', version=2)
#X = skdata.fetch_openml(name='blogger', version=1)
#X = skdata.fetch_openml(name='qualitative-bankruptcy', version=1)
#X = skdata.fetch_openml(name='KungChi3', version=1)
#X = skdata.fetch_openml(name='MegaWatt1', version=1)
#X = skdata.fetch_openml(name='diabetes_numeric', version=2)
#X = skdata.fetch_openml(name='mfeat-fourier', version=1) 
#X = skdata.fetch_openml(name='corral', version=1)
#X = skdata.fetch_openml(name='xd6', version=1)

dados = X['data']
target = X['target']  

n = dados.shape[0]
m = dados.shape[1]
c = len(np.unique(target))
nn = round(np.sqrt(n))

print('N = ', n)
print('M = ', m)
print('C = %d' %c)
print('K = %d' %nn)
print()

# Only for OpenML datasets
# Need to treat categorical data manually
if not isinstance(dados, np.ndarray):
    cat_cols = dados.select_dtypes(['category']).columns
    dados[cat_cols] = dados[cat_cols].apply(lambda x: x.cat.codes)
    dados = dados.to_numpy()
    target = target.to_numpy()

# Data standardization (to deal with variables having different units/scales)
dados = preprocessing.scale(dados)

#%%%%%%%%%%% Simple PCA
print('PCA\n') 
dados_pca = myPCA(dados, 2)

#%%%%%%%%%%% ISOMAP
print('ISOMAP\n')
model = Isomap(n_neighbors=nn, n_components=2)
dados_isomap = model.fit_transform(dados)
dados_isomap = dados_isomap.T

#%%%%%%%%%%% LLE
print('LLE\n')
model = LocallyLinearEmbedding(n_neighbors=nn, n_components=2)
dados_LLE = model.fit_transform(dados)
dados_LLE = dados_LLE.T

#%%%%%%%%%%% Lap. Eig.
print('Laplacian Eigenmaps\n')
model = SpectralEmbedding(n_neighbors=nn, n_components=2)
dados_Lap = model.fit_transform(dados)
dados_Lap = dados_Lap.T

#%%%%%%%%%%% LDA
print('LDA\n')
if c > 2:
    model = LinearDiscriminantAnalysis(n_components=2)
else:
    model = LinearDiscriminantAnalysis(n_components=1)
dados_lda = model.fit_transform(dados, target)
dados_lda = dados_lda.T

#%%%%%%%%%%% Supervised classification
L_pca = Classification(dados_pca.real, target, 'PCA')
L_iso = Classification(dados_isomap, target, 'ISOMAP')
L_lle = Classification(dados_LLE, target, 'LLE')
L_lap = Classification(dados_Lap, target, 'Lap. Eig.')
L_lda = Classification(dados_lda, target, 'LDA')

#%%%%%%%%%%%% SSDR
print('==================================== SSDR =====================================')
MAX = 15    # Number of executions
avg_ssdr = np.zeros(MAX)
sc_ssdr = np.zeros(MAX)
ch_ssdr = np.zeros(MAX)
db_ssdr = np.zeros(MAX)
for i in range(MAX):
    print('i = %d' %i)
    dados_ssdr = SSDR(dados, target, perc=0.1, alpha=1, beta=10, d=2)
    L_ssdr = Classification(dados_ssdr, target, 'SSDR')
    sc_ssdr[i] = L_ssdr[0]
    avg_ssdr[i] = L_ssdr[1]
    ch_ssdr[i] = L_ssdr[2]
    db_ssdr[i] = L_ssdr[3]

#%%%%%%%%%%%% Entropic SSDR
print('================================== Entropic SSDR =======================================')
MAX = 15    # Number of executions
fim = min(21, n//3)     # Interval of values for K (number of neighbors)
lista_k = list(range(2, fim))
avg_essdr = np.zeros(MAX)
sc_essdr = np.zeros(MAX)
ch_essdr = np.zeros(MAX)
db_essdr = np.zeros(MAX)
acuracias_essdr = np.zeros(len(lista_k))
scs_essdr = np.zeros(len(lista_k))
chs_essdr = np.zeros(len(lista_k))
dbs_essdr = np.zeros(len(lista_k))
for k in lista_k:
    print('k = %d' %k)
    for i in range(MAX):
        dados_ent_ssdr = Entropic_SSDR(dados, target, dist='KL', k=k, perc=0.1, alpha=1, beta=10, d=2)
        L_ent_ssdr = Classification(dados_ent_ssdr, target, 'Entropic SSDR')
        sc_essdr[i] = L_ent_ssdr[0]
        avg_essdr[i] = L_ent_ssdr[1]
        ch_essdr[i] = L_ent_ssdr[2]
        db_essdr[i] = L_ent_ssdr[3]
    acuracias_essdr[k-2] = avg_essdr.max() 
    scs_essdr[k-2] = sc_essdr.max()
    chs_essdr[k-2] = ch_essdr.max()
    dbs_essdr[k-2] = db_essdr.max()

#%%%%%%%%%%%% Entropic SSDR MST
print('=============================== Entropic SSDR MST ====================================')
MAX = 15    # Number of executions
fim = min(21, n//3)   # Interval of values for K (number of neighbors)
lista_k = list(range(2, fim))
avg_essdr_mst = np.zeros(MAX)
sc_essdr_mst = np.zeros(MAX)
ch_essdr_mst = np.zeros(MAX)
db_essdr_mst = np.zeros(MAX)
acuracias_essdr_mst = np.zeros(len(lista_k))
scs_essdr_mst = np.zeros(len(lista_k))
chs_essdr_mst = np.zeros(len(lista_k))
dbs_essdr_mst = np.zeros(len(lista_k))
for k in lista_k:
    print('k = %d' %k)
    for i in range(MAX):
        dados_ent_ssdr_mst = Entropic_SSDR_MST(dados, target, dist='KL', k=k, perc=0.1, alpha=1, beta = 2, d=2)
        L_ent_ssdr_mst = Classification(dados_ent_ssdr_mst, target, 'Entropic SSDR MST')
        sc_essdr_mst[i] = L_ent_ssdr_mst[0]
        avg_essdr_mst[i] = L_ent_ssdr_mst[1]
        ch_essdr_mst[i] = L_ent_ssdr_mst[2]
        db_essdr_mst[i] = L_ent_ssdr_mst[3]
    acuracias_essdr_mst[k-2] = avg_essdr_mst.max()
    scs_essdr_mst[k-2] = sc_essdr_mst.max()
    chs_essdr_mst[k-2] = ch_essdr_mst.max()
    dbs_essdr_mst[k-2] = db_essdr_mst.max()    

# Print results
print('========== RESULTS ==========')
print()

print('PCA SC: %f' %L_pca[0])
print('PCA acc: %f' %L_pca[1])
print()

print('ISOMAP SC: %f' %L_iso[0])
print('ISOMAP acc: %f' %L_iso[1])
print()

print('LLE SC: %f' %L_lle[0])
print('LLE acc: %f' %L_lle[1])
print()

print('Laplacian Eigenmaps SC: %f' %L_lap[0])
print('Laplacian Eigenmaps acc: %f' %L_lap[1])
print()

print('LDA SC (supervised): %f' %L_lda[0])
print('LDA acc (supervised): %f' %L_lda[1])
print()

print('SSDR SC: %f' %sc_ssdr.mean())
print('SSDR acc: %f' %avg_ssdr.mean())
print()

print('Entropic SSDR SC: %f' %max(scs_essdr))
print('Entropic SSDR acc: %f' %max(acuracias_essdr))
print('K* = %d' %(acuracias_essdr.argmax()+2))
print()

print('Entropic SSDR MST SC: %f' %max(scs_essdr_mst))
print('Entropic SSDR MST acc: %f' %max(acuracias_essdr_mst))
print('K* = %d' %(acuracias_essdr_mst.argmax()+2))
print()


# Plot data
PlotaDados(dados_pca.T, target, 'PCA')
PlotaDados(dados_isomap.T, target, 'ISOMAP')
PlotaDados(dados_LLE.T, target, 'LLE')
PlotaDados(dados_Lap.T, target, 'LAP')
PlotaDados(dados_lda.T, target, 'LDA')
PlotaDados(dados_ssdr.T, target, 'SSDR')
PlotaDados(dados_ent_ssdr.T, target, 'ENT SSDR')
PlotaDados(dados_ent_ssdr_mst.T, target, 'ENT SSDR MST')