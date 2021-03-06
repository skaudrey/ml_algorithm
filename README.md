# Content: M.L. algorithms
The naive implementation of some M.L. algorithms, which will be updated continuously.

The algorithms that have been implemented are listed as follows:

## LogisticRegression
1.  Code: /logistic/logistic.py
2.  The file of derivation: /logistic/file/lrTex/lr.pdf
3.  Test data: sklearn.datasets.make_moons
4.  Details: maximize cross entropy,gradient ascent
5.  Result: /logistic/fig/decision.fig
## SVM-SMO
1.  Code: /svm/svm.py
2.  The file of derivation: /svm/file/svmTex/svm.pdf
3.  Test data: /svm/flowers.csv
4.  Details: SMO,coordinate ascent
5.  Result: /svm/file/svmTex/svm.pdf

## K-Means
1.  Code: /kmeans/kmeans.py
2.  The file of derivation: /kmeans/file/kmeansTex/kmeans.pdf
3.  Test data: /kmeans/data.csv
4.  Details: minimize the summation of the square of errors
5.  Result: /kmeans/file/kmeansTex/kmeans.pdf

## GMM-EM
Some bugs still need to be fixed.
1.  Code: /EM-GMM/gmm.py
2.  The file of derivation: /EM-GMM/file/gmmTex/gmm.pdf
3.  Test data: function generateData(...) in gmm.py, which generates data from four Gaussian distribution
4.  Details: Maximization likelihood expectation
5.  Result: /EM-GMM/file/gmmTex/gmm.pdf.

## Perceptron
1.  Code: /perceptron/perceptron.py
2.  The file of derivation: /perceptron/file/percepTex/perceptron.pdf
3.  Test data: /svm/flowers.csv
4.  Details: Minimization #(samples which are classified incorrectly),SGD
5.  Result: /perceptronfile/percepTex/perceptron.pdf

## Naive Bayes
1.  Code: /naive-bayes/naivebayes.py
2.  The file of derivation: /naive-bayes/file/nbTex/nb.pdf
3.  Test data: function loadDataSet in naivebayes.py
4.  Details: Generative model,Laplace smoothing
5.  Result: /naive-bayes/file/nbTex/nb.pdf

## ANN
### LeNet-Keras
1.  Code: /ANN/LeNet/Lenet-keras.py

### MLP-Numpy
1.  Code: /ANN/MLP/mlp.py

# Content: Simulink of basic statistical ideas
## Sampling: MCMC
1.  Code: /sampling/Metropolis.py, /sampling/Gibbs.py
2.  The file of concept: /sampling/file/samplingTex/mcmc.pdf
3.  Test data: one dimension Gaussian distribution in Metropolis.py and two-dimensional Gaussian distribution in Gibbs.py
4.  Details: MCMC, Markov chain.
5.  Result: /sampling/file/samplingTex/mcmc.pdf

# Reference
1.  李航. 统计学习方法[M]. 清华大学出版社, 2012.
2.  Andrew Ng. CS229. Stanford, 2012.
3.  Raschka S. Python Machine Learning[M]. Packt Publishing, 2014.
4.  Feifie Li et al. CS231n. Stanford, 2017.
5.  Blogs etc. The details of which are listed in the derivation files of each packages.