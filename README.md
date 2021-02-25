# Active Learning for Bayesian Neural Networks with Gaussian Processes

_(last modified 01.02.2021 by Lukas Erlenbach, [LinkedIn profile](https://www.linkedin.com/in/lukas-erlenbach/))_

This directory contains parts of the code that I have written during my master thesis (which is available [here](https://drive.google.com/file/d/1pgNn8ZAEjHANyzPy7JGC8-J0p2qUgxoO/view?usp=sharing)).

The project implements a active learning framework for Bayesian Neural Networks and regression tasks and is based on a paper by Tsymbalov et al. [1]. Furthermore, it contains a generalization with faster runtime which is described in chapter 5.4 of the thesis. 

## (very short) Introduction

(I also published a [blog post](https://lerlenbach.medium.com/active-learning-for-bayesian-neural-networks-b8471212850f) with an intuitive introduction on Medium.)

As example data, the housing dataset from sklearn is used which is a well known regression problem. The aim is to train a Bayesian Neural Network with low RMSE while minimizing the number of training data points.

After a Bayesian Neural Network is trained of an initial set of points, Active Learning iterations are performed. In each iteration, a Sampler selects points from a pool (without having access to the lables of these points) which then get added to the training data before the training process is resumed.

For further details please consider the paper [1], my [blog post](https://lerlenbach.medium.com/active-learning-for-bayesian-neural-networks-b8471212850f) or my [thesis](https://drive.google.com/file/d/1pgNn8ZAEjHANyzPy7JGC8-J0p2qUgxoO/view?usp=sharing).

Two example scripts are provided which showcase:

  1. That the GPA Sampler from [1] is superior to randomly selecting additional points.
  2. That the Fast GPA Sampler computes the same points as the GPA Sampler from [1] while reducing the runtime.


## Usage

To run the experiments, it is recommended to set up a virtualenv with python3.8 and intall the requirements via

        virtualenv --python=python3.8 venv
        source venv/bin/activate
        pip install -r requirements.txt

Afterwards the two example scripts can be called via

        python source/compare_gpa_rand.py
        python source/compare_fastgpa_batchgpa.py

The first script takes about 10min to run on my machine, the second less than 2min. Both create a results directory which contains logfile, experiment configurations as well as a plot with the most important metrics.

The first scripts compares the GPA Sampler from [1] to random point selection. The results depend on the chosen random seed but in most of the cases using the GPA Sampler leads to a faster and more stable convergence. (For qualitative results, refer to chapter 6 in the thesis.)
  
The second script compares the Fast GPA Sampler (from my thesis) to the GPA Sampler (from [1]). Both Sampler in theory compute the same posterior variance, however, sometimes differences in the convergence occur from rounding errors. In general, the Fast version of the sampler does the same job in shorter time as it avoids the repeated inversion of the posterior covariance matrix.
  
To change the experimental setup, consider changing the parameter values (in particular random seed and train/pool/test sizes) in the python scripts and the .yaml configuration files in configs/.

## Example Results from source/compare_gpa_rand.py

![Example Results](https://github.com/LukasErlenbach/active_learning_bnn/blob/master/images/result_script1.png)

## Example Results from source/compare_fastgpa_batchgpa.py

![Example Results](https://github.com/LukasErlenbach/active_learning_bnn/blob/master/images/result_script2.png)


### References

  * [1] E. Tsymbalov, S. Makarychev, A. Shapeev, M. Panov, _Deeper Connections between Neural Networks and Gaussian Processes Speed-up Active Learning_, 2019, [https://arxiv.org/abs/1902.10350](https://arxiv.org/abs/1902.10350)


  * the code contained in source/models/ is modified from:

    [2] D. Hafner, D. Tran, T. Lillicrap, A. Irpan, J. Davidson, _Noise Contrastive Priors for Functional Uncertainty_, arXiv eprint 1807.09289, 2018, [https://github.com/brain-research/ncp](https://github.com/brain-research/ncp)


  * the AttrDict class in source/classes/attrdict.py is taken from:

    Danijar Hafner, Patterns for Fast Prototyping with TensorFlow, blog post, [https://danijar.com/patterns-for-fast-prototyping-with-tensorflow/](https://danijar.com/patterns-for-fast-prototyping-with-tensorflow/)
