## Discovery Symmetries in Data
### Symmetry Regularization on MNIST dataset under rotation trasformations

![](./img/1_test_results.png){fig-align="center" width=50%}

# Introduction

This is the first attempt to implement a Fully Connected 1 layer neural network (`number of neurons = 728`) multiclass classification problem that is invariant to the rotation of the MNIST dataset over a certain angle (in my analysis I selected an angle of 3 degree) and to compare the results with the same network but with different regularization terms, in particular:

- the *FCNN* with only the cross entropy loss. (**The CE LOSS**)
- the *FCNN* with the cross entropy loss and the regularization term for the challenge (**The Challenge LOSS**)
- the *FCNN* with the cross entropy loss and the Symmetry-Adapted Regularization terms (**The Symmetry LOSS**)

The parameters used are:

- `n_epochs = 500`
- `angle_rotation = 3 degree`
- `cardinality = 120` (360/angle_rotation)
- `batch_size_training = 6000` (6000 random rotated images + 120 atoms that is the orbit of 1 image selected randomly)
- `barch_size_test = 1000` (1000 random rotated images + 120 atoms that is the orbit of 1 image selected randomly)
- `learning_rate = 0.001`
- `lambda_reg=0.01`
- `lambda_symm=0.01`
- `lambda_comm=0.01`
- `lambda_symm_comm=0.01`
- `sigma_squared=0.001` (in order to compute the symm_loss)


Notes: 
These results are obtained using the best model weights over 10 runs over 500 epochs.
The plots of the Loss functions and the weight matrices of first layer are saved inside the folder `./img`
The weights of the best performing Loss for each analysis are saved in the folder `./best_model_weights`

- $\mathcal{S_N} \in \R^{10\times600}$ : Training set 
- $\mathcal{W1} \in \R^{120\times728}$ : Weights that connect input layer ($D_i= 784$) with the hidden layer ($D_h= 120$), cardinality of the group is `120`.
- $\mathcal{W2} \in \R^{120\times10}$ : Weights that connect hidden layer ($D_h= 120$) with the output layer ($D_y= 10$), cardinality of the group is `120`.
- $\mathcal{W}=\{W1,W2\}$

# The CE LOSS 

$$
\mathcal{L(S_N,W)} = \mathcal{L_{CE}(S_N, W)}
$$

For this Loss it was implemented only the [the cross entropy loss](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html), and it's the loss to beat 

- **Train Accuracy: 100.0%**
- Test Accuracy: 70.6% 
- Invariance evaluation:
  - Average Euclidean norm of the difference: 37.407 (39.304 control)
  - **Difference: 1.897**
- For 10 runs it tooks 12.5 minute and for the results we used the best recorded one

# The Challenge LOSS

$$
\mathcal{L(S_N,W)} = \mathcal{L_{CE}(S_N, W)} + \mathcal{L_{comm}(S_N,W1)} + \mathcal{L_{sort}(W1)}
$$

For this Loss it was implemented the [the cross entropy loss](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html) and the regularization term as explained for the challenge. (the commutator term between the covariance matrix of the weigths and the data and the Gramian sorted term)

- Train Accuracy: 87.88% 
- Test Accuracy: 65.0% <span style="color: red">(-5.9 decline) </span>
- Invariance evaluation:
  - Average Euclidean norm of the difference: 12.347 (13.014 control)
  - Difference: 0.667
- For 10 runs it tooks 16.13 minute and for the results we used the best recorded one

# The Symmetry LOSS

$$
\mathcal{L(S_N,W)} = \mathcal{L_{CE}(S_N, W)} + \mathcal{L_{comm}(S_N,W1)} + \mathcal{L_{symm}(W1)}
$$

For this Loss it was implemented the [the cross entropy loss](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html) and the regularization term as explained in [the paper](https://arxiv.org/abs/2006.14027).

- **Train Accuracy: 100.0%**
- **Test Accuracy: 70.9%** <span style="color: green">(+0.3 improvement) </span>
- Invariance evaluation:
  - Average Euclidean of the difference: 34.497 (36.124 control) 
  - Difference: 1.627
- For 10 runs it tooks **17.53 minute** and for the results we used the best recorded one

::: {layout-ncol=2}
![](./img/ce_loss.png)

![Cross Entropy Weight Matrices 10x784](./img/ce_weights.png)

![](./img/challenge_loss.png)

![Challenge Weight Matrices 10x784](./img/challenge_weights.png)

![](./img/symmetry_loss.png)

![Symmetry Weight Matrices 10x784](./img/symmetry_weights.png)

:::

# Conclusion and further investigations

This initial exploration into the application of symmetry regularization within a Fully Connected Neural Network (FCNN) framework on the MNIST dataset has provided invaluable insights into the operational intricacies of neural networks and the practical use of PyTorch. This investigation has termined with a solid <span style="color: green"> 0.03 improvement</span>  in the test set respect the CE loss performance. There were no notable differences observed in the symmetry loss slope when compared to the cross-entropy (CE) loss.

The adoption of a **Symmetry Regularization framework** has been instrumental in shedding light on how to effectively address sample complexity challenges, while also revealing areas that require further exploration to fully leverage this framework's potential.

Looking ahead, several critical questions remain unanswered, signifying the need for further investigation:

**Variability in the test/training set**: The observed improvement is not statistically significant, likely due to variability in the test/training set. Additional testing with alternative datasets may be required to mitigate the impact of noise inherent in the training data.

**Parameter Sensitivity Analysis**: Understanding the intricate relationship between various hyperparameters and their impact on the loss function's behavior and, subsequently, on the model's predictive accuracy.

**Generalization Across Diverse Image Sets**: Assessing the model's robustness and generalization capabilities when the training set (MNIST for Orbit classification) and a distinctly different test set (comprising entirely new images with the same dimensionality and color scale) are used. This inquiry aims to understand how variations in the test dataset impact the model's accuracy, providing insights into the model's adaptability to new, unseen data.

# References
- [1] Teacher lectures (2023-2024): Advanced Topics in Machine Learning
- [2] [Symmetry-Adapted Regularization for Learning invariance in Neural Networks](https://arxiv.org/abs/2006.14027)
- [3] [Equivariant Neural Networks](https://dmol.pub/dl/Equivariant.html#equivariant-neural-networks-%20with-constraints)
- [4] [Github code](https://github.com/giemsei/Discovery-Symmetries-in-Data) 
