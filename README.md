# Proximal generative models
Code for the paper **Robust Generative Learning with Lipschitz-Regularized
α-Divergences Allows Minimal Assumptions on Target Distributions** 

## Instruction and directory structure
Compare different proximal generative models in ```scripts/[model_name]/```.

* ```f-Gamma-GAN```: Lipschitz regularized alpha GAN vs alpha GAN; [(f,Γ)-Divergences: Interpolating between f-Divergences and Integral Probability Metrics](https://arxiv.org/abs/2011.05953)
* ```GPA_NN```: Lipschitz regularized alpha GPA vs alpha GPA; [Lipschitz-regularized gradient flows and generative particle algorithms for high-dimensional scarce data](https://arxiv.org/abs/2210.17230)
* ```ot_flow-master```: OT flow vs CNF; [OT-Flow: Fast and Accurate Continuous Normalizing Flows via Optimal Transport](https://arxiv.org/abs/2006.00104)
* ```sgm_simple```: Score-based generative model (SGM) with Variance Exploding SDE; [Score-Based Generative Modeling through Stochastic Differential Equations
](https://arxiv.org/abs/2011.13456)

```GPA_NN```, ```OT_Flow_GAN```, ```ot_flow-master``` comes with ```runscript.sh``` file to run the experiments with different cases for the examples. 

```f-Gamma-GAN``` has two python codes for different examples. Simply running the code to produce experiment results with different cases for the examples.

```sgm_simple``` contains two jupyter notebooks for different examples. For ```SGM-VE-Learning_student_t``` set df=1.0 or df=3.0 and then run the entire blocks to produce results for the examples. 

Generated samples will be stored in ```assets/[example_name]/```. 

* ```Learning_student_t```: 2D Student-t experiments from ```GPA_NN``` and ```f-Gamma-GAN``` will be stored.
* ```student-t```: 2D Student-t experiments from other models will be stored.
* ```Keystrokes```: Keystrokes experiments will be stored.


After the sample generations, open and run jupyter notebook ```2D_Student_t.ipynb``` for the 2D Student-t example, ```Keystrokes.ipynb``` for the Keystrokes example, ```Heavytail_submanifold.ipynb``` for the 10D Heavytailed distribution embedded in 110D example, and ```Lorenz63.ipynb``` for the Lorenz 63 attractor example to plot the results. 

The resulting plots are stored in ```assets/dataset_name/visualizations``` directory.


## Access to the original dataset

* 2D student-t example: random samples are generated by python library ```scipy.stats.multivariate_t```.
* Keystrokes example: Original data can be accessed from [Observations on Typing from 136 Million Keystrokes](https://dl.acm.org/doi/10.1145/3173574.3174220).
We preprocess the data by combining data from id=[27252, 36718, 56281, 64663, 67159, 97737, 145007, 159915, 264420, 271802] and then extract inter-arrival time between keystrokes.
Resulting dataset consists of 7160 samples and the data is stored at ```pn4874/inter_stroke_time.txt```.
* 10D Heavytailed distribution embedded in 110D example: random samples are generated similarly to [T. Huster, J. Cohen, Z. Lin, K. Chan, C. Kamhoua, N. O. Leslie, C.-Y. J. Chiang, and
V. Sekar, Pareto GAN: Extending the representational power of gans to heavy-tailed distributions,
in International Conference on Machine Learning, PMLR, 2021, pp. 4523–4532.](https://arxiv.org/abs/2101.09113)
* Lorenz 63 example: Lorenz 63 model simulation given standard set of parameters a=10, b=28, c=8.3 generates a trajectory. Target particles for Lorenz attractor are obtained by selecting positions at randomly sampled time t ~ Unif([9900, 10000]).
 
