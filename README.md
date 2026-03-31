# Enhanced KAN visualisation for better interpretation

![pic0](/Pictures/pic6.png)
*Left: visualization of KAN, proposed by Liu et al.*

*Right: Improved visualization of KAN with data distributions histograms of activation functions and coloring them in accordance with the reper channel.*

# $\lambda$-KAN with an architecturally integrated interpretation mechanism

![pic00](/Pictures/pic7.png)
*Graphic representation of KAN (left) and its modification λ-KAN (right).*


# General Description
This project presents the software code for applying Kolmogorov-Arnold networks (KAN) to the inverse problem of spectroscopy. The results are described in the articles:
* Kupriyanov, G. A., et al. (2025). Interpretation of Kolmogorov–Arnold Networks Using the Example of Solving the Inverse Problem of Photoluminescence Spectroscopy. *Optical Memory and Neural Networks*. https://doi.org/10.3103/S1060992X25602052
* Kupriyanov, G. A., et al. (2024). Solution of an inverse problem of spectroscopy using Kolmogorov-Arnold networks. *Optical Memory and Neural Networks*. https://doi.org/10.3103/S1060992X24700747

Statement of the inverse problem: to determine the concentration of metal ions in a solution based on the fluorescence spectrum.

The spectrum was represented in compressed (5 input values) and full representations (500 input values).

![pic1](/Pictures/Pic1.png)
*Left: two-dimensional excitation-emission matrix of a CD aqueous suspension in the presence of heavy metal ions. 
Right: one-dimensional fluorescence spectrum at 350 nm excitation and its parameterization using 5 parameters.*

The machine learning models used were Kolmogorov-Arnold networks (KAN), random forests (RF), gradient boosting (GB), and a perceptron with one hidden layer (MLP). The results are presented in bar plots:

![pic2](/Pictures/Pic2.png)
*Mean absolute error in determining ion concentrations by four methods based on compressed representation (left) and full spectrum representation (right).*

In terms of predictive capability, KAN performed no worse than the reference methods and, in some cases, even better. Moreover, the number of trainable parameters in KAN is an order of magnitude smaller than in the used perceptron model.

| Number of Trainable Parameters | MLP | KAN |
|--------------------------------|-----|-----|
| Compressed spectrum (5 input values) | 449 | 48 |
| Full spectrum (500 input values) | 32,129 | 4,008 |

## Visual Example of KAN Training *(with compressed spectra as input*).

### *1. Training KAN with B-splines as activation functions.*

![gif1](/Pictures/gif1.gif)

### *2. Approximating KAN activation functions with our chosen set of analytical functions. Fine-tuning.*

![gif2](/Pictures/gif2.gif)

### *3. Visual and formulaic representation of the KAN model after training.*

![pic3](/Pictures/pic3.png)

![pic4](/Pictures/pic4.png)

### *4. Enhanced visualisation of KAN model.*

![pic5](/Pictures/pic5.png)

## $\lambda$-KAN with an architecturally integrated interpretation mechanism

$\lambda$-KAN is modification of KAN, based on Kahane`s version of Kolmogorov-Arnold theorem:

$$f(x_{1},\cdot\cdot\cdot, x_{n})=\sum_{q=1}^{2n+1}\Phi_{q}(\sum_{p=1}^{n}\lambda_{p}\cdot\varphi_{q}(x_{p}))$$

This modification of KAN allows treat $\lambda_{p}$ coefficients as measure of sensitivity model to $x_{p}$ inputs.

![pic6](/Pictures/pic8.png)
_Results of training and analysis of λ-KAN for $Cr^{3+}$ ion._

*__Top left corner:__ paired plots of true and estimated ion concentrations; __Top right corner:__ histogram of distribution of the cosine metric between the λ-coefficients vector and the true gradient vector of the model prediction by input channels. __Bottom half:__ three histograms assessing the importance of input features by the methods of SHAP analysis, gradient analysis, and λ-coefficient analysis, respectively.*

The graphs of the input features importance obtained by Shapley and gradient analyses correlate with the $\lambda$-vectors.
The histogram of the distribution of the cosine measure is concentrated at values of ±1, which also confirms the correctness of the interpretation of $\lambda$-KAN using $\lambda$-coefficients.


---
# Code files


1) _Main_experiment_ : code implementing the training and validation of KAN and reference methods:
    * Squeezed_input.ipynb - Compression (parameterization) of input spectra up to 5 values. Perceptron and KAN training on compressed spectra.
    * Mult_exper_Cr/Cu/Ni/NO3.ipynb - Launch of RF, GB, MLP and KAN models to determine the metal ion Cr/Cu/Ni/NO3 while maintaining statistics.


2) _Interpretability_ : the study of the interpretation of KAN in solving the inverse problem of spectroscopy.
    * Accuracy_squeezed_input.ipynb - determining accuracies of KAN, $\lambda$-KAN and other models.
    * Interp_squeezed_input.ipynb - a study of the interpretative capabilities of improved KAN visualization.
    * Interp_squeezed_input_lmd.ipynb - exploring the interpretative capabilities of $\lambda$-KAN.
    * KAN__lmd_KAN_interp_compar.ipynb - comparing results of $\lambda$-KANs' and KANs' interpretations. 

3) _Supportive code_:
    * json_config.txt - configuration files with hyper parameters of models..
    * raw_data_processing.py - functions for loading fluorescence maps.
    * tools.py - 
        1. functions for working with JSON files; 
        2. functions for compressing input spectra;
        3. definition of class KAN_es(KAN) - KAN with early stopping based on the validation set. Saved here as legacy, which was used for gaining publicated materials. We recommend use KAN_es_2 class instead.
        4. definition of class KAN_es_2(KAN) - KAN with early stopping based on the validation set and enhanced plotting techniqe.
        5. definition of class tlmdKAN(KAN) - realisation of $\lambda$-KAN.
---