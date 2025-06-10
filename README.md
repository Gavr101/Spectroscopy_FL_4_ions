# Enhanced KAN visualisation for better interpretation

![pic0](/Pictures/pic6.png)
*Left: visualization of KAN, proposed by Liu et al.*

*Right: Improved visualization of KAN with data distributions histograms of activation functions and coloring them in accordance with the reper channel.*

# General Description
This project presents the software code for applying Kolmogorov-Arnold networks (KAN) to the inverse problem of spectroscopy. The results are partially described in the article "Solution of an inverse problem of spectroscopy using Kolmogorov-Arnold networks." (https://doi.org/10.3103/S1060992X24700747).

Apart from that, here introduced enhanced version of plotting KAN (see below).

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
---
# Code files


1) _Main_experiment_/ : code implementing the training and validation of KAN and reference methods:
    * Squeezed_input.ipynb - Compression (parameterization) of input spectra up to 5 values. Perceptron and KAN training on compressed spectra.
    * Mult_exper_Cr/Cu/Ni/NO3.ipynb - Launch of RF, GB, MLP and KAN models to determine the metal ion Cr/Cu/Ni/NO3 while maintaining statistics.


2) _Interpretability_/ : the study of the interpretation of KAN in solving the inverse problem of spectroscopy.
    * Interp_squeezed_input.ipynb - a study of the interpretative capabilities of improved KAN visualization.
    * Interp_squeezed_input_lmd.ipynb - exploring the interpretative capabilities of $\lambda$-KAN.

3) Supportive code:
    * json_config.txt - configuration files with hyper parameters of models..
    * raw_data_processing.py - functions for loading fluorescence maps.
    * tools.py - 
        1. functions for working with JSON files; 
        2. functions for compressing input spectra;
        3. definition of class KAN_es(KAN) - KAN with early stopping based on the validation set. Saved here as legacy, which was used for gaining publicated materials. We recommend use KAN_es_2 class instead.
        4. definition of class KAN_es_2(KAN) - KAN with early stopping based on the validation set and enhanced plotting techniqe.
---