# General Description
This project presents the software code for applying Kolmogorov-Arnold networks (KAN) to the inverse problem of spectroscopy, as described in the article "Solution of an inverse problem of spectroscopy using Kolmogorov-Arnold networks." (http://dx.doi.org/).

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
# Description of the purpose of the files in this project
Directory hierarchy: 

0) Auxiliary code:
    * json_config.txt - configuration file with hyperparameters of the models.
    * raw_data_processing.py - functions for loading fluorescence maps.
    * tools.py - 
        1. functions for working with JSON files; 
        2. functions for compressing input spectra;
        3. definition of class KAN_es(KAN) - KAN with early stopping based on the validation set. Saved here as legacy, which was used for gaining publicated materials. We recommend use KAN_es_2 class instead.
        4. definition of class KAN_es_2(KAN) - KAN with early stopping based on the validation set and enhanced plotting techniqe.

1) Code demonstrating the operation of several stages of the project:
    * Full_input.ipynb - Training a perceptron and KAN on full spectra as input (500 values).
    * Squeezed_input.ipynb - Compression (parameterization) of input spectra to 5 values. Training a perceptron and KAN on compressed spectra.
    * Mult_exper.ipynb - example of multiple runs of RF, GB, MLP, and KAN models to identify a single metal ion with a set of statistics.
    * Interp_squeezed_input.ipynb - investigating interpretation capabilities of defolt KAN plotting.
    * Interp_squeezed_input_2.ipynb - repeats after Interp_squeezed_input.ipynb, but with enhanced plotting of KAN.

2) Running the code from Mult_exper.ipynb for different ions.
    * Mult_exper_Cr.ipynb
    * Mult_exper_Cu.ipynb
    * Mult_exper_Ni.ipynb
    * Mult_exper_NO3.ipynb
    
---