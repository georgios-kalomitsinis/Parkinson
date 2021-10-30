<p align="center">
<img src="https://user-images.githubusercontent.com/80779522/139533684-4f9452c9-c53c-413b-8038-5dcb4c868636.png" width="350" />
  
# Parkinson

Parkinson's disease is one of the most painful, dangerous and incurable diseases that occur in older people (mainly over 50 years). It concerns the death of dopamine neurons in the brain. This neurodegeneration leads to a range of symptoms, such as coordination issues, slowness of movement, voice changes, stiffness and even progressive disability. The symptoms and course of the disease vary, so it is often not diagnosed for many years. So far, there is no cure, although there is medication that offers a significant relief of symptoms, especially in the early stages of the disease [2]. Therefore, it is crucial to develop more sensitive diagnostic tools for detecting the disease, which is the main goal of this repository to discriminate healthy people from those with parkinson disease (PD). 
  
<p align="center">
<img src="https://user-images.githubusercontent.com/80779522/139534329-046c979c-3e40-4570-8397-036b9307c3a3.png" width="600" />
<figcaption align = "center"><p align="center">
  Figure 1. Stages of PD.</figcaption>
</figure>

## Dataset Description
In this repository, the dataset is obtained from [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/concrete+compressive+strength). This dataset is composed of a range of biomedical voice measurements from 31 people, 23 with PD. Each column in the datset is a particular voice measure, and each row corresponds one of 195 voice recording from these individuals.

<div align="center">

name | ASCII subject name and recording number
:---:|:---:
MDVP:Fo(Hz) | Average vocal fundamental frequency
MDVP:Fhi(Hz) | Maximum vocal fundamental frequency
MDVP:Flo(Hz) | Minimum vocal fundamental frequency
MDVP:Jitter(%)<br>MDVP:Jitter(Abs)<br>MDVP:RAP<br>MDVP:PPQ<br>Jitter:DDP | Several measures of variation in fundamental frequency
MDVP:Shimmer<br>MDVP:Shimmer(dB)<br>Shimmer:APQ3<br>Shimmer:APQ5<br>MDVP:APQ<br>Shimmer:DDA | Several measures of variation in amplitude
NHR<br>HNR | Two measures of ratio of noise to tonal components in the voice
status | Health status of the subject<br>(one) - Parkinson's<br>(zero) - healthy
RPDE<br>D2 | Two nonlinear dynamical complexity measures
DFA | Signal fractal scaling exponent
spread1<br>spread2<br>PPE | Three nonlinear measures of fundamental frequency variation 

</div>
<figcaption align = "center"><p align="center">Table 1. Attribute Information.</figcaption>
</figure>

<p align="center">
<img src="https://user-images.githubusercontent.com/80779522/139536668-f9d8f067-44a8-4531-99d3-a82c24f867d8.png" width="600" />
<figcaption align = "center"><p align="center">
  Figure 2. PD and healthy voice instances.</figcaption>
</figure>
  
  
## Methodology

Each person has 6 or 7 voice measurements. For the evaluation of each algorithm taken into account, the dataset was divided into individuals and not at the level of voice measurements. Furthermore, the split of the dataset was performed 10 times, with different people in the train set and test set, with ```train_size = 0.8```, where it is equivalent to 25 people. Also, The GridSearchCV procedure was applied to find the best hyperparameters of each algorithm (```LeaveOneGroupOut method```). 

## Modelling and Evaluation
**ALGORITHMS**

* *Logistic regression*
* *Decision Tree classifier*
* *Gaussian Naive Bayes*
* *Random Forest*
* *Support Vector Machine*
* *XGB classifier*

**METRICS**
Due to the nature of the problem, as a medical, the goal is to reduce positive inaccuracies in the calculation. Either the precision score or the recall do not cover the purpose, as well as the accuracy. Therefore, for better results, the f1-score measure is taken into account, where a balance between precison and recall is sought even in imbalanced classes.

<div align="center">

| Name | Formula | 
| :---: | :---: | 
| Accuracy | ![equation](https://latex.codecogs.com/gif.latex?%5Cbg_white%20%5Cfrac%7BTP&plus;TN%7D%7BTP&plus;FP&plus;FN&plus;TN%7D) |
| Precision | ![equation](https://latex.codecogs.com/gif.latex?%5Cbg_white%20%5Cfrac%7BTP%7D%7BTP&plus;FP%7D) |
| Recall | ![equation](https://latex.codecogs.com/gif.latex?%5Cbg_white%20%5Cfrac%7BTP%7D%7BTP&plus;FN%7D) |
| F-Score | ![equation](https://latex.codecogs.com/gif.latex?%5Cbg_white%20%5Cfrac%7B2%5Ctimes%20%28Recall%20%5Ctimes%20Precision%29%7D%7BRecall%20&plus;%20Precision%7D) |

</div>
<figcaption align = "center"><p align="center">Table 3. Calculated metrics where TP, TN, FP, FN corresponds to True Positives, True Negatives, False Negatives and False Positives, respectively.</figcaption>
</figure>

## Results

<p align="center">
<img src="https://user-images.githubusercontent.com/80779522/139536894-64512cbe-dd26-425d-a4fe-1335467415a1.png" width="800" />
<figcaption align = "center"><p align="center">
  Figure 3. Average of the metrics of each classifier.</figcaption>
</figure>



