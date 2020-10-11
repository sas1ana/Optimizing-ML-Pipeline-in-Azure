# Optimizing an ML Pipeline in Azure

## Overview
This project is part of the Udacity Azure ML Nanodegree.
In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model.
This model is then compared to an Azure AutoML run.

## Summary
In this project, We have worked with the [Bank Marketing dataset](https://automlsamplenotebookdata.blob.core.windows.net/automl-sample-notebook-data/bankmarketing_train.csv) The dataset is related with direct marketing campaigns of a Portuguese banking institution. The classification goal is to predict if the client will subscribe (yes/no) a term deposit (variable "y"). For details regarding dataset please go through [description](https://archive.ics.uci.edu/ml/datasets/Bank+Marketing). There is heavy class imbalance in the dataset. There are less than 3200 'yes' records which accounts to around 10% of the total rows.

This is a classification problem. We would be also comparing the performance of AutoML model vs Logistic Regression Model tuned with Azure's HyperDrive.
The performance of both models are almost similar. While, the AutoML model performs slightly better than the Logistic Regression model, with Accuracy about 0.91794 as compared to 0.91525.

## Scikit-learn Pipeline
The Scikit-Learn Pipeline is as follows:

* Data loaded from TabularDatasetFactory
* Dataset is further cleaned using a function called clean_data
* One Hot Encoding has been used for the categorical columns
* Dataset has been splitted into train and test sets
* Built a Logistic Regression model
* Hyperparameters has been tuned using Hyperdrive
* Selected the best model
* Saved the model 

I have choosen Random Sampling to draw parameters at random from the defined search space.

**RandomParameterSampling**: In this sampling algorithm, parameter values are chosen from a set of discrete values or a distribution over a continuous range.
As in our case, {'C': uniform(0.1,0.8),'max_iter':choice(25,50,75)}. The "C" can have a uniform distribution with 0.1 as a minimum value and 0.8 as a maximum value, and the max_iter will be a choice of [25,50,75].

An early termination policy specifies that if we have a certain number of failures, HyperDrive will stop looking for the answer. As in our case, **BanditPolicy**, is based on slack criteria, and a frequency and delay interval for evaluation. The benefit of this policy is that any run that doesn't fall within the slack factor or slack amount of the evaluation metric with respect to the best performing run will be terminated.

**slack_factor**: The amount of slack allowed with respect to the best performing training run. This factor specifies the slack as a ratio. Consider (evaluation_interval=2, slack_factor=0.1) as in our case, Assume that run X is the currently best performing run with an AUC (performance metric) of 0.8 after 2 intervals. Further, assume the best AUC reported for a run is Y. This policy compares the value (Y + Y * 0.1) to 0.8, and if smaller, cancels the run.
## AutoML
VotingEnsemble is the algorithm selected for the best model by AutoML. The fitted model output is a pipeline with two steps: datatransformer and prefittedsoftvotingclassifier. The accuracy of this model was 0.91794. Model explanation give us insight about most important features as "duration", "nr.employed" & "emp.var.rate". The AutoML also performed some data checks, where it has found the class imbalance in the dataset. It tries to balance the dataset before going into model building.

## Pipeline comparison
The AutoML model tried many models before selecting this best model. As mentioned it also provides warning for class imbalance & data checks. Further it also provides explanation for important features in order to select the best fit model.
AutoML also checks for Missing feature values imputation and High cardinality feature detection. In comarison with SKlearn pipeline, we only built one model and tried to tuned it's hyperparameters. Although the difference in accuracy is very small, but AutoML covers plenty of models & preprocessing steps which were not included in our SKlearn pipeline.

## Future work
Although our simple logistic regression model was as effective as complex ensemble AutoML model, but we should address the concern of Class Imbalances in order to improve our model. Other sampling techniques like BayesianParameterSampling & GridParameterSampling, along with termination policies like MedianStoppingPolicy & TruncationSelectionPolicy could also be tried to improve existing models.

In order to improve the performance of our Automl model we could focus on following aspects:
* Dataset : We could modify our train/test dataset with a much enriched data.
* Optimization Metric : As we have choosed our **"primary_metric" : "AUC_weighted"**, we could try modifying with primary_metric along with other metric as well like,   "n_cross_validations" etc.
* Constraint Time/Cost : Try modifying with these constraints "experiment_timeout_minutes", "max_concurrent_iterations", "iteration_timeout_minutes" etc.
* Deep Learning Activation: While selecting Automl run - Classification, we could also choose **"Deep Learning Models"** to train in order to further optimize.

