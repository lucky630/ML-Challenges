# Quartic Machine Learning Challenge

## Objective
Have to build the most accurate model which can predict target column for data_test.csv. 

1. The column details are below:
> * id: id column for data_train, data_test, respectively
> * num*: numerical features
> * der*: derived features from other features
> * cat*: categorical features
> * target: target column, only exists in data_train. it is binary.
2. There are potentially missing values in each column. The goal is to predict target column for data_test.csv.The solution should have a result csv file with two columns:
> * 'id': the id column from data_test.csv
> * 'target': the predicted probability of target being 1


## Approach Discussion
Q.1:- Briefly describe the conceptual approach you chose! What are the trade-offs?
> First part of the choosen model is a gradient boosting ensemble model because decision boundary between two different classes are not Linearly separable.only Non-linear model will perform better in this case.VotingClassifier using two different Gradient Boosting packages (Xgboost,LightGbm) is the first part of the solution.
The second part is the same VotingClassifier with xgboost and lightgbm upon the dataset obtained from Denoising Auto Encoder which is chosen because the original train-test distribution is same and test set have more records then trainset.
The final solution is the weighted Average of both the models where weights were decided based upon Cross validation score of Traiset.
Trade-offs:- Have used only the Gradient Boosting models in the final submission this could cause higher correlation of models in the ensemble which can lead to overfitting the trainSet.

Q.2:- What's the model performance? What is the complexity? Where are the bottlenecks?
> Roc_Auc have choose to measure the model Performance.Accuracy can't be used because of Unbalanced class count in the dataset.So we are getting 64% Roc_Auc score for 3 fold cv for First model of Ensemble and 63% Accuracy on Second Model with weighted Average the Score went to 65%.
Because of Ensemble of 4 different predictive models It is difficult for the interpretability of the model.the Runtime also more than single model.


Q.3:- If you had more time, what improvements would you make, and in what order of priority?
> 1. More feature generation by feature interaction and removing the unuseful features by keeping the useful one.
> 2. Data Imputation by using predictive modelling instead of Mode.Features like num18 have higher feature importance but have lot of Null values.so,carefully filling these null values can result in accuracy increase.
> 3. Tune the Xgboost and Dae Models on different Parameter space.Running the Dae Network for More number of Epochs or changing the Network architecture can also help.
> 4. Doing Two Level Stacking with different and diverse Models.
