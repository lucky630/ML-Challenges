# WNS Machine Learning Challenge
Finished 7th on Public and 72 on private LeaderBoard

## Things Tried
- Need to do binning of the age features.
- Apply the mean encoding on the feature having text value and frequency more than 3.
- Avg points * no. of comp to get the total points.
- need to weighted sum some of the performance features to get one score for performance.
- count of 0/1 or yes/no in the row.
- mean encoding or median encoding instead of label.
- normalize or scale and then check the distribution it should be same.
- Use the data imputation technque like(mean,median,predictive model imputation) for missing values in (education & previous_year_ratings.)
- Add the (awards_won;KpIs_met & previous_year_rating) features,multiply the avg_training_score and no_of_training to get total training score.
- convert education into number's where mtech>btech>other.
- Remove the recruitment_channel that have no effect on the Target result.
- (age - length_of_service) for gettng the joining age.

## Final Solution Summary
- The missing values in the education is imputed by mode which was the "Bachelor's" & the missing value in the previous_year_rating is imputed by mode as well as by using the predictive modeling.
After Correlation matrix analysis,the features like previous_year_rating,length_of_service,KPIs_met have higher correlation with the Target value.
- The count of promotion vs no promotion is unbalanced.
- Pca on the train set and their target values show overlapping decision boundary which can't be separated by the Linear Models.for this tye of overlapping target values Decision Trees are best.
Different type of scaling on input features giving different distribution of the target values during Pca.

## Newly created Features
- sum_performance = addition of the important factors for the promotion (awards_won;KpIs_met & previous_year_rating).
- Total nmber of training hours = avg_training_score * no_of_training
- recruitment_channel have no impact on the promotion so removed that.
- Apply the Pca on the input set and get the single column which summarized the input features in the 1 Dimension and used that as a new features.helps to improve the score by 0.5 percent.

## Models
- Lightgbm model with parameter tuning with bayesian Optimization with 8 stratifiedKfolds.
- OOF predictions were used for finding the right threshold value.
- For second layer of predictions used the Average of the results from the different lightGbm models trained on different Input Features.
Pca for the first layer predictions and target values shows the linear separable boundary between two target values.so done the Logistic regression for ensembling of the models.

## What didn't worked
1. Linear models and Neural networks gave low scores as compared to the Decision Trees.
2. mean encoding of missing values, one hot encoding of categorical values almost gave the same score on the leaderboard.
3. blindly addition,multiplication and division of features gave low score.
4. additional features creation with Variance threshold gave same score.
5. prediction of missing values using the data from the train and test gave same core on leaderboard.
6. Ensembling didn't work because of highly correlated models in the First Layer(Have only used Lightgbm for the first layer predictions.)
7. Ensembling using Voting Classifier not gave any improvement over the single model.

## Mistakes
1. Should have selected ensembled prediction for the Final submission instead of selecting the single Tuned lightgbm model.
2. Lack of diverse models in the First layer of stacking.
3. Submission based upon public leaderboard score leads to Overfitting and drop of position in the private leaderboard.need better Local validation Strategy.

# Contributors
- [utsav aggarwal](https://github.com/utsav1)
