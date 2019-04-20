# HE_Flight_Pred
Solution for HackerEarth ML Codesprint - Euristica'19

## About

Problem statement and data can be dowloaded from the competition site
https://www.hackerearth.com/challenges/college/cognitia19/

## Solution
- Solution was the ensemble of H2OAutoML, with StackingRegressor of lightgbm,xgboost and rgf models for Question number-1.
- For Dataset-2 a simple addition interaction of all variables and a VotingClassifier on lightgbm and xgboost model with output rouding off gave the highest accuracy.
- For 3rd question we extracted the names,skills using the NLTk named entity recognition.and phone number and email using the regular expression.

## Requirement
- lightgbm
- H2OAutoML
- sklearn
- xgboost
- bayes_opt
