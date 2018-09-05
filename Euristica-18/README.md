# HE_Flight_Pred
Solution for HackerEarth ML Codesprint - Euristica'18

## About

Problem statement and data can be dowloaded from the competition site
https://www.hackerearth.com/challenge/college/euristica-ml/

## Solution
use all the newly created features from below and train different models on that.

tuned the model using bayesian optimization and ensemble those model into final submission 


## Newly created Features and findings

- with heights pressure decrease. so get the differenece bet pressure at height 45 and height 1. pressure should always decrease with height increase.
- high speed winds- max of the wind speed in 45 heights.
- dew point temperature can never be higher than air temperature.
- wind speed should be between 5 to 14 km/hr for pleasant weather.
- high pressure provide higher effeciency.
- 100% humidity doesn't mean rain will happen.
- Take dew point and temperature as a ratio and you will get humidity as percentage.humidity is amount of water vapor present in air relative to what the air can hold.
- Get the max,min,standard deviation,variance,mean and difference between top and lowerheight parameters for all the conditions
- Total sum for all the total flights in 289 spots

## Requirement
- lightgbm
- keras
- sklearn
- xgboost
- catboost
- bayes_opt
