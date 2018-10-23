# Capgemini Data Science Challenge
Ranked 13 on the leaderboard
https://techchallenge.in.capgemini.com/techchallenge/data-science?leaderboard=true

## Problem Statement
- The motivation behind the problem is to optimize the People Supply Chain management for InterstellarX Inc
- We have to Build a Predictive Demand Model which can Forecast the Demand for next two months.
- We need to plan the optimized supply needed per month for the next 12 months based upon demand forecast.
- We also need to showcase the Net Profit or loss of business if variable factors are changed and demand & supply changes as a consequence.

## Constraint And Assumptions
- Total budget for maintaining the Bench for the current year is $5.76 Mn.
- Average cost per resource per month is $685.
- Current Bench strength is 400, means annual Bench budget consumption is at $3.288 Mn on day 1 of the year.
- End of year average Bench cost cannot exceed total budget of $5,76 Mn
- Average annual attrition is 20% of total headcount and Total headcount at the beginning of the year is 10000 and cannot exceed 12000 at the end of the year.
- Once billed a resource stays with the same account forever and does not come back into the bench or move to any other project.

## Implementation Overview
 - Problem Understanding by Extensive Exploratory data analysis in Tableau as well as in Python.
 - Aggregate the records on Monthly basis for demand and headcount dataset.
 - Used monthly aggregate headcount data from 2004 to 2015 for training the model and 2016 demand data for testing purpose.
 - Perform Dickey-Fuller  test to check Stationarity in time series further perform the Decomposition of signal into Trend, Seasonality and Residual.
 - Applied Time series models like Arima and Prophet to forecast the demand.
 - Used the forecasted demand to plan the supply of resources.


## Stack Used
- Tableau
- Fbprophet
- Plotly
- Statsmodels
- IBM Watson Studio
