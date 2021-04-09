# SEN4018-project
Data science project- Analysis of electrical consumption, generation, pricing, and weather data in Spain

Dataset
This dataset spans for a period of 4 years.The consumption and generation
data was retrieved from ENTSOE a public portal for Transmission Service Operator (TSO) data. 
Settlement prices were obtained from the Spanish TSO Red Electric España.
Weather data was purchased as part of a personal project from the Open Weather API for the 5 largest cities in Spain and made public here. 
The dataset is unique because it contains hourly data for electrical consumption and 
the respective forecasts by the TSO for consumption and pricing. 

Feature Selection
For all regression models I analyzed the heatmap since it shows the correlation values of the dataset variables.
I selected variables based on their relationship. Because this is for prediction I considered variables that have a positive correlation.
I believe that good features will allow me to most accurately represent the structure of my data and therefore create the best model. 

Data splitting and Modeling
For every model I have chosen the proportion of 70% for the training set and 30% for the test.
To make the regression model better I provide a large training data and to make the error estimate more accurate extra test data is provided. 

 
Linear regression is the next step up after correlation.
According to our data, there is a positive correlation between hard coal and price day ahead. 
Therefore the two variables are linearly related.
So I decided to use the values of generation fossil hard coal to predict price day ahead values. 

A simple linear regression was used for this prediction problem. 
Firstly, we trained my model without splitting our data and I achieved impressive results. 
Then I splitted my data into a training set and a test set to avoid overfitting and to obtain a realistic evaluation of my learned model.
By doing this I got relatively good results. 



