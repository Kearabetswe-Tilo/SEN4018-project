import numpy as np 
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt

energy_dataset_df = pd.read_csv("energy_dataset.csv")

print(energy_dataset_df.isnull().sum())
print("---------------------------------------------")
energy_dataset_describe = energy_dataset_df.describe()
print(energy_dataset_describe)
print("********************************************")
df_energy = energy_dataset_df
#replace 0/0.0 with nan
energy_dataset_df = energy_dataset_df.replace(0, np.nan) 
#print column names with nan
print(energy_dataset_df.isnull().sum())
print([col for col in energy_dataset_df.columns if energy_dataset_df[col].isnull().any()])
print(energy_dataset_df.isnull().sum().sum())

#data cleaning
#----------------------------------------------------------------------------------------

print(energy_dataset_df['generation biomass'])

#REMOVING NAN
df_temp = energy_dataset_df[energy_dataset_df['generation biomass'].notnull()]
print(df_temp)
print(df_temp.isnull().sum())

#print column types
print(energy_dataset_df.dtypes)

#mean of generation biomass
print(df_temp['generation biomass'].astype(int).mean())

generation_biomass_mean = df_temp['generation biomass'].astype(int).mean()

#impute mean for generation biomass
energy_dataset_df['generation biomass'] = energy_dataset_df['generation biomass'].replace(np.nan, generation_biomass_mean).astype(int)
print(energy_dataset_df.isnull().sum())

#data cleaning
#------------------------------------------------------------------------------------------------------------------------------------------------

print(energy_dataset_df['generation fossil gas'])

#REMOVING NAN
df_temp = energy_dataset_df[energy_dataset_df['generation fossil gas'].notnull()]
print(df_temp)
print(df_temp.isnull().sum())

#mean of generation fossil gas
print(df_temp['generation fossil gas'].astype(int).mean())

generation_fossil_gas_mean = df_temp['generation fossil gas'].astype(int).mean()

#impute mean for generation fossil gas
energy_dataset_df['generation fossil gas'] = energy_dataset_df['generation fossil gas'].replace(np.nan, generation_fossil_gas_mean).astype(int)
print(energy_dataset_df.isnull().sum())

#data cleaning
#---------------------------------------------------------------------------------------------------------------------------------------------------
print(energy_dataset_df['generation fossil hard coal'])

#REMOVING NAN
df_temp = energy_dataset_df[energy_dataset_df['generation fossil hard coal'].notnull()]
print(df_temp)
print(df_temp.isnull().sum())

#mean of generation fossil hard coal
print(df_temp['generation fossil hard coal'].astype(int).mean())

generation_fossil_hard_coal_mean = df_temp['generation fossil hard coal'].astype(int).mean()

#impute mean for generation fossil hard coal
energy_dataset_df['generation fossil hard coal'] = energy_dataset_df['generation fossil hard coal'].replace(np.nan, generation_fossil_hard_coal_mean).astype(int)
print(energy_dataset_df.isnull().sum())

#data cleasing
#--------------------------------------------------------------------------------------------------------------------------------------------------

print(energy_dataset_df['generation fossil oil'])

#REMOVING NAN
df_temp = energy_dataset_df[energy_dataset_df['generation fossil oil'].notnull()]
print(df_temp)
print(df_temp.isnull().sum())

#mean of generation fossil oil
print(df_temp['generation fossil oil'].astype(int).mean())

generation_fossil_oil_mean = df_temp['generation fossil oil'].astype(int).mean()

#impute mean for generation fossil oil
energy_dataset_df['generation fossil oil'] = energy_dataset_df['generation fossil oil'].replace(np.nan, generation_fossil_oil_mean).astype(int)
print(energy_dataset_df.isnull().sum())

#data cleasing
#--------------------------------------------------------------------------------------------------------------------------------------------------

print(energy_dataset_df['generation hydro run-of-river and poundage'])

#REMOVING NAN
df_temp = energy_dataset_df[energy_dataset_df['generation hydro run-of-river and poundage'].notnull()]
print(df_temp)
print(df_temp.isnull().sum())

#mean of generation hydro run-of-river and poundage
print(df_temp['generation hydro run-of-river and poundage'].astype(int).mean())

generation_hydro_run_of_river_and_poundage_mean = df_temp['generation hydro run-of-river and poundage'].astype(int).mean()

#impute mean for generation hydro run-of-river and poundage
energy_dataset_df['generation hydro run-of-river and poundage'] = energy_dataset_df['generation hydro run-of-river and poundage'].replace(np.nan, generation_hydro_run_of_river_and_poundage_mean).astype(int)
print(energy_dataset_df.isnull().sum())

#data cleasing
#--------------------------------------------------------------------------------------------------------------------------------------------------

print(energy_dataset_df['generation hydro water reservoir'])

#REMOVING NAN
df_temp = energy_dataset_df[energy_dataset_df['generation hydro water reservoir'].notnull()]
print(df_temp)
print(df_temp.isnull().sum())

#mean of generation hydro water reservoire
print(df_temp['generation hydro water reservoir'].astype(int).mean())

generation_hydro_water_reservoir_mean = df_temp['generation hydro water reservoir'].astype(int).mean()

#impute mean for generation hydro water reservoir
energy_dataset_df['generation hydro water reservoir'] = energy_dataset_df['generation hydro water reservoir'].replace(np.nan, generation_hydro_water_reservoir_mean).astype(int)
print(energy_dataset_df.isnull().sum())

#data cleasing
#--------------------------------------------------------------------------------------------------------------------------------------------------

print(energy_dataset_df['generation nuclear'])

#REMOVING NAN
df_temp = energy_dataset_df[energy_dataset_df['generation nuclear'].notnull()]
print(df_temp)
print(df_temp.isnull().sum())

#mean of generation nuclear
print(df_temp['generation nuclear'].astype(int).mean())

generation_nuclear_mean = df_temp['generation nuclear'].astype(int).mean()

#impute mean for generation nuclear
energy_dataset_df['generation nuclear'] = energy_dataset_df['generation nuclear'].replace(np.nan, generation_nuclear_mean).astype(int)
print(energy_dataset_df.isnull().sum())

#data cleasing
#--------------------------------------------------------------------------------------------------------------------------------------------------

print(energy_dataset_df['generation other'])

#REMOVING NAN
df_temp = energy_dataset_df[energy_dataset_df['generation other'].notnull()]
print(df_temp)
print(df_temp.isnull().sum())

#mean of generation other
print(df_temp['generation nuclear'].astype(int).mean())

generation_other_mean = df_temp['generation other'].astype(int).mean()

#impute mean for generation other
energy_dataset_df['generation other'] = energy_dataset_df['generation other'].replace(np.nan, generation_other_mean).astype(int)
print(energy_dataset_df.isnull().sum())
 
#data cleasing
#--------------------------------------------------------------------------------------------------------------------------------------------------

print(energy_dataset_df['generation other renewable'])

#REMOVING NAN
df_temp = energy_dataset_df[energy_dataset_df['generation other renewable'].notnull()]
print(df_temp)
print(df_temp.isnull().sum())

#mean of generation other renewable
print(df_temp['generation other renewable'].astype(int).mean())

generation_other_renewable_mean = df_temp['generation other renewable'].astype(int).mean()

#impute mean for generation other renewable
energy_dataset_df['generation other renewable'] = energy_dataset_df['generation other renewable'].replace(np.nan, generation_other_renewable_mean).astype(int)
print(energy_dataset_df.isnull().sum())

#data cleasing
#--------------------------------------------------------------------------------------------------------------------------------------------------

print(energy_dataset_df['generation solar'])

#REMOVING NAN
df_temp = energy_dataset_df[energy_dataset_df['generation solar'].notnull()]
print(df_temp)
print(df_temp.isnull().sum())

#mean of generation solar
print(df_temp['generation solar'].astype(int).mean())

generation_solar_mean = df_temp['generation solar'].astype(int).mean()

#impute mean for generation solar
energy_dataset_df['generation solar'] = energy_dataset_df['generation solar'].replace(np.nan, generation_solar_mean).astype(int)
print(energy_dataset_df.isnull().sum())

#data cleasing
#--------------------------------------------------------------------------------------------------------------------------------------------------

print(energy_dataset_df['generation waste'])

#REMOVING NAN
df_temp = energy_dataset_df[energy_dataset_df['generation waste'].notnull()]
print(df_temp)
print(df_temp.isnull().sum())

#mean of generation waste
print(df_temp['generation waste'].astype(int).mean())

generation_waste_mean = df_temp['generation waste'].astype(int).mean()

#impute mean for generation waste
energy_dataset_df['generation waste'] = energy_dataset_df['generation waste'].replace(np.nan, generation_waste_mean).astype(int)
print(energy_dataset_df.isnull().sum())

#data cleasing
#--------------------------------------------------------------------------------------------------------------------------------------------------

print(energy_dataset_df['generation wind onshore'])

#REMOVING NAN
df_temp = energy_dataset_df[energy_dataset_df['generation wind onshore'].notnull()]
print(df_temp)
print(df_temp.isnull().sum())

#mean of generation wind onshore
print(df_temp['generation wind onshore'].astype(int).mean())

generation_wind_onshore_mean = df_temp['generation wind onshore'].astype(int).mean()

#impute mean for generation wind onshore
energy_dataset_df['generation wind onshore'] = energy_dataset_df['generation wind onshore'].replace(np.nan, generation_wind_onshore_mean).astype(int)
print(energy_dataset_df.isnull().sum())

#data cleasing
#--------------------------------------------------------------------------------------------------------------------------------------------------

print(energy_dataset_df['forecast solar day ahead'])

#REMOVING NAN
df_temp = energy_dataset_df[energy_dataset_df['forecast solar day ahead'].notnull()]
print(df_temp)
print(df_temp.isnull().sum())

#mean of forecast solar day ahead
print(df_temp['forecast solar day ahead'].astype(int).mean())

forecast_solar_day_ahead_mean = df_temp['forecast solar day ahead'].astype(int).mean()

#impute mean for forecast solar day ahead
energy_dataset_df['forecast solar day ahead'] = energy_dataset_df['forecast solar day ahead'].replace(np.nan, forecast_solar_day_ahead_mean).astype(int)
print(energy_dataset_df.isnull().sum())

#data cleasing
#--------------------------------------------------------------------------------------------------------------------------------------------------

print(energy_dataset_df['total load actual'])

#REMOVING NAN
df_temp = energy_dataset_df[energy_dataset_df['total load actual'].notnull()]
print(df_temp)
print(df_temp.isnull().sum())

#mean of total load actual
print(df_temp['total load actual'].astype(int).mean())

total_load_actual_mean = df_temp['total load actual'].astype(int).mean()

#impute mean for total load actual
energy_dataset_df['total load actual'] = energy_dataset_df['total load actual'].replace(np.nan, total_load_actual_mean).astype(int)
print(energy_dataset_df.isnull().sum())

# droping columns
#-----------------------------------------------------------------------------------------------------
"""
energy_dataset_df.drop('generation fossil brown coal/lignite', axis=1, inplace=True)
energy_dataset_df.drop('generation fossil coal-derived gas', axis=1, inplace=True)
energy_dataset_df.drop('generation fossil oil shale', axis=1, inplace=True)
energy_dataset_df.drop('generation fossil peat', axis=1, inplace=True)
energy_dataset_df.drop('generation geothermal', axis=1, inplace=True)
energy_dataset_df.drop('generation hydro pumped storage aggregated', axis=1, inplace=True)
energy_dataset_df.drop('generation hydro pumped storage consumption', axis=1, inplace=True)
energy_dataset_df.drop('generation marine', axis=1, inplace=True)
energy_dataset_df.drop('generation wind offshore', axis=1, inplace=True)
energy_dataset_df.drop('forecast wind offshore eday ahead', axis=1, inplace=True)
print(energy_dataset_df.isnull().sum())
"""
#Energy heatmap
corr = energy_dataset_df.corr()
plt.figure(figsize = (20, 10))
a = sb.heatmap(corr, annot = True, fmt = '.2f')
a.set_ylim(0, 10)


#histograms
plt.figure(figsize=(10,10))
energy_dataset_df[['generation biomass', 'generation fossil gas', 'generation fossil hard coal', 'generation fossil oil']].hist(figsize = (10, 8), bins = 5, color = 'gray')
plt.tight_layout()
plt.show()

#histograms
plt.figure(figsize=(10,10))
energy_dataset_df[['generation hydro run-of-river and poundage', 'generation hydro water reservoir', 'generation nuclear', 'generation other']].hist(figsize = (10, 8), bins = 5, color = 'gray')
plt.tight_layout()
plt.show()

#histograms
plt.figure(figsize=(10,10))
energy_dataset_df[['generation other renewable', 'generation solar', 'generation waste', 'generation wind onshore']].hist(figsize = (10, 8), bins = 5, color = 'gray')
plt.tight_layout()
plt.show()

#histograms
plt.figure(figsize=(10,10))
energy_dataset_df[['forecast solar day ahead', 'forecast wind onshore day ahead']].hist(figsize = (10, 8), bins = 5, color = 'gray')
plt.tight_layout()
plt.show()

#histograms
plt.figure(figsize=(10,10))
energy_dataset_df[['total load forecast', 'total load actual']].hist(figsize = (10, 8), bins = 5, color = 'gray')
plt.tight_layout()
plt.show()

#histograms
plt.figure(figsize=(10,10))
energy_dataset_df[['price day ahead', 'price actual']].hist(figsize = (10, 8), bins = 5, color = 'gray')
plt.tight_layout()
plt.show()

#scatter plots
plt.figure()
plt.scatter(energy_dataset_df['generation solar'], energy_dataset_df['forecast solar day ahead'])
plt.xlabel('Generation Solar')
plt.ylabel('Forecast Solar Day Ahead')

#scatter plots
plt.figure()
plt.scatter(energy_dataset_df['generation fossil hard coal'], energy_dataset_df['price day ahead'])
plt.xlabel('Generation Fossil Hard Coal')
plt.ylabel('Price Day Ahead')

#scatter plots
plt.figure()
plt.scatter(energy_dataset_df['generation fossil hard coal'], energy_dataset_df['price actual'])
plt.xlabel('Generation Fossil Hard Coal')
plt.ylabel('Price Actual')

#scatter plots
plt.figure()
plt.scatter(energy_dataset_df['generation fossil gas'], energy_dataset_df['price day ahead'])
plt.xlabel('Generation Fossil Gas')
plt.ylabel('Price Day Ahead')

#scatter plots
plt.figure()
plt.scatter(energy_dataset_df['generation fossil gas'], energy_dataset_df['price actual'])
plt.xlabel('Generation Fossil Gas')
plt.ylabel('Price Actual')

#scatter plots
plt.figure()
plt.scatter(energy_dataset_df['generation fossil oil'], energy_dataset_df['total load forecast'])
plt.xlabel('Generation Fossil Oil')
plt.ylabel('Total Load Forecast')

#scatter plots
plt.figure()
plt.scatter(energy_dataset_df['generation fossil oil'], energy_dataset_df['total load actual'])
plt.xlabel('Generation Fossil Oil')
plt.ylabel('Total Load Actual')


# Generation Fossil Hard Coal vs Price Day Ahead

x = energy_dataset_df['generation fossil hard coal'].values #get column Generation Fossil Hard Coal
y = energy_dataset_df['price day ahead'].values #get column Price Day Ahead


x = x.reshape(-1, 1)
y = y.reshape(-1, 1)

#simple linear regression
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x, y)
y_predicted_price_day_ahead = regressor.predict(x)

plt.figure()
plt.scatter(x, y, color = 'red')
plt.plot(x,y_predicted_price_day_ahead, color = 'blue')
plt.title('Generation Fossil Hard Coal vs Price Day Ahead')
plt.xlabel('Generation Fossil Hard Coal')
plt.ylabel('Price Day Ahead')
plt.show()

#finding error
msqe = sum((y_predicted_price_day_ahead - y) * (y_predicted_price_day_ahead - y)) / y.shape[0]
rmse = np.sqrt(msqe)

#-------------- before training ----------------------------------------------
print(msqe)
print(rmse)



#split dataset into train and test splits
from sklearn.model_selection import train_test_split 
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 1/3, random_state = 0)

#fit simple linear regression model on training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

y_pred = regressor.predict(x_test)

msqe = sum((y_pred - y_test) * (y_pred - y_test)) / y.shape[0]
rmse = np.sqrt(msqe)

#After training 
print("Accuracy of a model")
print("*****************************************")
print("mean square error:")
print(msqe)
print("-----------------------------------------")
print("Root mean sqaure error:")
print(rmse)
#-------------------------------------------------------------------

#Random Forest for coal and price day ahead
x = energy_dataset_df['generation fossil hard coal'].values #get column Generation Fossil Hard Coal
y = energy_dataset_df['price day ahead'].values #get column Price Day Ahead

x = x.reshape(-1, 1)
#y = y.reshape(-1, 1)


#random forest regression
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 100, random_state = 0)
regressor.fit(x, y)

y_pred_Random_Forest = regressor.predict(x)

#sort x
s_x = np.sort(x, axis = None).reshape(-1, 1)

plt.figure()
x_grid = np.arange(min(s_x), max(s_x), 0.01)
x_grid = x_grid.reshape((len(x_grid), 1))
plt.scatter(x, y, color = 'red')
plt.plot(x_grid, regressor.predict(x_grid), color = 'blue')
plt.title('Generation Fossil Hard Coal vs Price Day Ahead')
plt.xlabel('Generation Fossil Hard Coal')
plt.ylabel('Price Day Ahead')
plt.show()

#finding error
msqe = sum((y_pred_Random_Forest - y) * (y_pred_Random_Forest - y)) / y.shape[0]
rmse = np.sqrt(msqe)
print(msqe)
print(rmse)

#------ training ---------
#split dataset into train and test splits
from sklearn.model_selection import train_test_split 
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 1/3, random_state = 0)

#max depth versus error
md = 20;
md_errors = np.zeros(md)

from sklearn.ensemble import RandomForestRegressor
for i in range(1, md+1):
    regressor = RandomForestRegressor(n_estimators = 100, max_depth = i, random_state = 0)
    regressor.fit(x_train, y_train)
    y_pred = regressor.predict(x_test)
    #finding error
    msqe = sum((y_pred - y_test) * (y_pred - y_test)) / y_test.shape[0]
    md_errors[i-1] = np.sqrt(msqe)


plt.scatter(range(1, md+1), md_errors, color = 'red')
plt.plot(range(1, md+1), md_errors, color = 'blue')
plt.xlabel('max depth')
plt.ylabel('rmse')
plt.show()


#more hyperparameters
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 100, max_depth = 5, max_features = 0.5, min_samples_split = 5, random_state = 0)
regressor.fit(x_train, y_train)
y_pred = regressor.predict(x_test)
msqe = sum((y_pred - y_test) * (y_pred - y_test)) / y_test.shape[0]
rmse = np.sqrt(msqe)
print(msqe)
print(rmse)


#-----------------------------------------------------------------------

#polynomial linear regression
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 2)
x_poly = poly_reg.fit_transform(x)
lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly, y)

#sort x
s_x = np.sort(x, axis=None).reshape(-1, 1)

#visualize polynomial linear regression
plt.scatter(x, y, color = 'red')
plt.plot(s_x, lin_reg2.predict(poly_reg.fit_transform(s_x)), color = 'blue')
plt.title('Generation Fossil Hard Coal vs Total Load Actual')
plt.xlabel('Generation Fossil Hard Coal')
plt.ylabel('Total Load Actual')
plt.show()

#------------------------------------------------------------------------
# Generation Solar vs Forecast Solar Day Ahead
x = energy_dataset_df['generation solar'].values 
y = energy_dataset_df['forecast solar day ahead'].values 

x = x.reshape(-1, 1)
y = y.reshape(-1, 1)

#simple linear regression
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(x, y)

y_pred = lin_reg.predict(x)

#visualize simple linear regression
plt.figure()
plt.scatter(x, y, color = 'red')
plt.plot(x, y_pred, color = 'blue')
plt.title('generation solar vs forecast solar day ahead')
plt.xlabel('generation solar')
plt.ylabel('forecast solar day ahead')
plt.show()

#finding error
msqe = sum((y_pred - y) * (y_pred - y)) / y.shape[0]
rmse = np.sqrt(msqe)
print("simple linear regression error evaluation")
print("-----------------------------------------")
print(msqe)
print("*****************************************")
print(rmse)

#-----traning ------------------------
#split dataset into train and test splits
from sklearn.model_selection import train_test_split 
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 1/3, random_state = 0)

#fit simple linear regression model on training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

y_pred = regressor.predict(x_test)

msqe = sum((y_pred - y_test) * (y_pred - y_test)) / y.shape[0]
rmse = np.sqrt(msqe)

#After training 
print("Accuracy of a model")
print("*****************************************")
print("mean square error:")
print(msqe)
print("-----------------------------------------")
print("Root mean sqaure error:")
print(rmse)

# Generation Fossil Hard Coal vs Price Actual

x = energy_dataset_df['generation fossil hard coal'].values 
y = energy_dataset_df['price actual'].values 

x = x.reshape(-1, 1)
y = y.reshape(-1, 1)

#simple linear regression
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(x, y)

y_pred_coal_price01 = lin_reg.predict(x)

#visualize simple linear regression
plt.figure()
plt.scatter(x, y, color = 'red')
plt.plot(x, y_pred_coal_price01, color = 'blue')
plt.title('Generation Fossil Hard Coal vs Price Actual')
plt.xlabel('Generation Fossil Hard Coal')
plt.ylabel('Price Actual')
plt.show()

#finding error
msqe = sum((y_pred_coal_price01 - y) * (y_pred_coal_price01 - y)) / y.shape[0]
rmse = np.sqrt(msqe)
print("simple linear regression error evaluation")
print("-----------------------------------------")
print(msqe)
print("*****************************************")
print(rmse)

#split dataset into train and test splits
from sklearn.model_selection import train_test_split 
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 1/3, random_state = 0)

#fit simple linear regression model on training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

y_pred_coal_price02 = regressor.predict(x_test)

msqe = sum((y_pred_coal_price02 - y_test) * (y_pred_coal_price02 - y_test)) / y.shape[0]
rmse = np.sqrt(msqe)

#After training 
print("Accuracy of a model")
print("*****************************************")
print("mean square error:")
print(msqe)
print("-----------------------------------------")
print("Root mean sqaure error:")
print(rmse)

#polynomial linear regression
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
x_poly = poly_reg.fit_transform(x)
lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly, y)

#sort x
s_x = np.sort(x, axis=None).reshape(-1, 1)

#visualize polynomial linear regression
plt.scatter(x, y, color = 'red')
plt.plot(s_x, lin_reg2.predict(poly_reg.fit_transform(s_x)), color = 'blue')
plt.title('Generation Fossil Hard Coal vs Price Actual')
plt.xlabel('Generation Fossil Hard Coal')
plt.ylabel('Price Actual')
plt.show()

y_pred_polynomial_coal_price2 = lin_reg2.predict(x_poly)

#finding error
msqe = sum((y_pred_polynomial_coal_price2 - y) * (y_pred_polynomial_coal_price2 - y)) / y.shape[0]
rmse = np.sqrt(msqe)
print("simple linear regression error evaluation")
print("-----------------------------------------")
print(msqe)
print("*****************************************")
print(rmse)

#split dataset into train and test splits
from sklearn.model_selection import train_test_split 
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 1/3, random_state = 0)

#fit simple linear regression model on training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

y_pred_coal_price03 = regressor.predict(x_test)

msqe = sum((y_pred_coal_price03 - y_test) * (y_pred_coal_price03 - y_test)) / y.shape[0]
rmse = np.sqrt(msqe)

#After training 
print("Accuracy of a model")
print("*****************************************")
print("mean square error:")
print(msqe)
print("-----------------------------------------")
print("Root mean sqaure error:")
print(rmse)

# Weather features dataset 

weather_features_df = pd.read_csv("weather_features.csv")

print(weather_features_df.isnull().sum())

print(weather_features_df.isnull().sum())
print("---------------------------------------------")
weather_features_describe = weather_features_df.describe()
print(weather_features_describe)
print("********************************************")
df_weather = weather_features_df

#replace 0/0.0 with nan
weather_features_df = weather_features_df.replace(0, np.nan)

#print column names with nan
print(weather_features_df.isnull().sum())
print([col for col in weather_features_df.columns if weather_features_df[col].isnull().any()])
print(weather_features_df.isnull().sum().sum())

#data cleaning
#----------------------------------------------------------------------------------------

print(weather_features_df['humidity'])

#REMOVING NAN
df_temp = weather_features_df[weather_features_df['humidity'].notnull()]
print(df_temp)
print(df_temp.isnull().sum())

#print column types
print(weather_features_df.dtypes)

#mean of humidity
print(df_temp['humidity'].astype(int).mean())

humidity_mean = df_temp['humidity'].astype(int).mean()

#impute mean for humidity
weather_features_df['humidity'] = weather_features_df['humidity'].replace(np.nan, humidity_mean).astype(int)
print(weather_features_df.isnull().sum())

#data cleaning
#----------------------------------------------------------------------------------------

print(weather_features_df['pressure'])

#REMOVING NAN
df_temp = weather_features_df[weather_features_df['pressure'].notnull()]
print(df_temp)
print(df_temp.isnull().sum())

#print column types
print(weather_features_df.dtypes)

#mean of pressure
print(df_temp['pressure'].astype(int).mean())

pressure_mean = df_temp['pressure'].astype(int).mean()

#impute mean for pressure
weather_features_df['pressure'] = weather_features_df['pressure'].replace(np.nan, pressure_mean).astype(int)
print(weather_features_df.isnull().sum())

# droping columns
#-----------------------------------------------------------------------------------------------
"""
weather_features_df.drop('wind_speed', axis=1, inplace=True)
weather_features_df.drop('wind_deg', axis=1, inplace=True)
weather_features_df.drop('rain_1h', axis=1, inplace=True)
weather_features_df.drop('rain_3h', axis=1, inplace=True)
weather_features_df.drop('snow_3h', axis=1, inplace=True)
weather_features_df.drop('clouds_all', axis=1, inplace=True)
print(weather_features_df.isnull().sum())
"""
#heatmap
corr = weather_features_df.corr()
plt.figure(figsize = (20, 10))
a = sb.heatmap(corr, annot = True, fmt = '.2f')
a.set_ylim(0, 10)

# merging energy dataset and waether dataset

new_energy_dataset_df = energy_dataset_df

energy_weather = new_energy_dataset_df.merge(weather_features_df, left_on='time',right_on='time')

#heatmap
corr = energy_weather.corr()
plt.figure(figsize = (20, 10))
a = sb.heatmap(corr, annot = True, fmt = '.2f')
a.set_ylim(0, 10)

#scatter plots for generation solar and temperature
plt.figure()
plt.scatter(energy_weather['generation solar'], energy_weather['temp'])
plt.xlabel('Generation Solar')
plt.ylabel('Temperature')

#scatter plots for generation solar and mimimum temperature
plt.figure()
plt.scatter(energy_weather['generation solar'], energy_weather['temp_min'])
plt.xlabel('Generation Solar')
plt.ylabel('Minimum Temperature')

#scatter plots for generation solar and maximum temperature
plt.figure()
plt.scatter(energy_weather['generation solar'], energy_weather['temp_max'])
plt.xlabel('Generation Solar')
plt.ylabel('Maximum Temperature')

# generation solar vs temperature

x = energy_weather['generation solar'].values 
y = energy_weather['temp'].values

x = x.reshape(-1, 1)
y = y.reshape(-1, 1)

#decision tree regression
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 10, max_depth = 200) 
regressor.fit(x, y)

y_pred = regressor.predict(x)

#sort x
s_x = np.sort(x, axis = None).reshape(-1, 1)

plt.figure()
x_grid = np.arange(min(s_x), max(s_x), 0.01)
x_grid = x_grid.reshape((len(x_grid), 1))
plt.scatter(x, y, color = 'red')
plt.plot(x_grid, regressor.predict(x_grid), color = 'blue')
plt.title('Generation Solar vs Temperature')
plt.xlabel('Generation Solar')
plt.ylabel(' Temperature')
plt.show()

#split dataset into train and test splits
from sklearn.model_selection import train_test_split 
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 1/3, random_state = 0)

#decision tree regression
from sklearn.tree import DecisionTreeRegressor
regressor1 = DecisionTreeRegressor(max_depth = 100, random_state = 0) 
regressor1.fit(x_train, y_train)


y_pred = regressor1.predict(x_test)
"""
#finding error
msqe = sum((y_pred - y_test) * (y_pred - y_test)) / y_test.shape[0].astype(np.uint8)
rmse = np.sqrt(msqe)
"""
accuracy = regressor1.score(x_test, y_test)
print("Accuracy : {}%".format(int(round(accuracy*100))))


# generation fossil gas and price actual

x = energy_weather['generation fossil gas'].values 
y = energy_weather['price actual'].values

x = x.reshape(-1, 1)
y = y.reshape(-1, 1)

#decision tree regression
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0, max_depth = 200) 
regressor.fit(x, y)

y_pred = regressor.predict(x)

#sort x
s_x = np.sort(x, axis = None).reshape(-1, 1)

plt.figure()
x_grid = np.arange(min(s_x), max(s_x), 0.01)
x_grid = x_grid.reshape((len(x_grid), 1))
plt.scatter(x, y, color = 'red')
plt.plot(x_grid, regressor.predict(x_grid), color = 'blue')
plt.title('generation fossil gas vs price actual')
plt.xlabel('Generation Fossil Gas')
plt.ylabel('Price Actual')
plt.show()

#split dataset into train and test splits
from sklearn.model_selection import train_test_split 
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 1/3, random_state = 0)

#decision tree regression
from sklearn.tree import DecisionTreeRegressor
regressor1 = DecisionTreeRegressor(max_depth = 100, random_state = 0) 
regressor1.fit(x_train, y_train)

y_pred = regressor1.predict(x_test)

accuracy = regressor1.score(x_test, y_test)
print("Accuracy : {}%".format(int(round(accuracy*100))))

#-----------------------------------------------------------------------------------------------------------
#Random Forest generation fossil oil and total load actual 
x = energy_dataset_df['generation fossil oil'].values #get column generation fossil oil
y = energy_dataset_df['total load actual'].values #get column total load actual

x = x.reshape(-1, 1)
#y = y.reshape(-1, 1)

#random forest regression
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 100, random_state = 0)
regressor.fit(x, y)

y_pred_Random_Forest_oil_load = regressor.predict(x)

#sort x
s_x = np.sort(x, axis = None).reshape(-1, 1)

plt.figure()
x_grid = np.arange(min(s_x), max(s_x), 0.01)
x_grid = x_grid.reshape((len(x_grid), 1))
plt.scatter(x, y, color = 'red')
plt.plot(x_grid, regressor.predict(x_grid), color = 'blue')
plt.title('Generation Fossil Oil vs Total Load Actual')
plt.xlabel('Generation Fossil Oil')
plt.ylabel('Total Load Actual')
plt.show()

#------ training ----------------------------------
#split dataset into train and test splits
from sklearn.model_selection import train_test_split 
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 1/3, random_state = 0)

#more hyperparameters
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 200, max_depth = 10, max_features = 0.5, min_samples_split = 5, random_state = 0)
regressor.fit(x_train, y_train)

y_pred_Random_Forest_oil_load01 = regressor.predict(x_test)

msqe = sum((y_pred_Random_Forest_oil_load01 - y_test) * (y_pred_Random_Forest_oil_load01 - y_test)) / y_test.shape[0]
rmse = np.sqrt(msqe)
#After training 
print("Accuracy of a model")
print("*****************************************")
print("mean square error:")
print(msqe)
print("-----------------------------------------")
print("Root mean sqaure error:")
print(rmse)

#-----------------------------------------------------------------------------------------------------------
#Random Forest generation fossil oil and total load forecast
x = energy_dataset_df['generation fossil oil'].values #get column generation fossil oil
y = energy_dataset_df['total load forecast'].values #get column total load forecast'

x = x.reshape(-1, 1)
#y = y.reshape(-1, 1)

#random forest regression
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 100, random_state = 0)
regressor.fit(x, y)

y_pred_Random_Forest_oil_load = regressor.predict(x)

#sort x
s_x = np.sort(x, axis = None).reshape(-1, 1)

plt.figure()
x_grid = np.arange(min(s_x), max(s_x), 0.01)
x_grid = x_grid.reshape((len(x_grid), 1))
plt.scatter(x, y, color = 'red')
plt.plot(x_grid, regressor.predict(x_grid), color = 'blue')
plt.title('Generation Fossil Oil vs Total Load Forecast')
plt.xlabel('Generation Fossil Oil')
plt.ylabel('Total Load Forecast')
plt.show()

#------ training ----------------------------------
#split dataset into train and test splits
from sklearn.model_selection import train_test_split 
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 1/3, random_state = 0)

#more hyperparameters
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 100, max_depth = 5, max_features = 0.5, min_samples_split = 5, random_state = 0)
regressor.fit(x_train, y_train)

y_pred_Random_Forest_oil_forecast = regressor.predict(x_test)

msqe = sum((y_pred_Random_Forest_oil_forecast - y_test) * (y_pred_Random_Forest_oil_forecast - y_test)) / y_test.shape[0]
rmse = np.sqrt(msqe)
#After training 
print("Accuracy of a model")
print("*****************************************")
print("mean square error:")
print(msqe)
print("-----------------------------------------")
print("Root mean sqaure error:")
print(rmse)


