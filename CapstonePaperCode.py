# -*- coding: utf-8 -*-
"""
Created on Sat Feb 23 08:47:09 2019

@author: Ryan Talk
"""


#Load all the packages needed for this Lab
import pandas as pd
import numpy as np
import seaborn as sns
from IPython.display import HTML
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from matplotlib.dates import WeekdayLocator
from matplotlib.dates import MO
from matplotlib.dates import date2num
from dateutil import rrule
from datetime import datetime

from statsmodels.stats.stattools import durbin_watson
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.stattools import pacf

from statsmodels.regression.linear_model import OLS
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.arima_model import ARMA
from statsmodels.tsa.ar_model import AR


def createPlot(preds, y):
    ''' This will automatically create a scatter plot.  We can add to this function
    If we want to include more graphs of the preds and y.'''
    fig, ax = plt.subplots()
    #ax2 = ax.twiny()
    ax.plot(range(0,len(y)), y, 'o', label="Data")
    ax.plot(range(0,len(preds)), preds, "b-", label="Preds")
    ax.legend(loc="best");
    fig.set_size_inches(10,5)


def printSummary(model_fit):
    ''' This will print out all the stats.  I need to add if statements to ensure
    the model has that particular attribute before printing'''
    print("aic: {}".format(model_fit.aic))
    print("bic: {}".format(model_fit.bic))
    print("bse: {}".format(model_fit.bse))
    print("fittedvalues: {}".format(model_fit.fittedvalues.head()))
    print("fpe: {}".format(model_fit.fpe))
    print("k_ar: {}".format(model_fit.k_ar))
    print("k_trend: {}".format(model_fit.k_trend))
    print("nobs: {}".format(model_fit.nobs))
    


def dw(data):
    ''' This will print out the Durbin-Watson statistic, given only the Visits
    or time dependent data. '''
    ols_res = OLS(data, np.ones(len(data))).fit()
    return durbin_watson(ols_res.resid)


def weeks_between(start_date, end_date):
    ''' This fucntion calculates the weeks between a start date and end date.
    I used this for some plots to get an array of ints'''
    weeks = rrule.rrule(rrule.WEEKLY, dtstart=start_date, until=end_date)
    return weeks.count()


def diff_month(d1, d2):
    ''' Similar to weeks_between'''
    return (d1.dt.year - d2.year) * 12 + d1.dt.month - d2.month


# In[1]
################## START #################
    
#os.chdir(r"C:\Users\Ryan Talk\Documents\SMU_Courses\Capstone\Campaign_EDA")
df = pd.read_csv("cleanedData2.csv")
#df.Date = pd.to_datetime(df.Date, format="%Y-%M-%d")
df.Date = pd.to_datetime(df.Date, format="%m/%d/%Y")
df = df.drop("Unnamed: 0", axis=1)
df.info()


# ### Aggregate by Role

# In[206]:


# These are the categorical variables we hope to use in the model
# We include date as we need to group by it 
model_categorical_vars = ["Date", "Source_Engine", "Experience_Level", "Role", "Location", "Primary_Function"]

# These are the variables we wish to predict
response_vars = ["Visits", "Subscribes", "Apply_Starts", "Apply_Completes", "Qualifieds"]

#df_Agg = df.groupby(model_categorical_vars, as_index=False)[response_vars].sum()
df_Agg = df.groupby(["Date","Role"], as_index=False)[response_vars].mean()


# In[207]:


# Pull out the Analysts to get get things going
analystData = df_Agg.loc[df_Agg.Role == "Analyst",:]
analystData = analystData.set_index("Date")
idx = pd.date_range(analystData.index.min(),analystData.index.max())
analystData = analystData.reindex(idx, fill_value=0)
# 1-D response Variable
response = analystData["Visits"]

# In[4]


# TODO: Ryan - This needs to be fixed so that we can use the analystData that uses the index above.  
analystData = df_Agg.loc[df_Agg.Role == "Analyst",:]
# This will plot the residuals from an OLS.   We can see any potential patterns in our data here
fig, ax = plt.subplots(figsize=(12,5))
sns.residplot((analystData.Date-pd.datetime(2016,1,28)).dt.days, analystData.Visits, lowess=True,line_kws={"color":"cyan"})


# In[242]:

# The ACF plot at 60 lags.  This way we can see any significane at 52 weeks
plot_acf(analystData.Visits, lags=60)


# In[244]:

#This will plot PACF for 60 lags
plot_pacf(analystData.Visits, lags=60)
plt.show()


# In[245]:

# This will output an array of pacf values
pacf(analystData.Visits, nlags=60)


# In[246]:
# This calculates the DW statistic for the analysts data.  A value close to 2
# implies that no autocorrelation was detected at lag 1 (1.5-2.5)
# Closer to zero implies negative autocorrelation
# Closer to 4 implies positive autocorrelation
# https://en.wikipedia.org/wiki/Durbin%E2%80%93Watson_statistic
# https://www.statsmodels.org/dev/generated/statsmodels.stats.stattools.durbin_watson.html
# TODO - calculate critical values of the DW statistic such that they tell us
# If the result is significant.

print("dw of range=%f" % dw(analystData.Visits))
print("dw of rand=%f" % dw(np.random.randn(2000)))


# Durbin-Watson Statistic is 1.139.  This is not too far from 2, but it is still cause to explore some autoregressive models.  We will also compare these to the average per month.

# In[249]:
########################## START 2- WEEKLY DATA   ######################

# We group everything by date and  calculate the mean  This provides daily data
df_Agg2 = df.groupby(["Date"], as_index=False)[response_vars].mean()


# This sums all the values in df_Agg2 by Date using the Monday of each week as a starting point. 
weekly = df_Agg2.resample('W-Mon', on='Date').sum().reset_index().sort_values(by='Date')


# In[334]:

# Gets an array of numbers indicating the weeks between 1/28/2016 and the data 
# date.
weeks = [weeks_between(pd.datetime(2016,1,28),weekly.Date[i]) for i in range(0,len(test.Date))]

# In[338]:

# This will plot the residuals for the weekly data
fig, ax = plt.subplots(figsize=(12,5))
b = range(0,len(weekly.Visits))
sns.residplot(x=np.array(weeks), y=weekly.Visits, lowess=True,line_kws={"color":"cyan"})


# In[256]:

# Plots the acf at 60 lags.  We want to see any yearly connections
plot_acf(weekly.Visits, lags=60)


# In[257]:


print("dw of range=%f" % dw(weekly.Visits))


# In[258]:


plot_pacf(weekly.Visits, lags=60)
plt.show()


# In[244]

# TODO: Make the elow code work with our data:
import matplotlib.pyplot as plt
from matplotlib.dates import date2num
import datetime

x = [datetime.datetime(2011, 1, 4, 0, 0),
     datetime.datetime(2011, 1, 5, 0, 0),
     datetime.datetime(2011, 1, 6, 0, 0)]
# date2num may be better than the weeks_between function.
x = date2num(x)

y = [4, 9, 2]
z=[1,2,3]
k=[11,12,13]

ax = plt.subplot(111)
ax.bar(x-0.2, y,width=0.2,color='b',align='center')
ax.bar(x, z,width=0.2,color='g',align='center')
ax.bar(x+0.2, k,width=0.2,color='r',align='center')
ax.xaxis_date()

plt.show()


# In[261]:

###################### MONTHLY DATA EDA ###############################
test_month = df_Agg2.resample('m', on='Date').sum().reset_index().sort_values(by='Date')


# In[265]:


plot_acf(test_month.Visits, lags=15)
plot_pacf(test_month.Visits, lags=15)
print("dw of range=%f" % dw(test_month.Visits))


# In[279]:


fig, ax = plt.subplots(figsize=(12,5))
sns.residplot(diff_month(test_month.Date,pd.datetime(2016,1,28)), test_month.Visits, lowess=True,line_kws={"color":"cyan"})


# In[340]:

############################  MODELS USING WEEKLY DATA  ############################
# TODO - Incorporate the exogenous data to the models.  May need to one hot encode

# Moving Average
# fit model
response = weekly.Visits
model = ARMA(endog = response, order=(0, 1))
model_fit = model.fit(disp=False)
# make prediction
yhat = model_fit.predict(0, len(response)+10)
createPlot(yhat, response)


# In[341]:

# ## AutoRegression
# http://www.statsmodels.org/dev/generated/statsmodels.tsa.ar_model.AR.html
# http://www.statsmodels.org/dev/generated/statsmodels.tsa.ar_model.ARResults.html
# 
# exogenous refers to the explanatory/independent variables
# 
# endogenous refers to the response/dependent variables

# AutoRegression
model = AR(endog = response)#, dates = temp.Date, freq='M')
model_fit = model.fit(maxlag=1, method='mle', disp=-1)
# make prediction
yhat = model_fit.predict(0, len(response)+10)
createPlot(yhat, response)


# In[343]:


# Autoregressive Integrated Moving Average
# ## Autoregressive Integrated Moving Average
# 
# http://www.statsmodels.org/dev/generated/statsmodels.tsa.arima_model.ARIMA.html#statsmodels.tsa.arima_model.ARIMA
# 
# https://otexts.com/fpp2/non-seasonal-arima.html
model = ARIMA(endog = response, order=(1, 0, 1))
model_fit = model.fit(disp=False)
# make prediction
yhat = model_fit.predict(0, len(response)+10)
createPlot(yhat, response)


# In[344]:


# Seasonal Autoregressive Integrated Moving Average
# ## Seasonal Autoregressive Integrated Moving Average
# http://www.statsmodels.org/dev/generated/statsmodels.tsa.statespace.sarimax.SARIMAX.html

# fit model
model = SARIMAX(response, order=(1, 1, 1), seasonal_order=(1, 1, 1, 1))
model_fit = model.fit(disp=False)
# make prediction
yhat = model_fit.predict(0, len(response)+10)
createPlot(yhat, response)
