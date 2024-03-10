#!/usr/bin/env python
# coding: utf-8

# In[33]:


from dateutil.parser import parse 
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
plt.rcParams.update({'figure.figsize': (10, 7), 'figure.dpi': 120})


# In[34]:


df = pd.read_excel('Serie.xlsx')
df


# In[39]:


# Time series data source: fpp pacakge in R.
import matplotlib.pyplot as plt
df = pd.read_excel('Serie.xlsx', parse_dates=['Date'], index_col='Date', sheet_name='Hoja1')

# Draw Plot
def plot_df(df, x, y, title="", xlabel='Date', ylabel='Inversión Social Resto', dpi=100):
    plt.figure(figsize=(16,5), dpi=dpi)
    plt.plot(x, y, color='tab:red')
    plt.gca().set(title=title, xlabel=xlabel, ylabel=ylabel)
    plt.show()

plot_df(df, x=df.index, y=df['Inversión Social Resto'], title='Inversión Social Nacional')    


# In[36]:


# Time series data source: fpp pacakge in R.

# Draw Plot of social incidents ($ COP of impact on production) and social investment projects ($ COP)
def plot_inc(df, x, y, title="", xlabel='Date', ylabel='No Incidentes', dpi=100):
    plt.figure(figsize=(16,5), dpi=dpi)
    plt.plot(x, y, color='tab:red')
    plt.gca().set(title=title, xlabel=xlabel, ylabel=ylabel)
    plt.show()

plot_inc(df, x=df.index, y=df['No Incidentes Nacional'], title='Incidentes de entorno Nacional')  


# In[5]:


df.dropna(how='all',inplace=True, subset=['NPT Costo Nacional'])


# In[6]:


x  = df.index
y3 = df['No Incidentes Nacional']
y1 = df['Inversión Social Resto']
y2 = df['Inversión Social Castilla']
y4= df['No Incidentes Castilla']


# In[7]:


fig, ax = plt.subplots(2, 2, sharex='col', sharey='row')
grid = plt.GridSpec(2, 3, wspace=0.4, hspace=0.3)
ax[0][0].plot(x,y1)
ax[0][1].plot(x,y2)
ax[1][0].plot(x,y3)


# In[8]:


from matplotlib.dates import DateFormatter
import matplotlib.dates as mdates
grid = plt.GridSpec(2, 2, wspace=0.2, hspace=0.3)
ax1= plt.subplot(grid[0, 0])
ax2= plt.subplot(grid[0, 1])
ax3= plt.subplot(grid[1, 0])
ax4= plt.subplot(grid[1, 1])
ax1.plot(x,y1, color='green')
ax2.plot(x,y3, color='green')
ax3.plot(x,y2)
ax4.plot(x,y4)
ax1.xaxis.set_major_locator(mdates.WeekdayLocator(interval=40))
ax1.xaxis.set_major_formatter(DateFormatter("%m-%y"))
ax2.xaxis.set_major_locator(mdates.WeekdayLocator(interval=40))
ax2.xaxis.set_major_formatter(DateFormatter("%m-%y"))
ax3.xaxis.set_major_locator(mdates.WeekdayLocator(interval=40))
ax3.xaxis.set_major_formatter(DateFormatter("%m-%y"))
ax4.xaxis.set_major_locator(mdates.WeekdayLocator(interval=40))
ax4.xaxis.set_major_formatter(DateFormatter("%m-%y"))
ax1.title.set_text('Inversión Social Nacional')
ax2.title.set_text('Incidentes de entorno Nacional')
ax3.title.set_text('Inversión Social Castilla')
ax4.title.set_text('Incidentes de entorno Castilla')


# In[9]:
# Comparison between Castilla oil field and Ecopetrol national data

from matplotlib.dates import DateFormatter
import matplotlib.dates as mdates
grid = plt.GridSpec(1, 2, wspace=0.2, hspace=0.3)
ax1= plt.subplot(grid[0, 0])
ax2= plt.subplot(grid[0, 1])
ax1.plot(x,y1, color='green', label='Nacional')
ax1.plot(x,y2, color='blue', label='Activo Castilla')
ax2.plot(x,y3, color='green', label='Nacional')
ax2.plot(x,y4, color='blue', label='Activo Castilla')
ax1.xaxis.set_major_locator(mdates.WeekdayLocator(interval=40))
ax1.xaxis.set_major_formatter(DateFormatter("%m-%y"))
ax2.xaxis.set_major_locator(mdates.WeekdayLocator(interval=40))
ax2.xaxis.set_major_formatter(DateFormatter("%m-%y"))
ax1.title.set_text('Inversión Social')
ax2.title.set_text('Incidentes de entorno')


# In[10]:


from matplotlib.dates import DateFormatter
import matplotlib.dates as mdates
grid = plt.GridSpec(1, 2, wspace=0.2, hspace=0.3)
ax1= plt.subplot(grid[0, 0])
ax3= plt.subplot(grid[0, 1])

ax1.plot(x,y1, color='green', label='Inv Social')
ax2 = ax1.twinx()
ax2.plot(x,y3, color='blue', label='Inc Entorno')
ax3.plot(x,y2, color='green', label='Inv Social')
ax4=ax3.twinx()
ax4.plot(x,y4, color='blue', label='Inc Entorno')
ax1.xaxis.set_major_locator(mdates.WeekdayLocator(interval=40))
ax1.xaxis.set_major_formatter(DateFormatter("%m-%y"))
ax3.xaxis.set_major_locator(mdates.WeekdayLocator(interval=40))
ax3.xaxis.set_major_formatter(DateFormatter("%m-%y"))
ax1.title.set_text('Nacional')
ax3.title.set_text('Activo Castilla')


# In[11]:


# visualizing the relationship between features and response using scatterplots

y5 = df['NPT Costo Castilla']
y6 = df['NPT Horas Castilla']
y7 = df['NPT Costo Nacional']
y8= df['NPT Horas Nacional']

grid2 = plt.GridSpec(2, 2, wspace=0.2, hspace=0.3)
ax1= plt.subplot(grid2[0, 0])
ax2= plt.subplot(grid2[0, 1])
ax3= plt.subplot(grid2[1, 0])
ax4= plt.subplot(grid2[1, 1])
ax1.scatter(y1,y3, color='green')
ax2.scatter(y1,y8, color='green')
ax3.scatter(y2,y4)
ax4.scatter(y2,y6)
ax1.title.set_text('Inv. Soc vs Incidentes Nacional')
ax2.title.set_text('Inv Soc vs NPT horas Nacional')
ax3.title.set_text('Inv Soc vs Incidentes Castilla')
ax4.title.set_text('Inv Soc vs NPT Horas Castilla')


# In[50]:


df.columns


# In[13]:


#Inv Soc e Incidentes Castilla
import numpy as np

import statsmodels.api as sm
from statsmodels.compat import lzip
import statsmodels.formula.api as smf
import statsmodels.stats.api as sms

# Fit regression model (using the natural log of one of the regressors)
results = sm.OLS(y4, y2).fit()

# Inspect the results
print(results.summary())
nameN = ["Jarque-Bera", "Chi^2 two-tail prob.", "Skew", "Kurtosis"]
testN = sms.jarque_bera(results.resid)
nameH = ["F statisticH", "p-valueH"]
testH = sms.het_goldfeldquandt(results.resid, results.model.exog)
nameML = ["t valueML", "p valueML"]
testML = sms.linear_harvey_collier(results)
lzip(nameML, testML, nameN, testN,nameH, testH)


# In[14]:


from statsmodels.graphics.regressionplots import plot_leverage_resid2

fig, ax = plt.subplots(figsize=(8, 6))
fig = plot_leverage_resid2(results, ax=ax)


# In[15]:


# Fit regression model (using the natural log of one of the regressors)
results = sm.OLS(y6, y2).fit()

# Inspect the results
print(results.summary())
name = ["Jarque-Bera", "Chi^2 two-tail prob.", "Skew", "Kurtosis"]
test = sms.jarque_bera(results.resid)
lzip(name, test)
nameN = ["Jarque-Bera", "Chi^2 two-tail prob.", "Skew", "Kurtosis"]
testN = sms.jarque_bera(results.resid)
nameH = ["F statisticH", "p-valueH"]
testH = sms.het_goldfeldquandt(results.resid, results.model.exog)
nameML = ["t valueML", "p valueML"]
testML = sms.linear_harvey_collier(results)
lzip(nameML, testML, nameN, testN,nameH, testH)


# In[16]:


results = sm.OLS(y3, y1).fit()

# Inspect the results
print(results.summary())
name = ["Jarque-Bera", "Chi^2 two-tail prob.", "Skew", "Kurtosis"]
test = sms.jarque_bera(results.resid)
lzip(name, test)
nameN = ["Jarque-Bera", "Chi^2 two-tail prob.", "Skew", "Kurtosis"]
testN = sms.jarque_bera(results.resid)
nameH = ["F statisticH", "p-valueH"]
testH = sms.het_goldfeldquandt(results.resid, results.model.exog)
nameML = ["t valueML", "p valueML"]
testML = sms.linear_harvey_collier(results)
lzip(nameML, testML, nameN, testN,nameH, testH)


# In[17]:


results = sm.OLS(y8, y1).fit()

# Inspect the results
print(results.summary())
name = ["Jarque-Bera", "Chi^2 two-tail prob.", "Skew", "Kurtosis"]
test = sms.jarque_bera(results.resid)
lzip(name, test)
nameN = ["Jarque-Bera", "Chi^2 two-tail prob.", "Skew", "Kurtosis"]
testN = sms.jarque_bera(results.resid)
nameH = ["F statisticH", "p-valueH"]
testH = sms.het_goldfeldquandt(results.resid, results.model.exog)
nameML = ["t valueML", "p valueML"]
testML = sms.linear_harvey_collier(results)
lzip(nameML, testML, nameN, testN,nameH, testH)


# In[18]:


results = sm.OLS(y7, y1).fit()

# Inspect the results
print(results.summary())
name = ["Jarque-Bera", "Chi^2 two-tail prob.", "Skew", "Kurtosis"]
test = sms.jarque_bera(results.resid)
lzip(name, test)


# In[19]:


results = sm.OLS(y5, y2).fit()

# Inspect the results
print(results.summary())
name = ["Jarque-Bera", "Chi^2 two-tail prob.", "Skew", "Kurtosis"]
test = sms.jarque_bera(results.resid)
lzip(name, test)
nameN = ["Jarque-Bera", "Chi^2 two-tail prob.", "Skew", "Kurtosis"]
testN = sms.jarque_bera(results.resid)
nameH = ["F statisticH", "p-valueH"]
testH = sms.het_goldfeldquandt(results.resid, results.model.exog)
nameML = ["t valueML", "p valueML"]
testML = sms.linear_harvey_collier(results)
lzip(nameML, testML, nameN, testN,nameH, testH)


# In[20]:


from statsmodels.tsa.stattools import grangercausalitytests
import numpy as np
df= df.loc[:, ['No Incidentes Nacional','Inversión Social Resto']]
gc_res = grangercausalitytests(df, 12)


# In[47]:


df2= df.loc[:, ['No Incidentes Nacional']]
sm.graphics.tsa.plot_pacf(df2.values.squeeze(), lags=15, method="ywm")
plt.show()


# In[46]:


df3= df.loc[:, ['No Incidentes Castilla']]
sm.graphics.tsa.plot_pacf(df3.values.squeeze(), lags=15, method="ywm")
plt.show()


# In[76]:


def lag(x, n, validate=True):
    """Calculates the lag of a pandas Series

    Args:
        x (pd.Series): the data to lag
        n (int): How many periods to go back (lag length)
        validate (bool, optional): Validate the series index (monotonic increasing + no gaps + no duplicates). 
                                If specified, expect the index to be a pandas PeriodIndex
                                Defaults to True.

    Returns:
        pd.Series: pd.Series.shift(n) -- lagged series
    """

    if n == 0:
        return x

    if isinstance(x, pd.Series):
        if validate:
            assert x.index.is_monotonic_increasing, (
                "\u274c" + f"x.index is not monotonic_increasing"
            )
            assert x.index.is_unique, "\u274c" + f"x.index is not unique"
            idx_full = pd.period_range(start=x.index.min(), end=x.index.max(), freq=x.index.freq)
            assert np.all(x.index == idx_full), "\u274c" + f"Gaps found in x.index"
        return x.shift(n)

    return x.shift(n)


# Use the defined function in the formula:
df_na = df.fillna(method='ffill')
df_na["y2"] = df_na["Inversión Social Castilla"]
df_na['y5']=df_na['NPT Costo Castilla']
df_na = df_na.fillna(method='ffill')
smf.ols(formula="y5 ~ lag(y2,1)", data=df_na).fit().summary()


# Inspect the results
print(results.summary())


# In[82]:


from statsmodels.tsa.api import ARDL
from statsmodels.tsa.ardl import ardl_select_order


# In[ ]:


y = np.asarray(data['NPT Costo Castilla'])
x = np.asarray(data["Inversión Social Castilla"])
res = ARDL(y, 2, x, {0: 1, 1: 2, 2: 3}, trend="c").fit()
res.summary()


# In[74]:


df.columns


# In[ ]:




