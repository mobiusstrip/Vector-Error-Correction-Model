#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#---------------------------------------------------
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.tsa.ardl import ARDL,ardl_select_order,UECM
from arch.unitroot import ADF,PhillipsPerron
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.vector_ar.vecm import coint_johansen,select_coint_rank
from statsmodels.tsa.api import VAR
from statsmodels.tsa.vector_ar import irf
from statsmodels.tsa.vector_ar.var_model import FEVD
from statsmodels.tsa.vector_ar.vecm import VECM
from pmdarima import auto_arima
#---------------------------------------------------
df = pd.read_excel('/Users/steam/stuff/uni/thesis/reg/reg-data.xlsx',sheet_name="spyder")
df=df.dropna()
df.drop("Date",axis="columns",inplace=True)
dates = pd.date_range(start="31-03-2001", periods=85, freq="Q")
df["Dates"] = dates
df.set_index("Dates", inplace=True)
alpha=0.05
def pp_test(data):
    pp_test = PhillipsPerron(data)
    print(pp_test)

def pp_tests():
    for i in df.columns:
        pp_result = pp_test(df[i])
        print(f"pp test for column '{i}': {pp_result}")

def sw_test(results):
    shapiro_test = stats.shapiro(results.resid)
    if shapiro_test.pvalue < alpha:
        print("\nShapiro-Wilk:\n Non-normal\n p_value: ", shapiro_test.pvalue)
    else:
        print("\nShapiro-Wilk:\n Normal\n p_value: ", shapiro_test.pvalue)

# Kolmogorov-Smirnov Test
def ks_test(results):
    ks_test = stats.kstest(results.resid, 'norm')
    if ks_test.pvalue < alpha:
        print("\nKolmogorov-Smirnov: \n Non-normal \n p_value: ", ks_test.pvalue)
    else:
        print("\nKolmogorov-Smirnov: \n Normal  \n p_value: ", ks_test.pvalue)

def dw_test(results):
    dw_stat = sm.stats.durbin_watson(results.resid)  # Use statsmodels' function
    print("\nDurbin-Watson:\n",dw_stat)

def kpss(data):
    kpss_result = sm.tsa.stattools.kpss(data, regression='c', nlags='auto')
    p_value = kpss_result[1]
    print("KPSS Results:",data.name)

    if p_value > 0.05:
        print('Data is stationary around a constant','\np-value:', p_value)
        
    kpss_result = sm.tsa.stattools.kpss(data, regression='ct', nlags='auto')
    p_value = kpss_result[1]

    if p_value > 0.05:
        print('Data is stationary around a trend','\np-value:', p_value)
    
    else:
        print("Data is Non-stationary")

def ardl_analysis():
    data=df[["USD/TRY","US-TR-GDPPCPPP","Real-USD/TRY"]]
    y=df["LOG-DIFF-USD/TRY"]
    x=df[["US-TR-GDPPCPPP","Real-USD/TRY"]]
    #determine lags
    lag=ardl_select_order(endog=y,maxlag=4, exog=x,maxorder=4,ic="aic")
    print(lag.model.ardl_order)

    #ardl model
    model=ARDL(endog=y,lags=[4],exog=x,order=[4],trend="ct").fit()
    print(model.summary())

    #cointegration test
    bt=UECM(endog=y,lags=2,exog=x,order=3).fit()
    print(bt.summary())
    print(bt.bounds_test(case=2))
    print(bt.bounds_test(case=2).crit_vals)

def rm_season():
    import pandas as pd
    from statsmodels.tsa.seasonal import seasonal_decompose
    result = seasonal_decompose(df["US-TR-Real-Interest-Rate"], model='additive')
    deseasonalized = df["US-TR-Real-Interest-Rate"] - result.seasonal
    return deseasonalized

# deseasonalized_data=rm_season()

def data_plots():
    plt.style.use('dark_background')
    plt.figure(figsize=(5,5),dpi=600)
    plt.title("Nominal-USD/TRY")
    plt.xlabel('Dates')
    plt.ylabel('Nominal-USD/TRY')
    plt.axhline(y=0, color='tab:blue', linestyle='-',linewidth=2)
    plt.scatter(dates,df["Nominal-USD/TRY"],c="tab:blue",label="Rates")
    plt.legend()
    plt.grid(True)
    plt.show()
    
    plt.figure(figsize=(5,5),dpi=600)
    plt.title("Real-USD/TRY")
    plt.xlabel('Dates')
    plt.ylabel('Real-USD/TRY')
    plt.axhline(y=0, color='tab:blue', linestyle='-',linewidth=2)
    plt.scatter(dates,df["Real-USD/TRY"],c="tab:red",label="Rates")
    plt.legend()
    plt.grid(True)
    plt.show()

def ar1():
    import pandas as pd
    import matplotlib.pyplot as plt
    from statsmodels.tsa.arima.model import ARIMA
    
    # data= df[["USD/TRY","US-TR-GDPPCPPP","Real-USD/TRY","TR-NFA","US-TR-GDPPCPPP","US-TR-Real-Interest-Rate"]]
    data= df[["USD/TRY"]]

    model = ARIMA(data, order=(1, 0, 0)).fit()
    forecast = model.forecast(5)
    # Plot original data
    plt.plot(data, label='Original Data')
    # Plot in-sample predictions (fitted values)
    plt.plot(model.predict(), label='In-sample Fit')
    # Plot forecast
    plt.plot(pd.Series(forecast, index=range(len(data), len(data) + 5)), label='Forecast')
    
    plt.legend()
    plt.title('AR(1) Model Fit and Forecast')
    plt.show()
    
    
import numpy as np
import statsmodels.graphics.tsaplots as tsaplots
from statsmodels.graphics.tsaplots import plot_pacf
def acf_plot(data, name, acf=False, pacf=True):
    
    if acf:
        tsaplots.plot_acf(data, lags=15)
        plt.title(name + " ACF")
        plt.xlabel('Lag')
        plt.ylabel('Autocorrelation')
        plt.show()
    # pacf
    if pacf:
        plot_pacf(data, lags=15, method='ywm')
        plt.title(name + " PACF")
        plt.xlabel('Lag')
        plt.ylabel('Partial Autocorrelation')
        plt.show()

# acf_plot(df["Nominal-USD/TRY"], "Nominal-USD/TRY",acf=True,pacf=True)
# data_plots()
# adf_tests()
# pp_tests()
# johannsen_test(df[["USD/TRY","US-TR-GDPPCPPP","Real-USD/TRY","TR-NFA","US-TR-GDPPCPPP","US-TR-Real-Interest-Rate"]])
# ar1()

def lag_plot2(data):
    plt.style.use('dark_background')
    plt.figure(figsize=(5,5),dpi=600)
    plt.title("Nominal-USD/TRY")
    plt.xlabel('Dates')
    plt.ylabel('Nominal-USD/TRY')
    plt.axhline(y=0, color='tab:blue', linestyle='-',linewidth=2)
    plt.scatter(dates,df["Nominal-USD/TRY"],c="tab:blue",label="Rates")
    plt.legend()
    plt.grid(True)
    plt.show()



def lag_plot(data, order=1):
    diff_data = data.diff(order).dropna()
    if diff_data.empty:
        return  # Exit early if no data after differencing
    plt.style.use('dark_background')
    plt.figure(figsize=(8, 6), dpi=600)
    plt.title(f"Differenced Nominal-USD/TRY (Order={order})")
    plt.ylabel(f'Diff(Nominal-USD/TRY, {order})')
    plt.axhline(y=0, color='tab:red', linestyle='-', linewidth=2)
    plt.scatter(diff_data.index, diff_data, c="tab:orange", label="Differenced Rates")
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def johannsen_test(data, maxlags=1, significance_level=0.05):

    # Select VAR model order
    model = VAR(data)
    lag_order = model.select_order(maxlags=maxlags).aic

    # Johansen test
    result = coint_johansen(data, det_order=0, k_ar_diff=lag_order)

    # Prepare result DataFrame (with safer critical value indexing)
    num_rows = len(result.lr1)
    cv_index = min(int(significance_level * 100), num_rows - 1) 

    result_df = pd.DataFrame({
        'Trace Statistic': result.lr1,
        'Critical Value (Trace)': result.cvt[cv_index, :num_rows],
        'Eigen Statistic': result.lr2,
        'Critical Value (Eigen)': result.cvm[cv_index, :num_rows]
    })

    print("\nJohansen Cointegration Test Results:")
    print(result_df)

    # Determine the cointegrating rank
    for i in range(len(result_df)):
        if result_df["Trace Statistic"][i] > result_df["Critical Value (Trace)"][i]:
            print(f"There is evidence of cointegration at rank {i+1}")
        else:
            print(f"No cointegration at rank {i+1} or higher")
            break

def adf_test(data):
    adf_result = ADF(data, trend="c")  # "c" indicates constant mean
    p_value_adf = adf_result.pvalue 
    
    print("ADF Results:",data.name)
       
    if p_value_adf > 0.05:
        print('Data is stationary around a constant (ADF Test)\np-value:', p_value_adf)
    
    adf_result = ADF(data, trend="ct")  # "c" indicates constant mean
    p_value_adf = adf_result.pvalue 
    
    if p_value_adf > 0.05:
        print('Data is stationary around a trend (ADF Test)\np-value:', p_value_adf)
    
    else:
        print("Data is Non-Stationary")




data0=df.iloc[:, 0]
data1=df.iloc[:, 1]
# data2=df.iloc[:, 2]
    
    
# model=auto_arima(data0)
# print("NOMINAL:",model.summary())
    
# model=auto_arima(data1)
# print("REAL:",model)    
    
# model=auto_arima(data2)
# print("GDP:",model)    


data=df[["nominal","gdp"]]
model=VAR(df)
lag=model.select_order()
print(lag.selected_orders)
#25
j_test=coint_johansen(data, det_order=0, k_ar_diff=12)

print("TRACE",j_test.lr1)
print("TRACE CRIT",j_test.cvt[1])
print("TRACE2",j_test.lr2)
print("TRACE2CRIT",j_test.cvm[1])

vecm=VECM(data,k_ar_diff=12,deterministic="colo").fit()
print(vecm.summary())





















