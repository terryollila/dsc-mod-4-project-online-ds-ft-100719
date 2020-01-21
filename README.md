
# Premise

The purpose of this analysis is to explore  the forecasting for multiple zip codes in the United States in order to assess the best possible opportunities for real estate investment.

In pursuit of this investigation, I will attempt to begin at a national level, using ARIMA and/or SARIMAX methodology and modeling to assess the most promising locations. To do this, I will begin with the strongest state as determined by average forecasted percent return on invesment over the next ten years, and from there, find the strongest county within that state, then the strongest city within that county, and finally the five zip codes within that city, again, all judged by percent return on investment over the next ten years.

The models will be tuned in such a way as to capture this information with the greatest precision and accuracy as possible before drillingn down to the state, county, city, and ultimately zip code levels. 

I will not attempt to ascertain the top five zip codes in the entire country. Instead I will attempt to locate the top zipcodes in the top city of the top county of the top state. This method will ensure strength in the surrounding areas, rather than pursuing what could be isolated zip codes showing unsustainable and perhaps misleading or illusory figures. 

# Preparing Libraries, Functions, and Importing Data

## Importing libraries


```python
import pandas as pd
from pandas import datetime
import numpy as np
import statsmodels
import statsmodels.api as sm
import itertools
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pylab import rcParams
import seaborn as sns
import warnings
import random
import chart_studio.plotly as py
warnings.filterwarnings('ignore')
%matplotlib inline
sns.set()
sns.set_style({'axes.facecolor': (.75, .75, .75, .7)})
```


```python
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)
```

## My Functions

### SARIMAX Forecast

The SARIMAX and ARIMA testing and forecasting functions will be the primary tools I will use for modeling and graphing in both exploration and conclusion.


```python
def forecast_SARIMAX(data, order = (1,1,1), 
              seasonal_order = (0,0,0,0),
              start = '2018-04-01', 
              end = '2023-04-01',
              plot_start = '1996-04-01'):
    """Takes in zip code home value arguments to create SARIMAX summary table, 
    diagnostic graphs, and forecasting graph.
    
    Prameters:        
    
        zip: Series
            Time Series with median home values.
        
        order: tuple (optional)
            pdq for SARIMAX model.
            If empty, defaults to (1,1,1)
        
        seasonal_order: tuple (optional)
            PDQS for SARIMAX model.
            If empty, defaults to (0,0,0,0)
        
        start: string
            Beginning date for time series predictions.
            If empty, defaults to '2018-04-01'
        
        end: string
            End date for time series predictions. Note that if this date
            is beyond the end of the time series, it will create a forecast
            with confidence intervals.
            If empty, defaults to '2023-04-01'
            
        plot_start: string
            Start date for where to begin the graph on on the x axis.
            If empty, defaults to '1996-04-01'
    
    Returns:
        
        statsmodels SARIMAX summary table, diagnostic graphs, and forecasting
        graph."""
    
    # Creating a model
    model = sm.tsa.statespace.SARIMAX(data, order = order,
                                      seasonal_order = seasonal_order,
                                      enforce_stationarity = True,
                                      enforace_invertability = True)
    
    output = model.fit()
    # Print out summary table and aic score
    print(output.summary().tables[1])
    print('AIC: ', output.aic)
    
    #The prediction can be used to forecast into the future.
    prediction = output.get_prediction(start = pd.to_datetime(start),
                                       end = pd.to_datetime(end),
                                       dynamic = False)
    
    # Capturing confidence intervals
    pred_conf = prediction.conf_int()
    
    # Create diagnistic graphs
    output.plot_diagnostics(figsize = (12, 8));
    plt.show()
    
    # Create the prediction graph
    rcParams['figure.figsize'] = 12,8
    ax = data[plot_start:].plot(label = 'observed')
    
    # Plotting based on the predicted mean and filling in confidence intervals
    prediction.predicted_mean.plot(ax = ax, label = 'Forecast', alpha = .9)
    ax.fill_between(pred_conf.index, pred_conf.iloc[:, 0], pred_conf.iloc[:, 1],
                    color =  'g', alpha = .5)
    
    #Changing y limits so that the graph is in proper scale with 0 min.
    ax.set_ylim(bottom = 0)
    
    ax.legend(loc = 'upper left')
    plt.show()
```

### SARIMAX Train-Test


```python
def train_test_SARIMAX(data, order = (1,1,1), 
              seasonal_order = (0,0,0,0),
              start = '2013-04-01', 
              end = '2018-04-01',
              plot_start = '1996-04-01'):
    """Takes in zip code home value arguments to create SARIMAX summary table, 
    diagnostic graphs, and forecasting graph.
    
    Prameters:        
    
        data: Series
            Time Series with median home values.
        
        order: tuple (optional)
            pdq for SARIMAX model.
            If empty, defaults to (1,1,1)
        
        seasonal_order: tuple (optional)
            PDQS for SARIMAX model.
            If empty, defaults to (0,0,0,0)
        
        start: string
            Beginning date for time series predictions.
            If empty, defaults to '2018-04-01'
        
        end: string
            End date for time series predictions. Note that if this date
            is beyond the end of the time series, it will create a forecast
            with confidence intervals.
            If empty, defaults to '2023-04-01'
            
        plot_start: string
            Start date for where to begin the graph on on the x axis.
            If empty, defaults to '1996-04-01'
    
    Returns:
        
        statsmodels SARIMAX summary table, diagnostic graphs, and forecasting
        graph."""
    
    # Splitting up the data in train and test sets by date.    
    train = data[:start]
    test = data[start:]
    
    # Creating a model
    model = sm.tsa.statespace.SARIMAX(train, order = order,
                                      seasonal_order = seasonal_order,
                                      enforce_stationarity = False,
                                      enforace_invertability = False)
    
    output = model.fit()
    # Print out summary table and aic score
    print(output.summary().tables[1])
    print('AIC: ', output.aic)
    
    #The prediction can be used to forecast into the future.
    prediction = output.get_prediction(start = test.index[0],
                                       end = pd.to_datetime(end),
                                       dynamic = False)
    
    # Capturing confidence intervals
    pred_conf = prediction.conf_int()
    
    # Create diagnistic graphs
    output.plot_diagnostics(figsize = (12, 8));
    plt.show()
    
    # Create the prediction graph
    rcParams['figure.figsize'] = 12,8
    ax = data[plot_start:].plot(label = 'actual')
    
    # Plotting based on the predicted mean and filling in confidence intervals
    prediction.predicted_mean.plot(ax = ax, label = 'Forecast', alpha = .9)
    ax.fill_between(pred_conf.index, pred_conf.iloc[:, 0], pred_conf.iloc[:, 1],
                    color =  'g', alpha = .5)
    
    #Changing y limits so that the graph is in proper scale with 0 min.
    ax.set_ylim(bottom = 0)
    
    ax.legend(loc = 'upper left')
    plt.show()
```

### ARIMA Forecast


```python
def forecast_ARIMA(data, order = (1,1,1),
              start = '2018-04-01', 
              forecast_length = 60,
              figsize = (12,5),
              diagnostics = False,
              denver = False,
              title = None,
              color=None):
    """Model and graph a zip code time series.
    
    Parameters:
    
        data: Series
            Time Series with median home values.
        
        order: tuple (optional)
            pdq for SARIMAX model.
            If empty, defaults to (1,1,1)
            
        start: str
            Date to start the forecast from in yyyy-mm-dd format.
            
        forecast_length: int
            Number of months to extend the forecast.
            
        figsize: tuple
            Creates the size of the graph.
            
        diagnostics: bool
            Determines whether to display table of model information.
            
        denver: bool
            Indicates whether to use Broncos colors for Denver
            
        color: str
            Color of lines on the resulting graph.
            
    Returns:
        A graph and, if selected, diagnostic information about p-values etc."""
            
    
    # Will need a placeholder series to build the forecast against.
    ext_dates = pd.Series(range(0,forecast_length,1), 
                          index = pd.date_range(start, 
                                                periods = forecast_length, 
                                                freq = 'MS'))
    
    # ARIMA doesn't like series names to be integers, so zipcodes must be 
    # changed to strings.
    data.name = str(data.name)
    
    model = statsmodels.tsa.arima_model.ARIMA(data, order=order)  
    fitted = model.fit()  
    print(fitted.summary()) if diagnostics == True else ''
    
    # Forecast
    fc, se, conf = fitted.forecast(forecast_length, alpha=0.05)

    # Make as pandas series
    fc_series = pd.Series(fc, index=ext_dates.index)
    lower_series = pd.Series(conf[:, 0], index=ext_dates.index)
    upper_series = pd.Series(conf[:, 1], index=ext_dates.index)
    
    # Plot
    plt.figure(figsize=figsize, dpi=100)
    
    # Create the special option for a Rockies/Broncos version.
    if denver == True:
        plt.plot(data, label='current data', color = 'blue')
        plt.plot(fc_series, label='forecast', color = 'orange')
        plt.fill_between(lower_series.index, lower_series, upper_series, 
                         color='white', alpha=.85)
        plt.grid(b=True, axis='y', color = 'purple')
    else:
        plt.plot(data, label='current data')
        plt.plot(fc_series, label = 'forecast')
        plt.fill_between(lower_series.index, lower_series, upper_series,
                         color=color, alpha=.5)
    
    plt.title(title) if title else plt.title(data.name)
    plt.ylim(bottom = 0)
    plt.ylabel('value in $')
    plt.xlabel('year')
        
    plt.legend(loc='upper left', fontsize=10)
    
    # Now plot all the common elements.
    plt.show()
```

### ARIMA Forecast Subplots


```python
def forecast_ARIMA_sub(data, ax, order = (1,1,0),
              start = '2018-04-01', 
              forecast_length = 60,
              diagnostics = False,
              figsize = (12,5),
              denver = False,
              intervals = True,
              color=None,
              lw=1,
              ls='-'):
    """Model and graph a zip code time series in subplots.
    
    Parameters:
    
        data: Series
            Time Series with median home values.
        
        order: tuple (optional)
            pdq for SARIMAX model.
            If empty, defaults to (1,1,1)
            
        start: str
            Date to start the forecast from in yyyy-mm-dd format.
            
        forecast_length: int
            Number of months to extend the forecast.
            
        figsize: tuple
            Creates the size of the graph.
            
        diagnostics: bool
            Determines whether to display table of model information.
            
        denver: bool
            Indicates whether to use Broncos colors for Denver
            
        intervals: bool
            Whether or not to display the confidence intervals on graph.
                
        color: str
            Color of lines on the resulting graph.
            
        lw: int
            line width for graph.
            
        ls: str
            line style for graph. '-', '--' etc.
            
    Returns:
        A graph and, if selected, diagnostic information about p-values etc."""
    # Will need a placeholder series to build the forecast against.
    ext_dates = pd.Series(range(0,forecast_length,1), 
                          index = pd.date_range(start, 
                                                periods = forecast_length, 
                                                freq = 'MS'))
    
    # ARIMA doesn't like series names to be integers, so zipcodes must be 
    # changed to strings.
    data.name = str(data.name)
    
    model = statsmodels.tsa.arima_model.ARIMA(data, order=order)  
    fitted = model.fit()  
    print(fitted.summary()) if diagnostics == True else ''
    
    # Forecast
    fc, se, conf = fitted.forecast(forecast_length, alpha=0.05)

    # Make as pandas series
    fc_series = pd.Series(fc, index=ext_dates.index)
    lower_series = pd.Series(conf[:, 0], index=ext_dates.index)
    upper_series = pd.Series(conf[:, 1], index=ext_dates.index)
    
    # Plot   
    # Create the special option for a Rockies/Broncos version.    
    if denver == True:
        ax.plot(data, color = 'blue', lw=lw, ls=ls)
        ax.plot(fc_series, label=data.name, color = 'orange', lw=lw, ls=ls)
        ax.fill_between(lower_series.index, lower_series, upper_series, 
                         color='white', alpha=.7) if intervals else ''
#         ax.grid(b=True, axis='y', color = 'purple', lw = 1)
        ax.set_ylim(0, 1100000)
    else:
        ax.plot(data, lw=lw, color = 'g', ls=ls)
        ax.plot(fc_series, label=data.name, lw=lw, color=color, ls=ls)
        ax.fill_between(lower_series.index, lower_series, upper_series,
                            color = 'g', alpha=.4) if intervals else ''
        ax.set_ylim(0, 1100000)
    
    ax.set_title(data.name, fontsize = 28)
    ax.set_ylim(bottom = 0)
    ax.set_ylabel('value in $', fontsize = 20)
    ax.set_xlabel('year', fontsize = 20)
    ax.tick_params(axis="x", labelsize=16)
    ax.tick_params(axis="y", labelsize=16)
        
    ax.legend(loc='upper left', fontsize=16)
    
    return ax
```

### ARIMA Train-Test


```python
def train_test_ARIMA(data, order = (1,1,0),
              start = '2018-04-01'):
    """Model and graph a zip code time series.
    
    Parameters:
    
        data: Series
            Time Series with median home values.
        
        order: tuple (optional)
            pdq for SARIMAX model.
            If empty, defaults to (1,1,1)
            
        start: str
            Date to start the test from in yyyy-mm-dd format.
            Test will proceed to end of data set.
                        
    Returns:
        A graph showing train vs test information, and diagnostic chart."""
    
    # Splitting up the data in train and test sets by date.
    train = data[:start]
    test = data[start:]
    
    # ARIMA doesn't like series names to be integers, so zipcodes must be 
    # changed to strings.
    data.name = str(data.name)
    
    model = statsmodels.tsa.arima_model.ARIMA(train, order=order)  
    fitted = model.fit()  
    print(fitted.summary())

    # Forecast
    fc, se, conf = fitted.forecast(len(test), alpha=0.05)

    # Make as pandas series
    fc_series = pd.Series(fc, index=test.index)
    lower_series = pd.Series(conf[:, 0], index=test.index)
    upper_series = pd.Series(conf[:, 1], index=test.index)

    # Plot
    plt.figure(figsize=(12,5), dpi=100)
    plt.plot(train, label='training')
    plt.plot(test, label='actual')
    plt.plot(fc_series, label='predicted')
    plt.fill_between(lower_series.index, lower_series, upper_series, 
                     color='g', alpha=.5)
    plt.title('Forecast vs Actuals')
    plt.legend(loc='upper left', fontsize=8)
    plt.ylim(0)
    plt.show()
```

### AIC Grid Search

This function attempts modeling with all combinations of hyperparameters as specified.


```python
def aic_search(data, p_max=2, d_max=2, q_max=2, end='2013-04-01'):
    """Grid search to determine starting parameters for ARIMA modeling.
    
    Paramaters:
    
        data: series
            A time series of one zip with mean value data.
            
        p_max: int
            Max AR order value, or 'p'. This will be one greater  than 
            the largest value the grid search with check in its combination set. 
            
        d_max: int
            Max differencing order value, or 'd'. This will be one greater  than 
            the largest value the grid search with check in its combination set. 
            
        q_max: int
            Max differencing order value, or 'q'. This will be one greater  than 
            the largest value the grid search with check in its combination set. 
            
        end: str
            Date to end the training portion of the time series for determing
            parameters."""
        

    # I played with the length of the test period to ensure I had confidence
    # in the ultimate hyperparameter set that I used for forecasting.
    test_data = data[:end]
    
    #Turning the start/end dates into a forecast length.
    forecast_len = ((pd.to_datetime('2018-04-01') - pd.to_datetime(end)).days
                    // 365)*2
    
    # ARIMA doesn't like series names to be integers, so zipcodes must be 
    # changed to strings.
    data.name = str(test_data.name)
    
    # Set the ranges specified for parameters
    p = range(0,p_max)
    d = range(0,d_max)
    q = range(0,q_max)

    # Generate all different combinations of p, q and q triplets
    pdq = list(itertools.product(p, d, q))
    
    # Iterating through each of the combinations and running the model for each
    # to see which one has the lowest AIC score.
    ans = []
    ans_df = pd.DataFrame()
    rejects = []

    for comb in pdq:
        try:
            # create and fit the model
            model = statsmodels.tsa.arima_model.ARIMA(test_data, 
                                                      order=(comb)) 
            fitted = model.fit()

            # Add the newest combination to the ans list
            ans.append([comb, fitted.aic])

            # Create the variables for forecast, confidence intervals, and
            # whatever the hell se is.
            fc, se, conf = fitted.forecast(forecast_len, alpha=0.05)

            i = forecast_len - 1
            fc = round(fc[i],1)
            interval_size = round(conf[i,1] - conf[i,0],0)
            df_data = [[comb, fitted.aic, interval_size, fc]]
            columns = ['Orders', 'AIC', 'IntervalSize', 'forecasted_value']

            # Fill the array with AIC results and the dataframe with additional info
            temp_df = pd.DataFrame(data = df_data, columns = columns)
            ans_df = pd.concat([ans_df, temp_df])

        except:
            # If any combination fails, we probably want to know about it so 
            # we can potentially correct the problem and re-try. It might be 
            # the winner.
            rejects.append(comb)
#         print('Combination {} failed.'.format(comb))
            continue
    
    ans_df = ans_df.sort_values('AIC', ascending = True)
    ans.sort(key = lambda x: x[1], reverse = False)
    
    return ans, ans_df, rejects
```

### Nationwide Modeling

This is the main function for making the search to determine forecasting for every zip code in the data set. Can be used with smaller data sets as well.


```python
def national_models(source, info, params=(3,1,2)):
    """Takes parameters and a column set to create models for.
    
    parameters:
        source: DataFrame
            Specifies which data to use during modeling.
            
        info: DataFrame
            DataFrame housing zip info such as city and county.
            
        params: tuple of ints
            Set of parameters to use for modeling.
            
    returns:
        DataFrame housing the zip information for city etc, along with 
        projections for each year from last date of data set forward to 10 
        years, including $ and % gain."""
    
    # This is the blank DataFrame that all records will be concatenated to as they
    # are created.
    df = pd.DataFrame()  
    
    # The following loop will cycle through every zip code, create projections out
    # ten years, and build the all_zip_models DataFrame piece by piece.
    # for column in zdf_by_zip.columns[1:2]:
    for column in source.columns:
        try:
            data = source[column]
            # Will need a placeholder series to build the forecast against.
            ext_dates = pd.Series(range(0,120,1), 
                                  index = pd.date_range('2018-04-01', 
                                                        periods = 120, 
                                                        freq = 'MS'))

            # ARIMA doesn't like series names to be integers, so zipcodes must be 
            # changed to strings.
            data.name = str(data.name)

            model = statsmodels.tsa.arima_model.ARIMA(data, 
                                                      order = params)
            output = model.fit()

            # Forecast
            fc, se, conf = output.forecast(120, alpha=0.05)

            # Make as pandas series
            forecast = pd.Series(fc, index=ext_dates.index)

            # projections will house the metadata for each zip, and columns will
            # house the projection data as it is created.
            projections = [column]
            columns = ['Zip']

            # I'll be iterating 1 year at a time, beginning 1 year from the last 
            # date in the data set.
            j = 11
            columns.extend(info.columns)
            columns.append('2018-04-01')

            # Continuing to add the zip metadata.
            for col_name in info.columns:
                projections.append(zip_info.loc[column][col_name])

            # Defining where we will begin the forecast.
            start = source.loc['2018-04-01'][column]
            projections.append(start)

            # The 10 years of projections are added one at a time. 
            # Percent Gain is also added in at this point.
            for i in range(0, 10):
                this_proj = round(forecast[j],2)
                this_gain = round((this_proj - start) / start,2)
                projections.extend([this_proj, this_gain])
                columns.extend([str(i + 1) + '_Year', str(i + 1) + '_Gain'])
                j += 12
            # Create a temporary dataframe from the data I've created to 
            # concatenate to the master DataFrame.
            temp = pd.DataFrame([projections], columns = columns)
            df = pd.concat([df, temp], axis = 0,
                                       ignore_index = True)

        except:
            continue
            
    return df
```

### Grouping by Area

This function will group records together as specified by either state, county, or city.


```python
def group_areas(data, group = 'State', 
                years_ahead = 10):
    """Group a zipcode DataFrame in zip-wide format 
    according to state, county, or city
        
    Parameters:
        
        data (DataFrame)
            Zipcode DataFrame in zip-wide format with forecasts
            
        group (String)
            State, CountyName, or City value from zipcode dataframe.
            
        years_ahead (Int)
            1 to 10 years for forecast length.
            
    Returns:
        DataFrame with areas grouped by top precentage of gain.

    """
    forecast_len = str(years_ahead) + '_Gain'
    
    df = pd.DataFrame(data.groupby(
        group)[forecast_len].mean()).sort_values(
        forecast_len, ascending = False)

    df['Rank'] = [i + 1 for i in range(0, len(df))]
    
    return df
```

### Drill for Zip Codes

This is the function that will be used to find the top zip codes in the specified state, as discovered by drilling into county, city, and zipcode levels.


```python
def drill_for_zips(state, 
                   data=None, 
                   num_results=5,
                   years_ahead=10):
    """Drills down from state through county and city levels to find the 
    top ranked zip codes by ranking of precent gain over specified years.
    
    Parameters:
    
        state (string)
            In capital abbreviated format ('CA'), specifies the state
            to investigate.
            
        data (DataFrame)
            Generally uses all_zip_ranked or another DataFrame 
            with zip codes ranked by percent gains, 1 - 5 years out.
            
        num_results (Int)
            How many results to display, ranked from the first down.
            
        years_ahead (Int)
            How many years ahead to look when searching for percent gain."""
            
    #Create df of all records in the prescribed state
    top_state_df = data[data['State'] == state]

    # create df of all counties within the specified state.
    county_df = group_areas(top_state_df, 'CountyName', years_ahead)
    
    # This will be the list of top zip codes that will be used to create the 
    # dataframe that will be returned.
#     top_zips = []
    
    top_zips = pd.DataFrame()
    
    # This will loop through the necessary data sets, drilling into each 
    # county and city until the specified number of results has been reached.
    i = 0
    while len(top_zips) < num_results:
                
        # Create a df of only the data for the top county in this state.
        # Iterates if not enough records in the first county.
        top_county_df = top_state_df[top_state_df['CountyName']
                               == county_df.index[i]]
        
        # Drilling to the city level.
        j = 0
        while len(top_zips) < num_results:
            
            # Create a grouped df that groups the zips by city.
            city_group = group_areas(top_county_df, 'City', 10)
        
            # Create a df of the data for the top city in this county.
            try:
                top_city_df = top_state_df[(top_state_df['City'] 
                                     == city_group.index[j])
                                     & (data['State']
                                     == state)]
            except:
                j = 0
                break

            # Iterate through each zip in the 'top city' and add to top_zips.
            for a_zip in range(len(top_city_df)):
                if len(top_zips) < num_results:
                    
                    # Finding the zip code for this iteration, getting the full
                    # data, and adding it to top_zips.
                    this_zip = list(top_city_df.Zip)[a_zip]
                    temp = data[data['Zip'] == this_zip]
                    top_zips = pd.concat([top_zips, temp])
            
                else:
                    break
                
            j += 1
        
        i += 1
    
    return top_zips.sort_values('10_Gain', ascending = False)

```

## Importing and Reviewing Data


```python
zdf = pd.read_csv('zillow_data.csv')
zdf.head(1)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>RegionID</th>
      <th>RegionName</th>
      <th>City</th>
      <th>State</th>
      <th>Metro</th>
      <th>CountyName</th>
      <th>SizeRank</th>
      <th>1996-04</th>
      <th>1996-05</th>
      <th>1996-06</th>
      <th>1996-07</th>
      <th>1996-08</th>
      <th>1996-09</th>
      <th>1996-10</th>
      <th>1996-11</th>
      <th>1996-12</th>
      <th>1997-01</th>
      <th>1997-02</th>
      <th>1997-03</th>
      <th>1997-04</th>
      <th>1997-05</th>
      <th>1997-06</th>
      <th>1997-07</th>
      <th>1997-08</th>
      <th>1997-09</th>
      <th>1997-10</th>
      <th>1997-11</th>
      <th>1997-12</th>
      <th>1998-01</th>
      <th>1998-02</th>
      <th>1998-03</th>
      <th>1998-04</th>
      <th>1998-05</th>
      <th>1998-06</th>
      <th>1998-07</th>
      <th>1998-08</th>
      <th>1998-09</th>
      <th>1998-10</th>
      <th>1998-11</th>
      <th>1998-12</th>
      <th>1999-01</th>
      <th>1999-02</th>
      <th>1999-03</th>
      <th>1999-04</th>
      <th>1999-05</th>
      <th>1999-06</th>
      <th>1999-07</th>
      <th>1999-08</th>
      <th>1999-09</th>
      <th>1999-10</th>
      <th>1999-11</th>
      <th>1999-12</th>
      <th>2000-01</th>
      <th>2000-02</th>
      <th>2000-03</th>
      <th>2000-04</th>
      <th>2000-05</th>
      <th>2000-06</th>
      <th>2000-07</th>
      <th>2000-08</th>
      <th>2000-09</th>
      <th>2000-10</th>
      <th>2000-11</th>
      <th>2000-12</th>
      <th>2001-01</th>
      <th>2001-02</th>
      <th>2001-03</th>
      <th>2001-04</th>
      <th>2001-05</th>
      <th>2001-06</th>
      <th>2001-07</th>
      <th>2001-08</th>
      <th>2001-09</th>
      <th>2001-10</th>
      <th>2001-11</th>
      <th>2001-12</th>
      <th>2002-01</th>
      <th>2002-02</th>
      <th>2002-03</th>
      <th>2002-04</th>
      <th>2002-05</th>
      <th>2002-06</th>
      <th>2002-07</th>
      <th>2002-08</th>
      <th>2002-09</th>
      <th>2002-10</th>
      <th>2002-11</th>
      <th>2002-12</th>
      <th>2003-01</th>
      <th>2003-02</th>
      <th>2003-03</th>
      <th>2003-04</th>
      <th>2003-05</th>
      <th>2003-06</th>
      <th>2003-07</th>
      <th>2003-08</th>
      <th>2003-09</th>
      <th>2003-10</th>
      <th>2003-11</th>
      <th>2003-12</th>
      <th>2004-01</th>
      <th>2004-02</th>
      <th>2004-03</th>
      <th>2004-04</th>
      <th>2004-05</th>
      <th>2004-06</th>
      <th>2004-07</th>
      <th>2004-08</th>
      <th>2004-09</th>
      <th>2004-10</th>
      <th>2004-11</th>
      <th>2004-12</th>
      <th>2005-01</th>
      <th>2005-02</th>
      <th>2005-03</th>
      <th>2005-04</th>
      <th>2005-05</th>
      <th>2005-06</th>
      <th>2005-07</th>
      <th>2005-08</th>
      <th>2005-09</th>
      <th>2005-10</th>
      <th>2005-11</th>
      <th>2005-12</th>
      <th>2006-01</th>
      <th>2006-02</th>
      <th>2006-03</th>
      <th>2006-04</th>
      <th>2006-05</th>
      <th>2006-06</th>
      <th>2006-07</th>
      <th>2006-08</th>
      <th>2006-09</th>
      <th>2006-10</th>
      <th>2006-11</th>
      <th>2006-12</th>
      <th>2007-01</th>
      <th>2007-02</th>
      <th>2007-03</th>
      <th>2007-04</th>
      <th>2007-05</th>
      <th>2007-06</th>
      <th>2007-07</th>
      <th>2007-08</th>
      <th>2007-09</th>
      <th>2007-10</th>
      <th>2007-11</th>
      <th>2007-12</th>
      <th>2008-01</th>
      <th>2008-02</th>
      <th>2008-03</th>
      <th>2008-04</th>
      <th>2008-05</th>
      <th>2008-06</th>
      <th>2008-07</th>
      <th>2008-08</th>
      <th>2008-09</th>
      <th>2008-10</th>
      <th>2008-11</th>
      <th>2008-12</th>
      <th>2009-01</th>
      <th>2009-02</th>
      <th>2009-03</th>
      <th>2009-04</th>
      <th>2009-05</th>
      <th>2009-06</th>
      <th>2009-07</th>
      <th>2009-08</th>
      <th>2009-09</th>
      <th>2009-10</th>
      <th>2009-11</th>
      <th>2009-12</th>
      <th>2010-01</th>
      <th>2010-02</th>
      <th>2010-03</th>
      <th>2010-04</th>
      <th>2010-05</th>
      <th>2010-06</th>
      <th>2010-07</th>
      <th>2010-08</th>
      <th>2010-09</th>
      <th>2010-10</th>
      <th>2010-11</th>
      <th>2010-12</th>
      <th>2011-01</th>
      <th>2011-02</th>
      <th>2011-03</th>
      <th>2011-04</th>
      <th>2011-05</th>
      <th>2011-06</th>
      <th>2011-07</th>
      <th>2011-08</th>
      <th>2011-09</th>
      <th>2011-10</th>
      <th>2011-11</th>
      <th>2011-12</th>
      <th>2012-01</th>
      <th>2012-02</th>
      <th>2012-03</th>
      <th>2012-04</th>
      <th>2012-05</th>
      <th>2012-06</th>
      <th>2012-07</th>
      <th>2012-08</th>
      <th>2012-09</th>
      <th>2012-10</th>
      <th>2012-11</th>
      <th>2012-12</th>
      <th>2013-01</th>
      <th>2013-02</th>
      <th>2013-03</th>
      <th>2013-04</th>
      <th>2013-05</th>
      <th>2013-06</th>
      <th>2013-07</th>
      <th>2013-08</th>
      <th>2013-09</th>
      <th>2013-10</th>
      <th>2013-11</th>
      <th>2013-12</th>
      <th>2014-01</th>
      <th>2014-02</th>
      <th>2014-03</th>
      <th>2014-04</th>
      <th>2014-05</th>
      <th>2014-06</th>
      <th>2014-07</th>
      <th>2014-08</th>
      <th>2014-09</th>
      <th>2014-10</th>
      <th>2014-11</th>
      <th>2014-12</th>
      <th>2015-01</th>
      <th>2015-02</th>
      <th>2015-03</th>
      <th>2015-04</th>
      <th>2015-05</th>
      <th>2015-06</th>
      <th>2015-07</th>
      <th>2015-08</th>
      <th>2015-09</th>
      <th>2015-10</th>
      <th>2015-11</th>
      <th>2015-12</th>
      <th>2016-01</th>
      <th>2016-02</th>
      <th>2016-03</th>
      <th>2016-04</th>
      <th>2016-05</th>
      <th>2016-06</th>
      <th>2016-07</th>
      <th>2016-08</th>
      <th>2016-09</th>
      <th>2016-10</th>
      <th>2016-11</th>
      <th>2016-12</th>
      <th>2017-01</th>
      <th>2017-02</th>
      <th>2017-03</th>
      <th>2017-04</th>
      <th>2017-05</th>
      <th>2017-06</th>
      <th>2017-07</th>
      <th>2017-08</th>
      <th>2017-09</th>
      <th>2017-10</th>
      <th>2017-11</th>
      <th>2017-12</th>
      <th>2018-01</th>
      <th>2018-02</th>
      <th>2018-03</th>
      <th>2018-04</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>84654</td>
      <td>60657</td>
      <td>Chicago</td>
      <td>IL</td>
      <td>Chicago</td>
      <td>Cook</td>
      <td>1</td>
      <td>334200.0</td>
      <td>335400.0</td>
      <td>336500.0</td>
      <td>337600.0</td>
      <td>338500.0</td>
      <td>339500.0</td>
      <td>340400.0</td>
      <td>341300.0</td>
      <td>342600.0</td>
      <td>344400.0</td>
      <td>345700.0</td>
      <td>346700.0</td>
      <td>347800.0</td>
      <td>349000.0</td>
      <td>350400.0</td>
      <td>352000.0</td>
      <td>353900.0</td>
      <td>356200.0</td>
      <td>358800.0</td>
      <td>361800.0</td>
      <td>365700.0</td>
      <td>370200.0</td>
      <td>374700.0</td>
      <td>378900.0</td>
      <td>383500.0</td>
      <td>388300.0</td>
      <td>393300.0</td>
      <td>398500.0</td>
      <td>403800.0</td>
      <td>409100.0</td>
      <td>414600.0</td>
      <td>420100.0</td>
      <td>426200.0</td>
      <td>432600.0</td>
      <td>438600.0</td>
      <td>444200.0</td>
      <td>450000.0</td>
      <td>455900.0</td>
      <td>462100.0</td>
      <td>468500.0</td>
      <td>475300.0</td>
      <td>482500.0</td>
      <td>490200.0</td>
      <td>498200.0</td>
      <td>507200.0</td>
      <td>516800.0</td>
      <td>526300.0</td>
      <td>535300.0</td>
      <td>544500.0</td>
      <td>553500.0</td>
      <td>562400.0</td>
      <td>571200.0</td>
      <td>579800.0</td>
      <td>588100.0</td>
      <td>596300.0</td>
      <td>604200.0</td>
      <td>612200.0</td>
      <td>620200.0</td>
      <td>627700.0</td>
      <td>634500.0</td>
      <td>641000.0</td>
      <td>647000.0</td>
      <td>652700.0</td>
      <td>658100.0</td>
      <td>663300.0</td>
      <td>668400.0</td>
      <td>673400.0</td>
      <td>678300.0</td>
      <td>683200.0</td>
      <td>688300.0</td>
      <td>693300.0</td>
      <td>698000.0</td>
      <td>702400.0</td>
      <td>706400.0</td>
      <td>710200.0</td>
      <td>714000.0</td>
      <td>717800.0</td>
      <td>721700.0</td>
      <td>725700.0</td>
      <td>729900.0</td>
      <td>733400.0</td>
      <td>735600.0</td>
      <td>737200.0</td>
      <td>739000.0</td>
      <td>740900.0</td>
      <td>742700.0</td>
      <td>744400.0</td>
      <td>746000.0</td>
      <td>747200.0</td>
      <td>748000.0</td>
      <td>749000.0</td>
      <td>750200.0</td>
      <td>752300.0</td>
      <td>755300.0</td>
      <td>759200.0</td>
      <td>764000.0</td>
      <td>769600.0</td>
      <td>775600.0</td>
      <td>781900.0</td>
      <td>787900.0</td>
      <td>793200.0</td>
      <td>798200.0</td>
      <td>803100.0</td>
      <td>807900.0</td>
      <td>812900.0</td>
      <td>818100.0</td>
      <td>823100.0</td>
      <td>828300.0</td>
      <td>834100.0</td>
      <td>839800.0</td>
      <td>845600.0</td>
      <td>851800.0</td>
      <td>858000.0</td>
      <td>864400.0</td>
      <td>870700.0</td>
      <td>876200.0</td>
      <td>880700.0</td>
      <td>884400.0</td>
      <td>887600.0</td>
      <td>890500.0</td>
      <td>893300.0</td>
      <td>895500.0</td>
      <td>897300.0</td>
      <td>899000.0</td>
      <td>900400.0</td>
      <td>902000.0</td>
      <td>904400.0</td>
      <td>907100.0</td>
      <td>909700.0</td>
      <td>911900.0</td>
      <td>913000.0</td>
      <td>913000.0</td>
      <td>912000.0</td>
      <td>909300.0</td>
      <td>905300.0</td>
      <td>901400.0</td>
      <td>897900.0</td>
      <td>895400.0</td>
      <td>893600.0</td>
      <td>891100.0</td>
      <td>887000.0</td>
      <td>881700.0</td>
      <td>875900.0</td>
      <td>870300.0</td>
      <td>865100.0</td>
      <td>859000.0</td>
      <td>851500.0</td>
      <td>843800.0</td>
      <td>836400.0</td>
      <td>830700.0</td>
      <td>827300.0</td>
      <td>824800.0</td>
      <td>821600.0</td>
      <td>818300.0</td>
      <td>814600.0</td>
      <td>809800.0</td>
      <td>803600.0</td>
      <td>795500.0</td>
      <td>786900.0</td>
      <td>780700.0</td>
      <td>776900.0</td>
      <td>774700.0</td>
      <td>774200.0</td>
      <td>774400.0</td>
      <td>774600.0</td>
      <td>775600.0</td>
      <td>777800.0</td>
      <td>775200.0</td>
      <td>767900.0</td>
      <td>764700.0</td>
      <td>766100.0</td>
      <td>764100.0</td>
      <td>759700.0</td>
      <td>754900.0</td>
      <td>746200.0</td>
      <td>737300.0</td>
      <td>730800.0</td>
      <td>729300.0</td>
      <td>730200.0</td>
      <td>730700.0</td>
      <td>730000.0</td>
      <td>730100.0</td>
      <td>730100.0</td>
      <td>731200.0</td>
      <td>733900.0</td>
      <td>735500.0</td>
      <td>735400.0</td>
      <td>734400.0</td>
      <td>737500.0</td>
      <td>737700.0</td>
      <td>733700.0</td>
      <td>734000.0</td>
      <td>740300.0</td>
      <td>744600.0</td>
      <td>750500.0</td>
      <td>760400.0</td>
      <td>771800.0</td>
      <td>780600.0</td>
      <td>787900.0</td>
      <td>794100.0</td>
      <td>798900.0</td>
      <td>802300.0</td>
      <td>806100.0</td>
      <td>810900.0</td>
      <td>817400.0</td>
      <td>826800.0</td>
      <td>837900.0</td>
      <td>848100.0</td>
      <td>853800.0</td>
      <td>856700.0</td>
      <td>856600.0</td>
      <td>854400.0</td>
      <td>853000.0</td>
      <td>856200.0</td>
      <td>859700.0</td>
      <td>863900.0</td>
      <td>872900.0</td>
      <td>883300.0</td>
      <td>889500.0</td>
      <td>892800</td>
      <td>893600</td>
      <td>891300</td>
      <td>889900</td>
      <td>891500</td>
      <td>893000</td>
      <td>893000</td>
      <td>895000</td>
      <td>901200</td>
      <td>909400</td>
      <td>915000</td>
      <td>916700</td>
      <td>917700</td>
      <td>919800</td>
      <td>925800</td>
      <td>937100</td>
      <td>948200</td>
      <td>951000</td>
      <td>952500</td>
      <td>958600</td>
      <td>966200</td>
      <td>970400</td>
      <td>973900</td>
      <td>974700</td>
      <td>972600</td>
      <td>974300</td>
      <td>980800</td>
      <td>988000</td>
      <td>994700</td>
      <td>998700</td>
      <td>997000</td>
      <td>993700</td>
      <td>991300</td>
      <td>989200</td>
      <td>991300</td>
      <td>999100</td>
      <td>1005500</td>
      <td>1007500</td>
      <td>1007800</td>
      <td>1009600</td>
      <td>1013300</td>
      <td>1018700</td>
      <td>1024400</td>
      <td>1030700</td>
      <td>1033800</td>
      <td>1030600</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Find out the size so I know what I'm dealing with.
zdf.shape
```




    (14723, 272)




```python
# An example subset of a single city's zipcode.
zdf[(zdf['City'] == 'Portsmouth') & (zdf['State'] == 'NH')]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>RegionID</th>
      <th>RegionName</th>
      <th>City</th>
      <th>State</th>
      <th>Metro</th>
      <th>CountyName</th>
      <th>SizeRank</th>
      <th>1996-04</th>
      <th>1996-05</th>
      <th>1996-06</th>
      <th>1996-07</th>
      <th>1996-08</th>
      <th>1996-09</th>
      <th>1996-10</th>
      <th>1996-11</th>
      <th>1996-12</th>
      <th>1997-01</th>
      <th>1997-02</th>
      <th>1997-03</th>
      <th>1997-04</th>
      <th>1997-05</th>
      <th>1997-06</th>
      <th>1997-07</th>
      <th>1997-08</th>
      <th>1997-09</th>
      <th>1997-10</th>
      <th>1997-11</th>
      <th>1997-12</th>
      <th>1998-01</th>
      <th>1998-02</th>
      <th>1998-03</th>
      <th>1998-04</th>
      <th>1998-05</th>
      <th>1998-06</th>
      <th>1998-07</th>
      <th>1998-08</th>
      <th>1998-09</th>
      <th>1998-10</th>
      <th>1998-11</th>
      <th>1998-12</th>
      <th>1999-01</th>
      <th>1999-02</th>
      <th>1999-03</th>
      <th>1999-04</th>
      <th>1999-05</th>
      <th>1999-06</th>
      <th>1999-07</th>
      <th>1999-08</th>
      <th>1999-09</th>
      <th>1999-10</th>
      <th>1999-11</th>
      <th>1999-12</th>
      <th>2000-01</th>
      <th>2000-02</th>
      <th>2000-03</th>
      <th>2000-04</th>
      <th>2000-05</th>
      <th>2000-06</th>
      <th>2000-07</th>
      <th>2000-08</th>
      <th>2000-09</th>
      <th>2000-10</th>
      <th>2000-11</th>
      <th>2000-12</th>
      <th>2001-01</th>
      <th>2001-02</th>
      <th>2001-03</th>
      <th>2001-04</th>
      <th>2001-05</th>
      <th>2001-06</th>
      <th>2001-07</th>
      <th>2001-08</th>
      <th>2001-09</th>
      <th>2001-10</th>
      <th>2001-11</th>
      <th>2001-12</th>
      <th>2002-01</th>
      <th>2002-02</th>
      <th>2002-03</th>
      <th>2002-04</th>
      <th>2002-05</th>
      <th>2002-06</th>
      <th>2002-07</th>
      <th>2002-08</th>
      <th>2002-09</th>
      <th>2002-10</th>
      <th>2002-11</th>
      <th>2002-12</th>
      <th>2003-01</th>
      <th>2003-02</th>
      <th>2003-03</th>
      <th>2003-04</th>
      <th>2003-05</th>
      <th>2003-06</th>
      <th>2003-07</th>
      <th>2003-08</th>
      <th>2003-09</th>
      <th>2003-10</th>
      <th>2003-11</th>
      <th>2003-12</th>
      <th>2004-01</th>
      <th>2004-02</th>
      <th>2004-03</th>
      <th>2004-04</th>
      <th>2004-05</th>
      <th>2004-06</th>
      <th>2004-07</th>
      <th>2004-08</th>
      <th>2004-09</th>
      <th>2004-10</th>
      <th>2004-11</th>
      <th>2004-12</th>
      <th>2005-01</th>
      <th>2005-02</th>
      <th>2005-03</th>
      <th>2005-04</th>
      <th>2005-05</th>
      <th>2005-06</th>
      <th>2005-07</th>
      <th>2005-08</th>
      <th>2005-09</th>
      <th>2005-10</th>
      <th>2005-11</th>
      <th>2005-12</th>
      <th>2006-01</th>
      <th>2006-02</th>
      <th>2006-03</th>
      <th>2006-04</th>
      <th>2006-05</th>
      <th>2006-06</th>
      <th>2006-07</th>
      <th>2006-08</th>
      <th>2006-09</th>
      <th>2006-10</th>
      <th>2006-11</th>
      <th>2006-12</th>
      <th>2007-01</th>
      <th>2007-02</th>
      <th>2007-03</th>
      <th>2007-04</th>
      <th>2007-05</th>
      <th>2007-06</th>
      <th>2007-07</th>
      <th>2007-08</th>
      <th>2007-09</th>
      <th>2007-10</th>
      <th>2007-11</th>
      <th>2007-12</th>
      <th>2008-01</th>
      <th>2008-02</th>
      <th>2008-03</th>
      <th>2008-04</th>
      <th>2008-05</th>
      <th>2008-06</th>
      <th>2008-07</th>
      <th>2008-08</th>
      <th>2008-09</th>
      <th>2008-10</th>
      <th>2008-11</th>
      <th>2008-12</th>
      <th>2009-01</th>
      <th>2009-02</th>
      <th>2009-03</th>
      <th>2009-04</th>
      <th>2009-05</th>
      <th>2009-06</th>
      <th>2009-07</th>
      <th>2009-08</th>
      <th>2009-09</th>
      <th>2009-10</th>
      <th>2009-11</th>
      <th>2009-12</th>
      <th>2010-01</th>
      <th>2010-02</th>
      <th>2010-03</th>
      <th>2010-04</th>
      <th>2010-05</th>
      <th>2010-06</th>
      <th>2010-07</th>
      <th>2010-08</th>
      <th>2010-09</th>
      <th>2010-10</th>
      <th>2010-11</th>
      <th>2010-12</th>
      <th>2011-01</th>
      <th>2011-02</th>
      <th>2011-03</th>
      <th>2011-04</th>
      <th>2011-05</th>
      <th>2011-06</th>
      <th>2011-07</th>
      <th>2011-08</th>
      <th>2011-09</th>
      <th>2011-10</th>
      <th>2011-11</th>
      <th>2011-12</th>
      <th>2012-01</th>
      <th>2012-02</th>
      <th>2012-03</th>
      <th>2012-04</th>
      <th>2012-05</th>
      <th>2012-06</th>
      <th>2012-07</th>
      <th>2012-08</th>
      <th>2012-09</th>
      <th>2012-10</th>
      <th>2012-11</th>
      <th>2012-12</th>
      <th>2013-01</th>
      <th>2013-02</th>
      <th>2013-03</th>
      <th>2013-04</th>
      <th>2013-05</th>
      <th>2013-06</th>
      <th>2013-07</th>
      <th>2013-08</th>
      <th>2013-09</th>
      <th>2013-10</th>
      <th>2013-11</th>
      <th>2013-12</th>
      <th>2014-01</th>
      <th>2014-02</th>
      <th>2014-03</th>
      <th>2014-04</th>
      <th>2014-05</th>
      <th>2014-06</th>
      <th>2014-07</th>
      <th>2014-08</th>
      <th>2014-09</th>
      <th>2014-10</th>
      <th>2014-11</th>
      <th>2014-12</th>
      <th>2015-01</th>
      <th>2015-02</th>
      <th>2015-03</th>
      <th>2015-04</th>
      <th>2015-05</th>
      <th>2015-06</th>
      <th>2015-07</th>
      <th>2015-08</th>
      <th>2015-09</th>
      <th>2015-10</th>
      <th>2015-11</th>
      <th>2015-12</th>
      <th>2016-01</th>
      <th>2016-02</th>
      <th>2016-03</th>
      <th>2016-04</th>
      <th>2016-05</th>
      <th>2016-06</th>
      <th>2016-07</th>
      <th>2016-08</th>
      <th>2016-09</th>
      <th>2016-10</th>
      <th>2016-11</th>
      <th>2016-12</th>
      <th>2017-01</th>
      <th>2017-02</th>
      <th>2017-03</th>
      <th>2017-04</th>
      <th>2017-05</th>
      <th>2017-06</th>
      <th>2017-07</th>
      <th>2017-08</th>
      <th>2017-09</th>
      <th>2017-10</th>
      <th>2017-11</th>
      <th>2017-12</th>
      <th>2018-01</th>
      <th>2018-02</th>
      <th>2018-03</th>
      <th>2018-04</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3835</th>
      <td>59202</td>
      <td>3801</td>
      <td>Portsmouth</td>
      <td>NH</td>
      <td>Boston</td>
      <td>Rockingham</td>
      <td>3836</td>
      <td>127500.0</td>
      <td>128100.0</td>
      <td>128700.0</td>
      <td>129300.0</td>
      <td>129700.0</td>
      <td>130100.0</td>
      <td>130600.0</td>
      <td>131000.0</td>
      <td>131600.0</td>
      <td>132300.0</td>
      <td>133000.0</td>
      <td>133600.0</td>
      <td>134200.0</td>
      <td>134900.0</td>
      <td>135600.0</td>
      <td>136400.0</td>
      <td>137400.0</td>
      <td>138400.0</td>
      <td>139500.0</td>
      <td>140800.0</td>
      <td>142500.0</td>
      <td>144300.0</td>
      <td>146100.0</td>
      <td>147800.0</td>
      <td>149600.0</td>
      <td>151300.0</td>
      <td>153100.0</td>
      <td>154900.0</td>
      <td>156600.0</td>
      <td>158300.0</td>
      <td>160000.0</td>
      <td>161800.0</td>
      <td>163700.0</td>
      <td>165800.0</td>
      <td>167800.0</td>
      <td>169700.0</td>
      <td>171700.0</td>
      <td>173700.0</td>
      <td>175900.0</td>
      <td>178100.0</td>
      <td>180400.0</td>
      <td>182800.0</td>
      <td>185300.0</td>
      <td>187800.0</td>
      <td>190600.0</td>
      <td>193400.0</td>
      <td>196000.0</td>
      <td>198300.0</td>
      <td>200400.0</td>
      <td>202200.0</td>
      <td>203900.0</td>
      <td>205500.0</td>
      <td>206800.0</td>
      <td>208100.0</td>
      <td>209500.0</td>
      <td>210900.0</td>
      <td>212700.0</td>
      <td>214700.0</td>
      <td>216700.0</td>
      <td>218900.0</td>
      <td>221200.0</td>
      <td>223600.0</td>
      <td>226100.0</td>
      <td>228700.0</td>
      <td>231300.0</td>
      <td>234000.0</td>
      <td>236700.0</td>
      <td>239500.0</td>
      <td>242400.0</td>
      <td>245200.0</td>
      <td>247900.0</td>
      <td>250500.0</td>
      <td>253000.0</td>
      <td>255500.0</td>
      <td>257800.0</td>
      <td>260100.0</td>
      <td>262500.0</td>
      <td>264800.0</td>
      <td>267300.0</td>
      <td>269900.0</td>
      <td>272900.0</td>
      <td>276100.0</td>
      <td>279400.0</td>
      <td>282400.0</td>
      <td>284900.0</td>
      <td>286900.0</td>
      <td>288700.0</td>
      <td>290800.0</td>
      <td>293100.0</td>
      <td>295900.0</td>
      <td>298900.0</td>
      <td>301800.0</td>
      <td>304200.0</td>
      <td>306000.0</td>
      <td>307000.0</td>
      <td>307200.0</td>
      <td>307100.0</td>
      <td>307100.0</td>
      <td>307500.0</td>
      <td>308500.0</td>
      <td>309900.0</td>
      <td>311900.0</td>
      <td>314200.0</td>
      <td>316700.0</td>
      <td>319400.0</td>
      <td>322400.0</td>
      <td>326100.0</td>
      <td>330600.0</td>
      <td>336100.0</td>
      <td>342200.0</td>
      <td>348400.0</td>
      <td>354000.0</td>
      <td>358500.0</td>
      <td>361600.0</td>
      <td>363000.0</td>
      <td>363000.0</td>
      <td>361600.0</td>
      <td>359500.0</td>
      <td>357200.0</td>
      <td>355000.0</td>
      <td>352700.0</td>
      <td>349700.0</td>
      <td>345400.0</td>
      <td>339800.0</td>
      <td>333800.0</td>
      <td>328400.0</td>
      <td>324000.0</td>
      <td>320900.0</td>
      <td>318800.0</td>
      <td>317400.0</td>
      <td>316700.0</td>
      <td>316400.0</td>
      <td>316500.0</td>
      <td>317400.0</td>
      <td>319300.0</td>
      <td>322000.0</td>
      <td>324800.0</td>
      <td>327300.0</td>
      <td>329000.0</td>
      <td>329700.0</td>
      <td>329900.0</td>
      <td>329900.0</td>
      <td>329700.0</td>
      <td>329300.0</td>
      <td>328600.0</td>
      <td>327700.0</td>
      <td>326700.0</td>
      <td>325000.0</td>
      <td>322600.0</td>
      <td>320400.0</td>
      <td>318400.0</td>
      <td>316700.0</td>
      <td>315400.0</td>
      <td>314400.0</td>
      <td>313500.0</td>
      <td>312200.0</td>
      <td>310300.0</td>
      <td>308000.0</td>
      <td>306300.0</td>
      <td>306000.0</td>
      <td>306600.0</td>
      <td>307000.0</td>
      <td>306900.0</td>
      <td>306400.0</td>
      <td>306000.0</td>
      <td>306100.0</td>
      <td>306100.0</td>
      <td>307600.0</td>
      <td>311100.0</td>
      <td>315200.0</td>
      <td>317200.0</td>
      <td>318200.0</td>
      <td>318600.0</td>
      <td>318800.0</td>
      <td>319200.0</td>
      <td>318500.0</td>
      <td>316300.0</td>
      <td>315000.0</td>
      <td>314600.0</td>
      <td>312800.0</td>
      <td>310200.0</td>
      <td>306200.0</td>
      <td>302100.0</td>
      <td>299700.0</td>
      <td>298500.0</td>
      <td>296300.0</td>
      <td>295600.0</td>
      <td>298700.0</td>
      <td>303700.0</td>
      <td>305700.0</td>
      <td>305900.0</td>
      <td>306200.0</td>
      <td>306600.0</td>
      <td>308900.0</td>
      <td>312800.0</td>
      <td>315400.0</td>
      <td>316900.0</td>
      <td>319300.0</td>
      <td>321500.0</td>
      <td>322600.0</td>
      <td>322800.0</td>
      <td>323200.0</td>
      <td>324100.0</td>
      <td>325400.0</td>
      <td>326700.0</td>
      <td>328100.0</td>
      <td>329500.0</td>
      <td>331100.0</td>
      <td>333700.0</td>
      <td>336800.0</td>
      <td>339200.0</td>
      <td>341600.0</td>
      <td>344400.0</td>
      <td>347600.0</td>
      <td>350000.0</td>
      <td>351000.0</td>
      <td>351600.0</td>
      <td>352700.0</td>
      <td>354900.0</td>
      <td>357800</td>
      <td>360000</td>
      <td>361800</td>
      <td>363400</td>
      <td>363600</td>
      <td>363000</td>
      <td>363500</td>
      <td>365600</td>
      <td>367900</td>
      <td>369600</td>
      <td>372200</td>
      <td>374800</td>
      <td>377400</td>
      <td>380500</td>
      <td>384300</td>
      <td>387300</td>
      <td>389100</td>
      <td>390900</td>
      <td>392900</td>
      <td>393600</td>
      <td>393700</td>
      <td>395600</td>
      <td>398900</td>
      <td>402700</td>
      <td>405800</td>
      <td>405900</td>
      <td>402600</td>
      <td>399500</td>
      <td>399200</td>
      <td>401300</td>
      <td>405400</td>
      <td>413000</td>
      <td>421500</td>
      <td>426700</td>
      <td>429000</td>
      <td>430500</td>
      <td>431500</td>
      <td>435100</td>
      <td>443600</td>
      <td>453600</td>
      <td>461400</td>
      <td>466400</td>
      <td>469100</td>
      <td>472000</td>
      <td>478000</td>
      <td>484200</td>
    </tr>
  </tbody>
</table>
</div>



The provided heading says to filter to the chosen zipcodes. I'm actually not going to do that right now. I think I can go for broke and chase the top zips in the entire set.

# Data Preprocessing


```python
# I'll need to marry the metadata to the time series later on. 
zip_info = pd.DataFrame(zdf[['RegionName', 'City', 'State', 'Metro', 
                             'CountyName', 'SizeRank']])
```

Here's where the data gets melted from wide into long format. Dates go from the rows into the columns, such that each zipcode has numerous months associated with it.


```python
def melt_data(df):
    melted = pd.melt(df, id_vars=['RegionID','RegionName', 'City', 'State', 
                                  'Metro', 'CountyName', 
                                  'SizeRank'], var_name='Month', 
                     value_name='MeanValue')
    melted['Month'] = pd.to_datetime(melted['Month'], format='%Y-%m')
    melted = melted.dropna(subset=['MeanValue'])
    return melted
```


```python
zdf_melted = melt_data(zdf)
```


```python
zdf_melted.head(1)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>RegionID</th>
      <th>RegionName</th>
      <th>City</th>
      <th>State</th>
      <th>Metro</th>
      <th>CountyName</th>
      <th>SizeRank</th>
      <th>Month</th>
      <th>MeanValue</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>84654</td>
      <td>60657</td>
      <td>Chicago</td>
      <td>IL</td>
      <td>Chicago</td>
      <td>Cook</td>
      <td>1</td>
      <td>1996-04-01</td>
      <td>334200.0</td>
    </tr>
  </tbody>
</table>
</div>



I'm not going to do anything with that melted data for now. I actually want everything together with the time series broken out for each zipcode. That way I can have an easy-to-get column for each zip, and I can also easily put the national mean in its own column.


```python
# By pivoting the data from where it is in long format, 
# I can easily get the format I want.
zdf_by_zip = zdf_melted.pivot('Month', 'RegionName', 'MeanValue')
```


```python
zdf_by_zip.head(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>RegionName</th>
      <th>1001</th>
      <th>1002</th>
      <th>1005</th>
      <th>1007</th>
      <th>1008</th>
      <th>1010</th>
      <th>1011</th>
      <th>1013</th>
      <th>1020</th>
      <th>1026</th>
      <th>1027</th>
      <th>1028</th>
      <th>1030</th>
      <th>1033</th>
      <th>1034</th>
      <th>1035</th>
      <th>1036</th>
      <th>1038</th>
      <th>1040</th>
      <th>1050</th>
      <th>1053</th>
      <th>1054</th>
      <th>1056</th>
      <th>1060</th>
      <th>1062</th>
      <th>1068</th>
      <th>1071</th>
      <th>1072</th>
      <th>1073</th>
      <th>1075</th>
      <th>1077</th>
      <th>1081</th>
      <th>1082</th>
      <th>1083</th>
      <th>1085</th>
      <th>1089</th>
      <th>1092</th>
      <th>1095</th>
      <th>1096</th>
      <th>1098</th>
      <th>1104</th>
      <th>1106</th>
      <th>1107</th>
      <th>1108</th>
      <th>1109</th>
      <th>1118</th>
      <th>1119</th>
      <th>1128</th>
      <th>1129</th>
      <th>1151</th>
      <th>1201</th>
      <th>1220</th>
      <th>1223</th>
      <th>1225</th>
      <th>1226</th>
      <th>1230</th>
      <th>1235</th>
      <th>1238</th>
      <th>1240</th>
      <th>1245</th>
      <th>1247</th>
      <th>1254</th>
      <th>1255</th>
      <th>1257</th>
      <th>1262</th>
      <th>1266</th>
      <th>1267</th>
      <th>1270</th>
      <th>1301</th>
      <th>1330</th>
      <th>1331</th>
      <th>1337</th>
      <th>1338</th>
      <th>1339</th>
      <th>1340</th>
      <th>1341</th>
      <th>1344</th>
      <th>1351</th>
      <th>1354</th>
      <th>1360</th>
      <th>1364</th>
      <th>1366</th>
      <th>1368</th>
      <th>1370</th>
      <th>1373</th>
      <th>1375</th>
      <th>1376</th>
      <th>1420</th>
      <th>1430</th>
      <th>1431</th>
      <th>1432</th>
      <th>1440</th>
      <th>1450</th>
      <th>1452</th>
      <th>1453</th>
      <th>1460</th>
      <th>1462</th>
      <th>1463</th>
      <th>1464</th>
      <th>1468</th>
      <th>1469</th>
      <th>1473</th>
      <th>1474</th>
      <th>1475</th>
      <th>1501</th>
      <th>1503</th>
      <th>1504</th>
      <th>1505</th>
      <th>1506</th>
      <th>1507</th>
      <th>1510</th>
      <th>1515</th>
      <th>1516</th>
      <th>1518</th>
      <th>1519</th>
      <th>1520</th>
      <th>1521</th>
      <th>1522</th>
      <th>1523</th>
      <th>1524</th>
      <th>1527</th>
      <th>1529</th>
      <th>1532</th>
      <th>1534</th>
      <th>1535</th>
      <th>1536</th>
      <th>1537</th>
      <th>1540</th>
      <th>1541</th>
      <th>1542</th>
      <th>1545</th>
      <th>1550</th>
      <th>1560</th>
      <th>1562</th>
      <th>1564</th>
      <th>1566</th>
      <th>1568</th>
      <th>1569</th>
      <th>1571</th>
      <th>1581</th>
      <th>1583</th>
      <th>1585</th>
      <th>1588</th>
      <th>1590</th>
      <th>1602</th>
      <th>1603</th>
      <th>1604</th>
      <th>1605</th>
      <th>1606</th>
      <th>1607</th>
      <th>1609</th>
      <th>1610</th>
      <th>1611</th>
      <th>1612</th>
      <th>1701</th>
      <th>1720</th>
      <th>1730</th>
      <th>1740</th>
      <th>1741</th>
      <th>1742</th>
      <th>1746</th>
      <th>1747</th>
      <th>1748</th>
      <th>1749</th>
      <th>1752</th>
      <th>1754</th>
      <th>1756</th>
      <th>1757</th>
      <th>1760</th>
      <th>1770</th>
      <th>1772</th>
      <th>1773</th>
      <th>1775</th>
      <th>1776</th>
      <th>1778</th>
      <th>1801</th>
      <th>1803</th>
      <th>1810</th>
      <th>1821</th>
      <th>1824</th>
      <th>1826</th>
      <th>1827</th>
      <th>1830</th>
      <th>1832</th>
      <th>1833</th>
      <th>1835</th>
      <th>1841</th>
      <th>1843</th>
      <th>1844</th>
      <th>1845</th>
      <th>1850</th>
      <th>1851</th>
      <th>1852</th>
      <th>1854</th>
      <th>1860</th>
      <th>1862</th>
      <th>1863</th>
      <th>1864</th>
      <th>1867</th>
      <th>1876</th>
      <th>1879</th>
      <th>1880</th>
      <th>1886</th>
      <th>1887</th>
      <th>1890</th>
      <th>1902</th>
      <th>1904</th>
      <th>1905</th>
      <th>1906</th>
      <th>1907</th>
      <th>1908</th>
      <th>1913</th>
      <th>1915</th>
      <th>1921</th>
      <th>1922</th>
      <th>1923</th>
      <th>1930</th>
      <th>1938</th>
      <th>1940</th>
      <th>1944</th>
      <th>1945</th>
      <th>1949</th>
      <th>1950</th>
      <th>1951</th>
      <th>1952</th>
      <th>1960</th>
      <th>1966</th>
      <th>1969</th>
      <th>1970</th>
      <th>1982</th>
      <th>1983</th>
      <th>1984</th>
      <th>1985</th>
      <th>2019</th>
      <th>2021</th>
      <th>2025</th>
      <th>2026</th>
      <th>2030</th>
      <th>2032</th>
      <th>2035</th>
      <th>2038</th>
      <th>2043</th>
      <th>2045</th>
      <th>2048</th>
      <th>2050</th>
      <th>2052</th>
      <th>2053</th>
      <th>2054</th>
      <th>2056</th>
      <th>2061</th>
      <th>...</th>
      <th>98277</th>
      <th>98279</th>
      <th>98281</th>
      <th>98282</th>
      <th>98284</th>
      <th>98290</th>
      <th>98292</th>
      <th>98294</th>
      <th>98295</th>
      <th>98296</th>
      <th>98303</th>
      <th>98310</th>
      <th>98311</th>
      <th>98312</th>
      <th>98320</th>
      <th>98321</th>
      <th>98327</th>
      <th>98328</th>
      <th>98329</th>
      <th>98332</th>
      <th>98333</th>
      <th>98335</th>
      <th>98337</th>
      <th>98338</th>
      <th>98339</th>
      <th>98340</th>
      <th>98342</th>
      <th>98346</th>
      <th>98349</th>
      <th>98351</th>
      <th>98354</th>
      <th>98356</th>
      <th>98358</th>
      <th>98359</th>
      <th>98360</th>
      <th>98362</th>
      <th>98363</th>
      <th>98365</th>
      <th>98366</th>
      <th>98367</th>
      <th>98368</th>
      <th>98370</th>
      <th>98371</th>
      <th>98372</th>
      <th>98373</th>
      <th>98374</th>
      <th>98375</th>
      <th>98376</th>
      <th>98380</th>
      <th>98382</th>
      <th>98383</th>
      <th>98387</th>
      <th>98388</th>
      <th>98390</th>
      <th>98391</th>
      <th>98392</th>
      <th>98394</th>
      <th>98402</th>
      <th>98403</th>
      <th>98404</th>
      <th>98405</th>
      <th>98406</th>
      <th>98407</th>
      <th>98408</th>
      <th>98409</th>
      <th>98418</th>
      <th>98422</th>
      <th>98424</th>
      <th>98443</th>
      <th>98444</th>
      <th>98445</th>
      <th>98446</th>
      <th>98465</th>
      <th>98466</th>
      <th>98467</th>
      <th>98498</th>
      <th>98499</th>
      <th>98501</th>
      <th>98502</th>
      <th>98503</th>
      <th>98506</th>
      <th>98512</th>
      <th>98513</th>
      <th>98516</th>
      <th>98520</th>
      <th>98524</th>
      <th>98528</th>
      <th>98531</th>
      <th>98532</th>
      <th>98537</th>
      <th>98541</th>
      <th>98547</th>
      <th>98550</th>
      <th>98557</th>
      <th>98563</th>
      <th>98564</th>
      <th>98568</th>
      <th>98569</th>
      <th>98570</th>
      <th>98576</th>
      <th>98579</th>
      <th>98580</th>
      <th>98584</th>
      <th>98588</th>
      <th>98589</th>
      <th>98592</th>
      <th>98595</th>
      <th>98596</th>
      <th>98597</th>
      <th>98601</th>
      <th>98604</th>
      <th>98606</th>
      <th>98607</th>
      <th>98610</th>
      <th>98611</th>
      <th>98625</th>
      <th>98626</th>
      <th>98629</th>
      <th>98632</th>
      <th>98642</th>
      <th>98645</th>
      <th>98648</th>
      <th>98660</th>
      <th>98661</th>
      <th>98662</th>
      <th>98663</th>
      <th>98664</th>
      <th>98665</th>
      <th>98671</th>
      <th>98674</th>
      <th>98675</th>
      <th>98682</th>
      <th>98683</th>
      <th>98684</th>
      <th>98685</th>
      <th>98686</th>
      <th>98801</th>
      <th>98802</th>
      <th>98812</th>
      <th>98815</th>
      <th>98822</th>
      <th>98823</th>
      <th>98826</th>
      <th>98828</th>
      <th>98837</th>
      <th>98840</th>
      <th>98841</th>
      <th>98844</th>
      <th>98847</th>
      <th>98848</th>
      <th>98851</th>
      <th>98855</th>
      <th>98856</th>
      <th>98862</th>
      <th>98901</th>
      <th>98902</th>
      <th>98903</th>
      <th>98908</th>
      <th>98922</th>
      <th>98925</th>
      <th>98926</th>
      <th>98930</th>
      <th>98932</th>
      <th>98936</th>
      <th>98937</th>
      <th>98940</th>
      <th>98941</th>
      <th>98942</th>
      <th>98944</th>
      <th>98947</th>
      <th>98948</th>
      <th>98951</th>
      <th>98953</th>
      <th>99001</th>
      <th>99003</th>
      <th>99004</th>
      <th>99005</th>
      <th>99006</th>
      <th>99009</th>
      <th>99016</th>
      <th>99019</th>
      <th>99021</th>
      <th>99022</th>
      <th>99025</th>
      <th>99026</th>
      <th>99027</th>
      <th>99037</th>
      <th>99109</th>
      <th>99110</th>
      <th>99114</th>
      <th>99116</th>
      <th>99123</th>
      <th>99141</th>
      <th>99148</th>
      <th>99163</th>
      <th>99181</th>
      <th>99201</th>
      <th>99202</th>
      <th>99203</th>
      <th>99204</th>
      <th>99205</th>
      <th>99206</th>
      <th>99207</th>
      <th>99208</th>
      <th>99212</th>
      <th>99216</th>
      <th>99217</th>
      <th>99218</th>
      <th>99223</th>
      <th>99224</th>
      <th>99301</th>
      <th>99320</th>
      <th>99323</th>
      <th>99324</th>
      <th>99336</th>
      <th>99337</th>
      <th>99338</th>
      <th>99350</th>
      <th>99352</th>
      <th>99354</th>
      <th>99361</th>
      <th>99362</th>
      <th>99501</th>
      <th>99502</th>
      <th>99503</th>
      <th>99504</th>
      <th>99507</th>
      <th>99508</th>
      <th>99515</th>
      <th>99516</th>
      <th>99517</th>
      <th>99518</th>
      <th>99567</th>
      <th>99577</th>
      <th>99587</th>
      <th>99603</th>
      <th>99611</th>
      <th>99615</th>
      <th>99623</th>
      <th>99645</th>
      <th>99654</th>
      <th>99664</th>
      <th>99669</th>
      <th>99701</th>
      <th>99705</th>
      <th>99709</th>
      <th>99712</th>
      <th>99801</th>
      <th>99835</th>
      <th>99901</th>
    </tr>
    <tr>
      <th>Month</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1996-04-01</th>
      <td>113100.0</td>
      <td>161000.0</td>
      <td>103100.0</td>
      <td>133400.0</td>
      <td>117500.0</td>
      <td>115800.0</td>
      <td>87000.0</td>
      <td>88500.0</td>
      <td>97200.0</td>
      <td>91200.0</td>
      <td>117600.0</td>
      <td>129100.0</td>
      <td>121700.0</td>
      <td>122200.0</td>
      <td>128400.0</td>
      <td>149900.0</td>
      <td>135400.0</td>
      <td>127600.0</td>
      <td>93700.0</td>
      <td>108100.0</td>
      <td>141000.0</td>
      <td>141300.0</td>
      <td>111100.0</td>
      <td>141100.0</td>
      <td>127000.0</td>
      <td>124400.0</td>
      <td>117200.0</td>
      <td>113300.0</td>
      <td>142100.0</td>
      <td>118900.0</td>
      <td>122900.0</td>
      <td>99900.0</td>
      <td>105300.0</td>
      <td>99100.0</td>
      <td>128100.0</td>
      <td>110000.0</td>
      <td>99600.0</td>
      <td>151100.0</td>
      <td>113500.0</td>
      <td>119900.0</td>
      <td>65100.0</td>
      <td>176900.0</td>
      <td>74100.0</td>
      <td>73300.0</td>
      <td>60200.0</td>
      <td>83400.0</td>
      <td>76400.0</td>
      <td>90800.0</td>
      <td>87500.0</td>
      <td>67500.0</td>
      <td>91600.0</td>
      <td>88600.0</td>
      <td>97300.0</td>
      <td>113400.0</td>
      <td>102300.0</td>
      <td>133900.0</td>
      <td>99400.0</td>
      <td>117200.0</td>
      <td>158900.0</td>
      <td>166700.0</td>
      <td>81800.0</td>
      <td>168800.0</td>
      <td>124700.0</td>
      <td>121000.0</td>
      <td>163900.0</td>
      <td>165900.0</td>
      <td>130100.0</td>
      <td>108600.0</td>
      <td>100000.0</td>
      <td>113800.0</td>
      <td>76300.0</td>
      <td>104800.0</td>
      <td>94600.0</td>
      <td>67300.0</td>
      <td>84500.0</td>
      <td>117900.0</td>
      <td>83100.0</td>
      <td>100300.0</td>
      <td>100500.0</td>
      <td>91900.0</td>
      <td>79000.0</td>
      <td>113500.0</td>
      <td>88000.0</td>
      <td>99300.0</td>
      <td>128000.0</td>
      <td>126000.0</td>
      <td>95900.0</td>
      <td>83500.0</td>
      <td>101800.0</td>
      <td>112400.0</td>
      <td>130400.0</td>
      <td>86100.0</td>
      <td>216600.0</td>
      <td>114500.0</td>
      <td>111000.0</td>
      <td>193800.0</td>
      <td>119000.0</td>
      <td>150000.0</td>
      <td>144200.0</td>
      <td>93700.0</td>
      <td>127900.0</td>
      <td>113400.0</td>
      <td>128400.0</td>
      <td>89900.0</td>
      <td>110500.0</td>
      <td>179900.0</td>
      <td>123600.0</td>
      <td>164000.0</td>
      <td>107900.0</td>
      <td>125200.0</td>
      <td>114200.0</td>
      <td>106800.0</td>
      <td>123100.0</td>
      <td>129200.0</td>
      <td>175700.0</td>
      <td>134400.0</td>
      <td>97800.0</td>
      <td>145100.0</td>
      <td>141900.0</td>
      <td>110000.0</td>
      <td>117500.0</td>
      <td>133200.0</td>
      <td>178900.0</td>
      <td>143200.0</td>
      <td>103500.0</td>
      <td>140300.0</td>
      <td>108200.0</td>
      <td>107300.0</td>
      <td>169200.0</td>
      <td>116200.0</td>
      <td>164600.0</td>
      <td>95800.0</td>
      <td>171700.0</td>
      <td>107500.0</td>
      <td>150100.0</td>
      <td>123800.0</td>
      <td>187000.0</td>
      <td>134500.0</td>
      <td>115300.0</td>
      <td>204500.0</td>
      <td>126700.0</td>
      <td>107300.0</td>
      <td>125600.0</td>
      <td>150100.0</td>
      <td>101500.0</td>
      <td>85000.0</td>
      <td>90000.0</td>
      <td>93800.0</td>
      <td>92700.0</td>
      <td>87600.0</td>
      <td>145600.0</td>
      <td>77400.0</td>
      <td>96400.0</td>
      <td>136800.0</td>
      <td>169500.0</td>
      <td>265500.0</td>
      <td>262000.0</td>
      <td>259400.0</td>
      <td>367600.0</td>
      <td>384000.0</td>
      <td>185600.0</td>
      <td>149000.0</td>
      <td>258700.0</td>
      <td>145000.0</td>
      <td>150100.0</td>
      <td>152100.0</td>
      <td>176800.0</td>
      <td>140500.0</td>
      <td>197600.0</td>
      <td>395600.0</td>
      <td>277100.0</td>
      <td>545700.0</td>
      <td>227400.0</td>
      <td>342100.0</td>
      <td>289500.0</td>
      <td>163900.0</td>
      <td>185500.0</td>
      <td>260500.0</td>
      <td>153400.0</td>
      <td>172100.0</td>
      <td>128900.0</td>
      <td>191700.0</td>
      <td>121800.0</td>
      <td>125200.0</td>
      <td>182800.0</td>
      <td>129500.0</td>
      <td>66100.0</td>
      <td>83900.0</td>
      <td>124900.0</td>
      <td>250900.0</td>
      <td>81700.0</td>
      <td>94700.0</td>
      <td>110700.0</td>
      <td>109800.0</td>
      <td>153700.0</td>
      <td>155200.0</td>
      <td>142600.0</td>
      <td>205000.0</td>
      <td>203200.0</td>
      <td>164100.0</td>
      <td>161400.0</td>
      <td>183000.0</td>
      <td>229600.0</td>
      <td>173300.0</td>
      <td>320200.0</td>
      <td>85500.0</td>
      <td>120100.0</td>
      <td>91000.0</td>
      <td>152800.0</td>
      <td>200300.0</td>
      <td>187600.0</td>
      <td>132200.0</td>
      <td>164900.0</td>
      <td>295900.0</td>
      <td>199300.0</td>
      <td>174800.0</td>
      <td>150800.0</td>
      <td>202100.0</td>
      <td>237200.0</td>
      <td>300800.0</td>
      <td>247200.0</td>
      <td>225500.0</td>
      <td>170000.0</td>
      <td>168900.0</td>
      <td>126900.0</td>
      <td>162500.0</td>
      <td>179800.0</td>
      <td>196100.0</td>
      <td>139800.0</td>
      <td>211800.0</td>
      <td>253500.0</td>
      <td>260000.0</td>
      <td>221000.0</td>
      <td>129700.0</td>
      <td>227700.0</td>
      <td>277200.0</td>
      <td>166700.0</td>
      <td>495500.0</td>
      <td>191100.0</td>
      <td>174200.0</td>
      <td>183500.0</td>
      <td>260400.0</td>
      <td>118900.0</td>
      <td>177300.0</td>
      <td>158100.0</td>
      <td>265600.0</td>
      <td>184100.0</td>
      <td>170200.0</td>
      <td>226200.0</td>
      <td>240200.0</td>
      <td>...</td>
      <td>149600.0</td>
      <td>193800.0</td>
      <td>106400.0</td>
      <td>181900.0</td>
      <td>105800.0</td>
      <td>172500.0</td>
      <td>168700.0</td>
      <td>133700.0</td>
      <td>167300.0</td>
      <td>221000.0</td>
      <td>122900.0</td>
      <td>102400.0</td>
      <td>122500.0</td>
      <td>106600.0</td>
      <td>117600.0</td>
      <td>145000.0</td>
      <td>169000.0</td>
      <td>134800.0</td>
      <td>122600.0</td>
      <td>179300.0</td>
      <td>230500.0</td>
      <td>193300.0</td>
      <td>76100.0</td>
      <td>143800.0</td>
      <td>116100.0</td>
      <td>149000.0</td>
      <td>140200.0</td>
      <td>148800.0</td>
      <td>116800.0</td>
      <td>139100.0</td>
      <td>135300.0</td>
      <td>95100.0</td>
      <td>202000.0</td>
      <td>137700.0</td>
      <td>137200.0</td>
      <td>115800.0</td>
      <td>115600.0</td>
      <td>165900.0</td>
      <td>115900.0</td>
      <td>137500.0</td>
      <td>148500.0</td>
      <td>148800.0</td>
      <td>138500.0</td>
      <td>148500.0</td>
      <td>135900.0</td>
      <td>144200.0</td>
      <td>138700.0</td>
      <td>128600.0</td>
      <td>151000.0</td>
      <td>143800.0</td>
      <td>146600.0</td>
      <td>126800.0</td>
      <td>153300.0</td>
      <td>131600.0</td>
      <td>150900.0</td>
      <td>102900.0</td>
      <td>143100.0</td>
      <td>94800.0</td>
      <td>162300.0</td>
      <td>92500.0</td>
      <td>87900.0</td>
      <td>132900.0</td>
      <td>135700.0</td>
      <td>94500.0</td>
      <td>88800.0</td>
      <td>83200.0</td>
      <td>166700.0</td>
      <td>140300.0</td>
      <td>131200.0</td>
      <td>100400.0</td>
      <td>121000.0</td>
      <td>129200.0</td>
      <td>132000.0</td>
      <td>149500.0</td>
      <td>178200.0</td>
      <td>132900.0</td>
      <td>114700.0</td>
      <td>148000.0</td>
      <td>162100.0</td>
      <td>124100.0</td>
      <td>131400.0</td>
      <td>154100.0</td>
      <td>133700.0</td>
      <td>155200.0</td>
      <td>86600.0</td>
      <td>139600.0</td>
      <td>135700.0</td>
      <td>101100.0</td>
      <td>116400.0</td>
      <td>100000.0</td>
      <td>98200.0</td>
      <td>90800.0</td>
      <td>84200.0</td>
      <td>93400.0</td>
      <td>98900.0</td>
      <td>108600.0</td>
      <td>110400.0</td>
      <td>102600.0</td>
      <td>113200.0</td>
      <td>128000.0</td>
      <td>122200.0</td>
      <td>144100.0</td>
      <td>111600.0</td>
      <td>131900.0</td>
      <td>135800.0</td>
      <td>146900.0</td>
      <td>94000.0</td>
      <td>110300.0</td>
      <td>124300.0</td>
      <td>121500.0</td>
      <td>124500.0</td>
      <td>187700.0</td>
      <td>111700.0</td>
      <td>85300.0</td>
      <td>114700.0</td>
      <td>121200.0</td>
      <td>98300.0</td>
      <td>97600.0</td>
      <td>102400.0</td>
      <td>133200.0</td>
      <td>118100.0</td>
      <td>114600.0</td>
      <td>106600.0</td>
      <td>120000.0</td>
      <td>120600.0</td>
      <td>116000.0</td>
      <td>122800.0</td>
      <td>134300.0</td>
      <td>116000.0</td>
      <td>119900.0</td>
      <td>106100.0</td>
      <td>117700.0</td>
      <td>141400.0</td>
      <td>123700.0</td>
      <td>142700.0</td>
      <td>146000.0</td>
      <td>129800.0</td>
      <td>133700.0</td>
      <td>67700.0</td>
      <td>123900.0</td>
      <td>108500.0</td>
      <td>72000.0</td>
      <td>161300.0</td>
      <td>125200.0</td>
      <td>87300.0</td>
      <td>62700.0</td>
      <td>64200.0</td>
      <td>50600.0</td>
      <td>121500.0</td>
      <td>97200.0</td>
      <td>NaN</td>
      <td>56300.0</td>
      <td>97900.0</td>
      <td>115400.0</td>
      <td>83900.0</td>
      <td>78000.0</td>
      <td>94800.0</td>
      <td>130200.0</td>
      <td>126500.0</td>
      <td>132800.0</td>
      <td>122300.0</td>
      <td>77700.0</td>
      <td>70100.0</td>
      <td>106400.0</td>
      <td>91300.0</td>
      <td>127400.0</td>
      <td>80800.0</td>
      <td>125400.0</td>
      <td>76400.0</td>
      <td>83600.0</td>
      <td>69500.0</td>
      <td>71600.0</td>
      <td>98300.0</td>
      <td>93500.0</td>
      <td>130200.0</td>
      <td>114300.0</td>
      <td>148400.0</td>
      <td>101900.0</td>
      <td>103000.0</td>
      <td>118200.0</td>
      <td>158800.0</td>
      <td>134800.0</td>
      <td>120700.0</td>
      <td>113600.0</td>
      <td>114400.0</td>
      <td>119800.0</td>
      <td>131000.0</td>
      <td>46300.0</td>
      <td>NaN</td>
      <td>68600.0</td>
      <td>54200.0</td>
      <td>65500.0</td>
      <td>58400.0</td>
      <td>64400.0</td>
      <td>NaN</td>
      <td>35700.0</td>
      <td>64400.0</td>
      <td>69700.0</td>
      <td>123100.0</td>
      <td>104200.0</td>
      <td>80400.0</td>
      <td>111000.0</td>
      <td>70000.0</td>
      <td>121200.0</td>
      <td>88900.0</td>
      <td>103700.0</td>
      <td>86000.0</td>
      <td>126600.0</td>
      <td>136200.0</td>
      <td>134500.0</td>
      <td>92400.0</td>
      <td>NaN</td>
      <td>104200.0</td>
      <td>85600.0</td>
      <td>90500.0</td>
      <td>112500.0</td>
      <td>134700.0</td>
      <td>90000.0</td>
      <td>NaN</td>
      <td>112300.0</td>
      <td>59700.0</td>
      <td>87300.0</td>
      <td>118000.0</td>
      <td>142500.0</td>
      <td>117400.0</td>
      <td>123000.0</td>
      <td>130500.0</td>
      <td>117900.0</td>
      <td>178500.0</td>
      <td>183900.0</td>
      <td>121700.0</td>
      <td>114200.0</td>
      <td>147200.0</td>
      <td>147000.0</td>
      <td>170600.0</td>
      <td>114100.0</td>
      <td>110700.0</td>
      <td>131300.0</td>
      <td>116300.0</td>
      <td>119600.0</td>
      <td>122700.0</td>
      <td>112200.0</td>
      <td>112500.0</td>
      <td>89400.0</td>
      <td>101900.0</td>
      <td>103000.0</td>
      <td>109200.0</td>
      <td>166800.0</td>
      <td>155500.0</td>
      <td>117100.0</td>
    </tr>
    <tr>
      <th>1996-05-01</th>
      <td>112800.0</td>
      <td>160100.0</td>
      <td>103400.0</td>
      <td>132700.0</td>
      <td>117300.0</td>
      <td>115700.0</td>
      <td>87000.0</td>
      <td>88500.0</td>
      <td>97100.0</td>
      <td>91100.0</td>
      <td>117000.0</td>
      <td>128400.0</td>
      <td>121500.0</td>
      <td>122200.0</td>
      <td>128300.0</td>
      <td>150000.0</td>
      <td>135000.0</td>
      <td>128000.0</td>
      <td>93100.0</td>
      <td>107500.0</td>
      <td>140800.0</td>
      <td>140800.0</td>
      <td>110400.0</td>
      <td>140200.0</td>
      <td>126200.0</td>
      <td>125100.0</td>
      <td>116700.0</td>
      <td>113300.0</td>
      <td>141700.0</td>
      <td>118500.0</td>
      <td>123100.0</td>
      <td>99200.0</td>
      <td>104800.0</td>
      <td>99000.0</td>
      <td>128000.0</td>
      <td>109500.0</td>
      <td>99500.0</td>
      <td>150400.0</td>
      <td>112900.0</td>
      <td>119100.0</td>
      <td>64500.0</td>
      <td>176200.0</td>
      <td>73900.0</td>
      <td>72800.0</td>
      <td>60100.0</td>
      <td>83000.0</td>
      <td>76300.0</td>
      <td>90700.0</td>
      <td>87100.0</td>
      <td>67500.0</td>
      <td>91300.0</td>
      <td>88200.0</td>
      <td>96800.0</td>
      <td>112200.0</td>
      <td>101700.0</td>
      <td>134400.0</td>
      <td>98900.0</td>
      <td>117200.0</td>
      <td>158000.0</td>
      <td>166700.0</td>
      <td>81500.0</td>
      <td>168500.0</td>
      <td>125100.0</td>
      <td>121400.0</td>
      <td>163500.0</td>
      <td>165600.0</td>
      <td>130600.0</td>
      <td>107400.0</td>
      <td>99400.0</td>
      <td>113400.0</td>
      <td>76400.0</td>
      <td>104300.0</td>
      <td>94300.0</td>
      <td>67500.0</td>
      <td>84400.0</td>
      <td>117500.0</td>
      <td>83000.0</td>
      <td>99800.0</td>
      <td>100200.0</td>
      <td>91700.0</td>
      <td>78500.0</td>
      <td>113700.0</td>
      <td>88000.0</td>
      <td>98900.0</td>
      <td>127700.0</td>
      <td>125800.0</td>
      <td>95600.0</td>
      <td>83900.0</td>
      <td>101900.0</td>
      <td>112900.0</td>
      <td>131300.0</td>
      <td>86200.0</td>
      <td>216500.0</td>
      <td>115200.0</td>
      <td>111600.0</td>
      <td>194300.0</td>
      <td>119600.0</td>
      <td>150400.0</td>
      <td>144800.0</td>
      <td>93900.0</td>
      <td>128000.0</td>
      <td>114000.0</td>
      <td>128800.0</td>
      <td>89900.0</td>
      <td>111000.0</td>
      <td>180700.0</td>
      <td>124100.0</td>
      <td>164500.0</td>
      <td>108700.0</td>
      <td>125700.0</td>
      <td>114500.0</td>
      <td>107300.0</td>
      <td>124000.0</td>
      <td>130200.0</td>
      <td>176600.0</td>
      <td>135200.0</td>
      <td>97300.0</td>
      <td>145400.0</td>
      <td>142500.0</td>
      <td>110300.0</td>
      <td>117900.0</td>
      <td>133900.0</td>
      <td>179500.0</td>
      <td>143500.0</td>
      <td>103900.0</td>
      <td>140800.0</td>
      <td>108400.0</td>
      <td>107600.0</td>
      <td>169000.0</td>
      <td>116600.0</td>
      <td>165300.0</td>
      <td>96000.0</td>
      <td>172800.0</td>
      <td>108000.0</td>
      <td>150500.0</td>
      <td>124000.0</td>
      <td>187800.0</td>
      <td>134700.0</td>
      <td>115900.0</td>
      <td>205200.0</td>
      <td>127100.0</td>
      <td>107800.0</td>
      <td>126100.0</td>
      <td>150300.0</td>
      <td>102200.0</td>
      <td>85300.0</td>
      <td>90300.0</td>
      <td>94000.0</td>
      <td>93100.0</td>
      <td>87700.0</td>
      <td>146200.0</td>
      <td>77900.0</td>
      <td>96500.0</td>
      <td>137400.0</td>
      <td>170000.0</td>
      <td>265700.0</td>
      <td>262600.0</td>
      <td>261600.0</td>
      <td>370100.0</td>
      <td>385400.0</td>
      <td>185900.0</td>
      <td>149300.0</td>
      <td>259700.0</td>
      <td>145200.0</td>
      <td>150300.0</td>
      <td>152800.0</td>
      <td>177700.0</td>
      <td>141400.0</td>
      <td>197900.0</td>
      <td>395200.0</td>
      <td>278300.0</td>
      <td>548200.0</td>
      <td>228000.0</td>
      <td>342100.0</td>
      <td>291000.0</td>
      <td>164200.0</td>
      <td>185800.0</td>
      <td>262600.0</td>
      <td>153800.0</td>
      <td>172500.0</td>
      <td>129000.0</td>
      <td>193200.0</td>
      <td>122800.0</td>
      <td>125800.0</td>
      <td>184000.0</td>
      <td>130100.0</td>
      <td>66300.0</td>
      <td>84500.0</td>
      <td>125300.0</td>
      <td>252200.0</td>
      <td>82000.0</td>
      <td>95200.0</td>
      <td>111300.0</td>
      <td>110300.0</td>
      <td>154200.0</td>
      <td>155800.0</td>
      <td>143400.0</td>
      <td>205600.0</td>
      <td>204700.0</td>
      <td>164600.0</td>
      <td>162100.0</td>
      <td>183300.0</td>
      <td>229900.0</td>
      <td>173600.0</td>
      <td>322200.0</td>
      <td>85900.0</td>
      <td>120600.0</td>
      <td>91700.0</td>
      <td>153300.0</td>
      <td>201500.0</td>
      <td>189200.0</td>
      <td>133000.0</td>
      <td>165900.0</td>
      <td>297800.0</td>
      <td>200500.0</td>
      <td>175900.0</td>
      <td>151500.0</td>
      <td>203300.0</td>
      <td>238600.0</td>
      <td>303100.0</td>
      <td>247700.0</td>
      <td>227100.0</td>
      <td>170800.0</td>
      <td>169600.0</td>
      <td>127300.0</td>
      <td>163500.0</td>
      <td>180500.0</td>
      <td>197200.0</td>
      <td>140600.0</td>
      <td>213000.0</td>
      <td>255300.0</td>
      <td>261100.0</td>
      <td>222300.0</td>
      <td>130100.0</td>
      <td>228600.0</td>
      <td>278300.0</td>
      <td>166900.0</td>
      <td>496000.0</td>
      <td>191200.0</td>
      <td>174300.0</td>
      <td>183600.0</td>
      <td>261100.0</td>
      <td>119000.0</td>
      <td>177600.0</td>
      <td>158700.0</td>
      <td>266000.0</td>
      <td>184100.0</td>
      <td>170300.0</td>
      <td>226600.0</td>
      <td>241800.0</td>
      <td>...</td>
      <td>149800.0</td>
      <td>194600.0</td>
      <td>107800.0</td>
      <td>182500.0</td>
      <td>106400.0</td>
      <td>172900.0</td>
      <td>167800.0</td>
      <td>133500.0</td>
      <td>168800.0</td>
      <td>220900.0</td>
      <td>122800.0</td>
      <td>102900.0</td>
      <td>123000.0</td>
      <td>107500.0</td>
      <td>117700.0</td>
      <td>144900.0</td>
      <td>169600.0</td>
      <td>134200.0</td>
      <td>122700.0</td>
      <td>180200.0</td>
      <td>231400.0</td>
      <td>193600.0</td>
      <td>76100.0</td>
      <td>143200.0</td>
      <td>116200.0</td>
      <td>149600.0</td>
      <td>140300.0</td>
      <td>149400.0</td>
      <td>116800.0</td>
      <td>139200.0</td>
      <td>135200.0</td>
      <td>95300.0</td>
      <td>202300.0</td>
      <td>137500.0</td>
      <td>136100.0</td>
      <td>115900.0</td>
      <td>115700.0</td>
      <td>166500.0</td>
      <td>116400.0</td>
      <td>137700.0</td>
      <td>148900.0</td>
      <td>149600.0</td>
      <td>138500.0</td>
      <td>148300.0</td>
      <td>136000.0</td>
      <td>143600.0</td>
      <td>138400.0</td>
      <td>128600.0</td>
      <td>151400.0</td>
      <td>144600.0</td>
      <td>147300.0</td>
      <td>126800.0</td>
      <td>153800.0</td>
      <td>131800.0</td>
      <td>150800.0</td>
      <td>102700.0</td>
      <td>143400.0</td>
      <td>94800.0</td>
      <td>162600.0</td>
      <td>92500.0</td>
      <td>87900.0</td>
      <td>133100.0</td>
      <td>135600.0</td>
      <td>94800.0</td>
      <td>88800.0</td>
      <td>83400.0</td>
      <td>167100.0</td>
      <td>140500.0</td>
      <td>131500.0</td>
      <td>100400.0</td>
      <td>120900.0</td>
      <td>129300.0</td>
      <td>132500.0</td>
      <td>150200.0</td>
      <td>178700.0</td>
      <td>132700.0</td>
      <td>114700.0</td>
      <td>148000.0</td>
      <td>161900.0</td>
      <td>123700.0</td>
      <td>131700.0</td>
      <td>154100.0</td>
      <td>133800.0</td>
      <td>155000.0</td>
      <td>86300.0</td>
      <td>139400.0</td>
      <td>135400.0</td>
      <td>101200.0</td>
      <td>116600.0</td>
      <td>100100.0</td>
      <td>98200.0</td>
      <td>90700.0</td>
      <td>83800.0</td>
      <td>93200.0</td>
      <td>99100.0</td>
      <td>108700.0</td>
      <td>110400.0</td>
      <td>102600.0</td>
      <td>113500.0</td>
      <td>128300.0</td>
      <td>122000.0</td>
      <td>143700.0</td>
      <td>111600.0</td>
      <td>131400.0</td>
      <td>135700.0</td>
      <td>146700.0</td>
      <td>94000.0</td>
      <td>110500.0</td>
      <td>124000.0</td>
      <td>122200.0</td>
      <td>126800.0</td>
      <td>189400.0</td>
      <td>114800.0</td>
      <td>85500.0</td>
      <td>115100.0</td>
      <td>121900.0</td>
      <td>98800.0</td>
      <td>99200.0</td>
      <td>102700.0</td>
      <td>135200.0</td>
      <td>118500.0</td>
      <td>114900.0</td>
      <td>107300.0</td>
      <td>120800.0</td>
      <td>122000.0</td>
      <td>116900.0</td>
      <td>123800.0</td>
      <td>135400.0</td>
      <td>117400.0</td>
      <td>120600.0</td>
      <td>107200.0</td>
      <td>118900.0</td>
      <td>142900.0</td>
      <td>124800.0</td>
      <td>144300.0</td>
      <td>147300.0</td>
      <td>130500.0</td>
      <td>134300.0</td>
      <td>68000.0</td>
      <td>124400.0</td>
      <td>109200.0</td>
      <td>72100.0</td>
      <td>161500.0</td>
      <td>125500.0</td>
      <td>88000.0</td>
      <td>63000.0</td>
      <td>64500.0</td>
      <td>50900.0</td>
      <td>122200.0</td>
      <td>97700.0</td>
      <td>NaN</td>
      <td>56600.0</td>
      <td>98300.0</td>
      <td>115500.0</td>
      <td>84800.0</td>
      <td>79300.0</td>
      <td>95200.0</td>
      <td>130900.0</td>
      <td>127100.0</td>
      <td>133100.0</td>
      <td>123000.0</td>
      <td>78100.0</td>
      <td>70500.0</td>
      <td>105800.0</td>
      <td>91600.0</td>
      <td>127800.0</td>
      <td>81300.0</td>
      <td>125600.0</td>
      <td>77100.0</td>
      <td>84200.0</td>
      <td>70000.0</td>
      <td>72000.0</td>
      <td>98800.0</td>
      <td>93400.0</td>
      <td>129800.0</td>
      <td>114600.0</td>
      <td>147500.0</td>
      <td>101600.0</td>
      <td>102600.0</td>
      <td>118400.0</td>
      <td>158200.0</td>
      <td>134300.0</td>
      <td>120300.0</td>
      <td>113500.0</td>
      <td>114600.0</td>
      <td>119900.0</td>
      <td>130500.0</td>
      <td>46600.0</td>
      <td>NaN</td>
      <td>68700.0</td>
      <td>54100.0</td>
      <td>65500.0</td>
      <td>58600.0</td>
      <td>64700.0</td>
      <td>NaN</td>
      <td>35900.0</td>
      <td>64400.0</td>
      <td>69400.0</td>
      <td>122600.0</td>
      <td>103900.0</td>
      <td>80100.0</td>
      <td>110800.0</td>
      <td>69700.0</td>
      <td>120500.0</td>
      <td>88700.0</td>
      <td>103400.0</td>
      <td>85700.0</td>
      <td>126500.0</td>
      <td>135700.0</td>
      <td>133300.0</td>
      <td>91900.0</td>
      <td>NaN</td>
      <td>104200.0</td>
      <td>86000.0</td>
      <td>90600.0</td>
      <td>112500.0</td>
      <td>134400.0</td>
      <td>90100.0</td>
      <td>NaN</td>
      <td>112700.0</td>
      <td>60000.0</td>
      <td>87400.0</td>
      <td>118200.0</td>
      <td>142700.0</td>
      <td>117700.0</td>
      <td>124200.0</td>
      <td>131100.0</td>
      <td>118200.0</td>
      <td>178100.0</td>
      <td>184200.0</td>
      <td>122100.0</td>
      <td>114700.0</td>
      <td>147900.0</td>
      <td>148400.0</td>
      <td>169300.0</td>
      <td>115000.0</td>
      <td>111100.0</td>
      <td>132400.0</td>
      <td>117100.0</td>
      <td>120700.0</td>
      <td>124200.0</td>
      <td>112400.0</td>
      <td>113100.0</td>
      <td>89800.0</td>
      <td>102900.0</td>
      <td>103900.0</td>
      <td>109600.0</td>
      <td>166200.0</td>
      <td>154900.0</td>
      <td>117200.0</td>
    </tr>
    <tr>
      <th>1996-06-01</th>
      <td>112600.0</td>
      <td>159300.0</td>
      <td>103600.0</td>
      <td>132000.0</td>
      <td>117100.0</td>
      <td>115500.0</td>
      <td>87100.0</td>
      <td>88400.0</td>
      <td>97000.0</td>
      <td>91000.0</td>
      <td>116400.0</td>
      <td>127800.0</td>
      <td>121200.0</td>
      <td>122000.0</td>
      <td>128100.0</td>
      <td>150000.0</td>
      <td>134700.0</td>
      <td>128300.0</td>
      <td>92700.0</td>
      <td>106800.0</td>
      <td>140600.0</td>
      <td>140300.0</td>
      <td>109700.0</td>
      <td>139200.0</td>
      <td>125300.0</td>
      <td>125700.0</td>
      <td>116400.0</td>
      <td>113300.0</td>
      <td>141100.0</td>
      <td>118100.0</td>
      <td>123400.0</td>
      <td>98500.0</td>
      <td>104100.0</td>
      <td>99000.0</td>
      <td>127800.0</td>
      <td>108900.0</td>
      <td>99300.0</td>
      <td>149800.0</td>
      <td>112300.0</td>
      <td>118300.0</td>
      <td>64000.0</td>
      <td>175400.0</td>
      <td>73700.0</td>
      <td>72400.0</td>
      <td>59900.0</td>
      <td>82500.0</td>
      <td>76100.0</td>
      <td>90600.0</td>
      <td>86500.0</td>
      <td>67500.0</td>
      <td>90900.0</td>
      <td>87800.0</td>
      <td>96300.0</td>
      <td>110900.0</td>
      <td>101100.0</td>
      <td>135000.0</td>
      <td>98300.0</td>
      <td>117200.0</td>
      <td>157100.0</td>
      <td>166600.0</td>
      <td>81100.0</td>
      <td>168200.0</td>
      <td>125500.0</td>
      <td>121800.0</td>
      <td>163100.0</td>
      <td>165400.0</td>
      <td>131100.0</td>
      <td>106200.0</td>
      <td>98800.0</td>
      <td>113000.0</td>
      <td>76500.0</td>
      <td>103900.0</td>
      <td>94000.0</td>
      <td>67700.0</td>
      <td>84200.0</td>
      <td>117300.0</td>
      <td>82900.0</td>
      <td>99400.0</td>
      <td>99900.0</td>
      <td>91600.0</td>
      <td>78200.0</td>
      <td>113800.0</td>
      <td>88000.0</td>
      <td>98600.0</td>
      <td>127400.0</td>
      <td>125600.0</td>
      <td>95400.0</td>
      <td>84300.0</td>
      <td>102100.0</td>
      <td>113300.0</td>
      <td>132200.0</td>
      <td>86400.0</td>
      <td>216500.0</td>
      <td>115900.0</td>
      <td>112200.0</td>
      <td>194900.0</td>
      <td>120300.0</td>
      <td>150800.0</td>
      <td>145300.0</td>
      <td>94200.0</td>
      <td>128200.0</td>
      <td>114600.0</td>
      <td>129200.0</td>
      <td>90000.0</td>
      <td>111500.0</td>
      <td>181400.0</td>
      <td>124600.0</td>
      <td>165000.0</td>
      <td>109400.0</td>
      <td>126200.0</td>
      <td>114700.0</td>
      <td>107800.0</td>
      <td>124900.0</td>
      <td>131200.0</td>
      <td>177400.0</td>
      <td>135900.0</td>
      <td>96800.0</td>
      <td>145600.0</td>
      <td>143100.0</td>
      <td>110600.0</td>
      <td>118200.0</td>
      <td>134600.0</td>
      <td>180000.0</td>
      <td>143800.0</td>
      <td>104200.0</td>
      <td>141400.0</td>
      <td>108600.0</td>
      <td>107800.0</td>
      <td>168800.0</td>
      <td>117000.0</td>
      <td>166100.0</td>
      <td>96000.0</td>
      <td>173900.0</td>
      <td>108500.0</td>
      <td>150800.0</td>
      <td>124200.0</td>
      <td>188700.0</td>
      <td>134700.0</td>
      <td>116400.0</td>
      <td>205900.0</td>
      <td>127500.0</td>
      <td>108400.0</td>
      <td>126600.0</td>
      <td>150700.0</td>
      <td>102900.0</td>
      <td>85600.0</td>
      <td>90600.0</td>
      <td>94200.0</td>
      <td>93600.0</td>
      <td>87700.0</td>
      <td>146700.0</td>
      <td>78500.0</td>
      <td>96600.0</td>
      <td>137900.0</td>
      <td>170500.0</td>
      <td>265800.0</td>
      <td>263200.0</td>
      <td>263700.0</td>
      <td>372700.0</td>
      <td>386700.0</td>
      <td>186400.0</td>
      <td>149600.0</td>
      <td>260700.0</td>
      <td>145400.0</td>
      <td>150500.0</td>
      <td>153500.0</td>
      <td>178500.0</td>
      <td>142300.0</td>
      <td>198400.0</td>
      <td>394800.0</td>
      <td>279500.0</td>
      <td>550700.0</td>
      <td>228600.0</td>
      <td>342100.0</td>
      <td>292600.0</td>
      <td>164500.0</td>
      <td>186100.0</td>
      <td>264700.0</td>
      <td>154300.0</td>
      <td>173000.0</td>
      <td>129200.0</td>
      <td>194700.0</td>
      <td>123800.0</td>
      <td>126400.0</td>
      <td>185100.0</td>
      <td>130800.0</td>
      <td>66500.0</td>
      <td>85200.0</td>
      <td>125800.0</td>
      <td>253700.0</td>
      <td>82200.0</td>
      <td>95700.0</td>
      <td>111900.0</td>
      <td>110700.0</td>
      <td>154600.0</td>
      <td>156500.0</td>
      <td>144100.0</td>
      <td>206100.0</td>
      <td>206200.0</td>
      <td>165100.0</td>
      <td>162800.0</td>
      <td>183800.0</td>
      <td>230200.0</td>
      <td>173900.0</td>
      <td>324500.0</td>
      <td>86400.0</td>
      <td>121100.0</td>
      <td>92500.0</td>
      <td>153800.0</td>
      <td>202800.0</td>
      <td>190800.0</td>
      <td>133700.0</td>
      <td>166900.0</td>
      <td>299600.0</td>
      <td>201700.0</td>
      <td>177000.0</td>
      <td>152200.0</td>
      <td>204500.0</td>
      <td>239900.0</td>
      <td>305300.0</td>
      <td>248300.0</td>
      <td>228600.0</td>
      <td>171600.0</td>
      <td>170400.0</td>
      <td>127800.0</td>
      <td>164700.0</td>
      <td>181200.0</td>
      <td>198300.0</td>
      <td>141500.0</td>
      <td>214100.0</td>
      <td>257100.0</td>
      <td>262300.0</td>
      <td>223700.0</td>
      <td>130300.0</td>
      <td>229400.0</td>
      <td>279300.0</td>
      <td>167000.0</td>
      <td>496500.0</td>
      <td>191400.0</td>
      <td>174400.0</td>
      <td>183600.0</td>
      <td>261700.0</td>
      <td>119000.0</td>
      <td>177900.0</td>
      <td>159300.0</td>
      <td>266400.0</td>
      <td>184100.0</td>
      <td>170500.0</td>
      <td>227100.0</td>
      <td>243200.0</td>
      <td>...</td>
      <td>150000.0</td>
      <td>195500.0</td>
      <td>109200.0</td>
      <td>183000.0</td>
      <td>107000.0</td>
      <td>173300.0</td>
      <td>167000.0</td>
      <td>133300.0</td>
      <td>170300.0</td>
      <td>220800.0</td>
      <td>122700.0</td>
      <td>103300.0</td>
      <td>123500.0</td>
      <td>108200.0</td>
      <td>117800.0</td>
      <td>144900.0</td>
      <td>170200.0</td>
      <td>133400.0</td>
      <td>122700.0</td>
      <td>180800.0</td>
      <td>232200.0</td>
      <td>193700.0</td>
      <td>75900.0</td>
      <td>142700.0</td>
      <td>116400.0</td>
      <td>150200.0</td>
      <td>140500.0</td>
      <td>149900.0</td>
      <td>116700.0</td>
      <td>139100.0</td>
      <td>135200.0</td>
      <td>95500.0</td>
      <td>202600.0</td>
      <td>137200.0</td>
      <td>135000.0</td>
      <td>115900.0</td>
      <td>115800.0</td>
      <td>166900.0</td>
      <td>116900.0</td>
      <td>137800.0</td>
      <td>149300.0</td>
      <td>150400.0</td>
      <td>138800.0</td>
      <td>148200.0</td>
      <td>136000.0</td>
      <td>143100.0</td>
      <td>138200.0</td>
      <td>128600.0</td>
      <td>151700.0</td>
      <td>145500.0</td>
      <td>147900.0</td>
      <td>126800.0</td>
      <td>154200.0</td>
      <td>131900.0</td>
      <td>150800.0</td>
      <td>102700.0</td>
      <td>143600.0</td>
      <td>94800.0</td>
      <td>162900.0</td>
      <td>92500.0</td>
      <td>87900.0</td>
      <td>133200.0</td>
      <td>135600.0</td>
      <td>95100.0</td>
      <td>88700.0</td>
      <td>83700.0</td>
      <td>167500.0</td>
      <td>140800.0</td>
      <td>131700.0</td>
      <td>100500.0</td>
      <td>120900.0</td>
      <td>129400.0</td>
      <td>133100.0</td>
      <td>151000.0</td>
      <td>179300.0</td>
      <td>132600.0</td>
      <td>114700.0</td>
      <td>148000.0</td>
      <td>161500.0</td>
      <td>123400.0</td>
      <td>132100.0</td>
      <td>154000.0</td>
      <td>133800.0</td>
      <td>154700.0</td>
      <td>86100.0</td>
      <td>139100.0</td>
      <td>135100.0</td>
      <td>101400.0</td>
      <td>116900.0</td>
      <td>100300.0</td>
      <td>98200.0</td>
      <td>90600.0</td>
      <td>83400.0</td>
      <td>93000.0</td>
      <td>99300.0</td>
      <td>108800.0</td>
      <td>110500.0</td>
      <td>102700.0</td>
      <td>113900.0</td>
      <td>128400.0</td>
      <td>121800.0</td>
      <td>143200.0</td>
      <td>111600.0</td>
      <td>131100.0</td>
      <td>135700.0</td>
      <td>146600.0</td>
      <td>93900.0</td>
      <td>110700.0</td>
      <td>123700.0</td>
      <td>122900.0</td>
      <td>129200.0</td>
      <td>191000.0</td>
      <td>117700.0</td>
      <td>85700.0</td>
      <td>115400.0</td>
      <td>122500.0</td>
      <td>99200.0</td>
      <td>100800.0</td>
      <td>102900.0</td>
      <td>137100.0</td>
      <td>118700.0</td>
      <td>115100.0</td>
      <td>108000.0</td>
      <td>121700.0</td>
      <td>123700.0</td>
      <td>117900.0</td>
      <td>125000.0</td>
      <td>136700.0</td>
      <td>118700.0</td>
      <td>121300.0</td>
      <td>108300.0</td>
      <td>120400.0</td>
      <td>144700.0</td>
      <td>126100.0</td>
      <td>146100.0</td>
      <td>148800.0</td>
      <td>131100.0</td>
      <td>134900.0</td>
      <td>68200.0</td>
      <td>125000.0</td>
      <td>109700.0</td>
      <td>72100.0</td>
      <td>161800.0</td>
      <td>125600.0</td>
      <td>88500.0</td>
      <td>63300.0</td>
      <td>64800.0</td>
      <td>51100.0</td>
      <td>122800.0</td>
      <td>98100.0</td>
      <td>NaN</td>
      <td>56800.0</td>
      <td>98700.0</td>
      <td>115500.0</td>
      <td>85700.0</td>
      <td>80600.0</td>
      <td>95500.0</td>
      <td>131700.0</td>
      <td>127800.0</td>
      <td>133500.0</td>
      <td>123800.0</td>
      <td>78400.0</td>
      <td>70900.0</td>
      <td>105100.0</td>
      <td>91700.0</td>
      <td>128200.0</td>
      <td>81800.0</td>
      <td>125700.0</td>
      <td>77700.0</td>
      <td>84800.0</td>
      <td>70500.0</td>
      <td>72300.0</td>
      <td>99200.0</td>
      <td>93200.0</td>
      <td>129200.0</td>
      <td>114700.0</td>
      <td>146500.0</td>
      <td>101300.0</td>
      <td>102200.0</td>
      <td>118400.0</td>
      <td>157500.0</td>
      <td>133600.0</td>
      <td>119800.0</td>
      <td>113400.0</td>
      <td>114800.0</td>
      <td>119800.0</td>
      <td>130000.0</td>
      <td>46900.0</td>
      <td>NaN</td>
      <td>68800.0</td>
      <td>53900.0</td>
      <td>65500.0</td>
      <td>58900.0</td>
      <td>65000.0</td>
      <td>NaN</td>
      <td>36000.0</td>
      <td>64400.0</td>
      <td>69200.0</td>
      <td>122000.0</td>
      <td>103700.0</td>
      <td>79800.0</td>
      <td>110500.0</td>
      <td>69500.0</td>
      <td>119700.0</td>
      <td>88500.0</td>
      <td>103100.0</td>
      <td>85400.0</td>
      <td>126300.0</td>
      <td>135200.0</td>
      <td>132000.0</td>
      <td>91400.0</td>
      <td>NaN</td>
      <td>104200.0</td>
      <td>86400.0</td>
      <td>90700.0</td>
      <td>112400.0</td>
      <td>134100.0</td>
      <td>90200.0</td>
      <td>NaN</td>
      <td>113100.0</td>
      <td>60400.0</td>
      <td>87500.0</td>
      <td>118500.0</td>
      <td>143000.0</td>
      <td>118000.0</td>
      <td>125300.0</td>
      <td>131900.0</td>
      <td>118500.0</td>
      <td>177700.0</td>
      <td>184300.0</td>
      <td>122500.0</td>
      <td>115100.0</td>
      <td>148500.0</td>
      <td>149700.0</td>
      <td>167800.0</td>
      <td>116000.0</td>
      <td>111400.0</td>
      <td>133600.0</td>
      <td>117800.0</td>
      <td>121800.0</td>
      <td>125700.0</td>
      <td>112600.0</td>
      <td>113800.0</td>
      <td>90200.0</td>
      <td>103800.0</td>
      <td>104700.0</td>
      <td>110000.0</td>
      <td>165400.0</td>
      <td>154200.0</td>
      <td>117300.0</td>
    </tr>
    <tr>
      <th>1996-07-01</th>
      <td>112300.0</td>
      <td>158600.0</td>
      <td>103800.0</td>
      <td>131400.0</td>
      <td>117000.0</td>
      <td>115300.0</td>
      <td>87200.0</td>
      <td>88300.0</td>
      <td>96800.0</td>
      <td>90800.0</td>
      <td>115700.0</td>
      <td>127200.0</td>
      <td>121000.0</td>
      <td>121900.0</td>
      <td>128000.0</td>
      <td>150100.0</td>
      <td>134300.0</td>
      <td>128700.0</td>
      <td>92200.0</td>
      <td>106200.0</td>
      <td>140400.0</td>
      <td>139800.0</td>
      <td>108900.0</td>
      <td>138200.0</td>
      <td>124400.0</td>
      <td>126300.0</td>
      <td>116000.0</td>
      <td>113400.0</td>
      <td>140600.0</td>
      <td>117700.0</td>
      <td>123600.0</td>
      <td>97900.0</td>
      <td>103500.0</td>
      <td>98900.0</td>
      <td>127500.0</td>
      <td>108400.0</td>
      <td>99000.0</td>
      <td>149100.0</td>
      <td>111800.0</td>
      <td>117400.0</td>
      <td>63500.0</td>
      <td>174700.0</td>
      <td>73500.0</td>
      <td>71900.0</td>
      <td>59700.0</td>
      <td>82100.0</td>
      <td>75900.0</td>
      <td>90500.0</td>
      <td>86100.0</td>
      <td>67400.0</td>
      <td>90700.0</td>
      <td>87300.0</td>
      <td>95900.0</td>
      <td>109700.0</td>
      <td>100500.0</td>
      <td>135600.0</td>
      <td>97700.0</td>
      <td>117100.0</td>
      <td>156100.0</td>
      <td>166200.0</td>
      <td>80700.0</td>
      <td>167900.0</td>
      <td>126000.0</td>
      <td>122300.0</td>
      <td>162800.0</td>
      <td>165200.0</td>
      <td>131600.0</td>
      <td>105000.0</td>
      <td>98200.0</td>
      <td>112600.0</td>
      <td>76600.0</td>
      <td>103500.0</td>
      <td>93700.0</td>
      <td>67900.0</td>
      <td>84100.0</td>
      <td>117100.0</td>
      <td>82900.0</td>
      <td>99000.0</td>
      <td>99700.0</td>
      <td>91500.0</td>
      <td>77900.0</td>
      <td>113800.0</td>
      <td>88000.0</td>
      <td>98400.0</td>
      <td>127200.0</td>
      <td>125500.0</td>
      <td>95100.0</td>
      <td>84600.0</td>
      <td>102200.0</td>
      <td>113700.0</td>
      <td>133100.0</td>
      <td>86700.0</td>
      <td>216600.0</td>
      <td>116600.0</td>
      <td>112700.0</td>
      <td>195500.0</td>
      <td>121000.0</td>
      <td>151200.0</td>
      <td>145800.0</td>
      <td>94500.0</td>
      <td>128300.0</td>
      <td>115200.0</td>
      <td>129700.0</td>
      <td>90100.0</td>
      <td>112000.0</td>
      <td>182100.0</td>
      <td>125100.0</td>
      <td>165500.0</td>
      <td>110000.0</td>
      <td>126600.0</td>
      <td>114800.0</td>
      <td>108200.0</td>
      <td>125800.0</td>
      <td>132200.0</td>
      <td>178300.0</td>
      <td>136600.0</td>
      <td>96200.0</td>
      <td>145900.0</td>
      <td>143700.0</td>
      <td>110900.0</td>
      <td>118400.0</td>
      <td>135100.0</td>
      <td>180600.0</td>
      <td>144100.0</td>
      <td>104500.0</td>
      <td>141900.0</td>
      <td>108700.0</td>
      <td>108000.0</td>
      <td>168500.0</td>
      <td>117400.0</td>
      <td>166700.0</td>
      <td>96000.0</td>
      <td>175100.0</td>
      <td>108900.0</td>
      <td>151100.0</td>
      <td>124300.0</td>
      <td>189500.0</td>
      <td>134800.0</td>
      <td>116800.0</td>
      <td>206500.0</td>
      <td>127900.0</td>
      <td>108900.0</td>
      <td>127100.0</td>
      <td>151000.0</td>
      <td>103600.0</td>
      <td>85800.0</td>
      <td>91000.0</td>
      <td>94300.0</td>
      <td>94000.0</td>
      <td>87800.0</td>
      <td>147200.0</td>
      <td>79000.0</td>
      <td>96700.0</td>
      <td>138500.0</td>
      <td>171000.0</td>
      <td>266000.0</td>
      <td>263900.0</td>
      <td>265900.0</td>
      <td>375500.0</td>
      <td>388100.0</td>
      <td>186800.0</td>
      <td>149900.0</td>
      <td>261700.0</td>
      <td>145600.0</td>
      <td>150600.0</td>
      <td>154200.0</td>
      <td>179300.0</td>
      <td>143300.0</td>
      <td>198900.0</td>
      <td>394400.0</td>
      <td>280700.0</td>
      <td>553200.0</td>
      <td>229200.0</td>
      <td>342100.0</td>
      <td>294300.0</td>
      <td>164800.0</td>
      <td>186400.0</td>
      <td>267000.0</td>
      <td>154700.0</td>
      <td>173500.0</td>
      <td>129300.0</td>
      <td>196200.0</td>
      <td>124800.0</td>
      <td>127000.0</td>
      <td>186400.0</td>
      <td>131500.0</td>
      <td>66800.0</td>
      <td>85800.0</td>
      <td>126300.0</td>
      <td>255200.0</td>
      <td>82400.0</td>
      <td>96200.0</td>
      <td>112500.0</td>
      <td>111100.0</td>
      <td>155000.0</td>
      <td>157100.0</td>
      <td>144900.0</td>
      <td>206400.0</td>
      <td>207900.0</td>
      <td>165700.0</td>
      <td>163400.0</td>
      <td>184300.0</td>
      <td>230500.0</td>
      <td>174200.0</td>
      <td>327100.0</td>
      <td>86800.0</td>
      <td>121600.0</td>
      <td>93200.0</td>
      <td>154100.0</td>
      <td>204000.0</td>
      <td>192400.0</td>
      <td>134400.0</td>
      <td>168100.0</td>
      <td>301500.0</td>
      <td>203000.0</td>
      <td>178000.0</td>
      <td>152900.0</td>
      <td>205700.0</td>
      <td>241100.0</td>
      <td>307600.0</td>
      <td>248900.0</td>
      <td>230000.0</td>
      <td>172400.0</td>
      <td>171200.0</td>
      <td>128200.0</td>
      <td>165800.0</td>
      <td>181800.0</td>
      <td>199400.0</td>
      <td>142300.0</td>
      <td>215300.0</td>
      <td>258800.0</td>
      <td>263400.0</td>
      <td>225100.0</td>
      <td>130500.0</td>
      <td>230300.0</td>
      <td>280400.0</td>
      <td>167100.0</td>
      <td>496800.0</td>
      <td>191500.0</td>
      <td>174500.0</td>
      <td>183700.0</td>
      <td>262100.0</td>
      <td>119100.0</td>
      <td>178200.0</td>
      <td>160100.0</td>
      <td>266700.0</td>
      <td>184100.0</td>
      <td>170600.0</td>
      <td>227600.0</td>
      <td>244500.0</td>
      <td>...</td>
      <td>150300.0</td>
      <td>196200.0</td>
      <td>110600.0</td>
      <td>183500.0</td>
      <td>107600.0</td>
      <td>173700.0</td>
      <td>166200.0</td>
      <td>133100.0</td>
      <td>171900.0</td>
      <td>220900.0</td>
      <td>122600.0</td>
      <td>103700.0</td>
      <td>123900.0</td>
      <td>108900.0</td>
      <td>117900.0</td>
      <td>144900.0</td>
      <td>170700.0</td>
      <td>132600.0</td>
      <td>122700.0</td>
      <td>181200.0</td>
      <td>232900.0</td>
      <td>193400.0</td>
      <td>75800.0</td>
      <td>142200.0</td>
      <td>116400.0</td>
      <td>150700.0</td>
      <td>140600.0</td>
      <td>150400.0</td>
      <td>116600.0</td>
      <td>139100.0</td>
      <td>135100.0</td>
      <td>95900.0</td>
      <td>202700.0</td>
      <td>136900.0</td>
      <td>133900.0</td>
      <td>116000.0</td>
      <td>116000.0</td>
      <td>167300.0</td>
      <td>117200.0</td>
      <td>137800.0</td>
      <td>149700.0</td>
      <td>151200.0</td>
      <td>139200.0</td>
      <td>148300.0</td>
      <td>136200.0</td>
      <td>142700.0</td>
      <td>138200.0</td>
      <td>128500.0</td>
      <td>151800.0</td>
      <td>146400.0</td>
      <td>148500.0</td>
      <td>126900.0</td>
      <td>154600.0</td>
      <td>132000.0</td>
      <td>150800.0</td>
      <td>102600.0</td>
      <td>143600.0</td>
      <td>94800.0</td>
      <td>163300.0</td>
      <td>92600.0</td>
      <td>87900.0</td>
      <td>133400.0</td>
      <td>135600.0</td>
      <td>95400.0</td>
      <td>88600.0</td>
      <td>83900.0</td>
      <td>167900.0</td>
      <td>141100.0</td>
      <td>131900.0</td>
      <td>100600.0</td>
      <td>120700.0</td>
      <td>129400.0</td>
      <td>133600.0</td>
      <td>151900.0</td>
      <td>179800.0</td>
      <td>132500.0</td>
      <td>114700.0</td>
      <td>148100.0</td>
      <td>161100.0</td>
      <td>123100.0</td>
      <td>132400.0</td>
      <td>153800.0</td>
      <td>133800.0</td>
      <td>154300.0</td>
      <td>85900.0</td>
      <td>138900.0</td>
      <td>134900.0</td>
      <td>101600.0</td>
      <td>117100.0</td>
      <td>100500.0</td>
      <td>98200.0</td>
      <td>90500.0</td>
      <td>83100.0</td>
      <td>92900.0</td>
      <td>99400.0</td>
      <td>108900.0</td>
      <td>110500.0</td>
      <td>102700.0</td>
      <td>114300.0</td>
      <td>128600.0</td>
      <td>121600.0</td>
      <td>142500.0</td>
      <td>111500.0</td>
      <td>130800.0</td>
      <td>135600.0</td>
      <td>146700.0</td>
      <td>93900.0</td>
      <td>111000.0</td>
      <td>123500.0</td>
      <td>123400.0</td>
      <td>131500.0</td>
      <td>192600.0</td>
      <td>120300.0</td>
      <td>85800.0</td>
      <td>115700.0</td>
      <td>122900.0</td>
      <td>99600.0</td>
      <td>102300.0</td>
      <td>103100.0</td>
      <td>139100.0</td>
      <td>118900.0</td>
      <td>115300.0</td>
      <td>108700.0</td>
      <td>122800.0</td>
      <td>125600.0</td>
      <td>119000.0</td>
      <td>126300.0</td>
      <td>138100.0</td>
      <td>120100.0</td>
      <td>121800.0</td>
      <td>109200.0</td>
      <td>122200.0</td>
      <td>146800.0</td>
      <td>127600.0</td>
      <td>147900.0</td>
      <td>150300.0</td>
      <td>131700.0</td>
      <td>135400.0</td>
      <td>68400.0</td>
      <td>125500.0</td>
      <td>110200.0</td>
      <td>72200.0</td>
      <td>162000.0</td>
      <td>125700.0</td>
      <td>88900.0</td>
      <td>63600.0</td>
      <td>65000.0</td>
      <td>51300.0</td>
      <td>123400.0</td>
      <td>98500.0</td>
      <td>NaN</td>
      <td>57100.0</td>
      <td>99100.0</td>
      <td>115600.0</td>
      <td>86500.0</td>
      <td>81900.0</td>
      <td>95800.0</td>
      <td>132600.0</td>
      <td>128400.0</td>
      <td>133900.0</td>
      <td>124400.0</td>
      <td>78800.0</td>
      <td>71200.0</td>
      <td>104300.0</td>
      <td>91900.0</td>
      <td>128600.0</td>
      <td>82300.0</td>
      <td>125800.0</td>
      <td>78400.0</td>
      <td>85400.0</td>
      <td>71000.0</td>
      <td>72700.0</td>
      <td>99500.0</td>
      <td>93000.0</td>
      <td>128700.0</td>
      <td>114700.0</td>
      <td>145400.0</td>
      <td>101000.0</td>
      <td>101900.0</td>
      <td>118100.0</td>
      <td>156500.0</td>
      <td>132900.0</td>
      <td>119100.0</td>
      <td>113200.0</td>
      <td>115000.0</td>
      <td>119700.0</td>
      <td>129300.0</td>
      <td>47200.0</td>
      <td>NaN</td>
      <td>68900.0</td>
      <td>53800.0</td>
      <td>65500.0</td>
      <td>59100.0</td>
      <td>65300.0</td>
      <td>NaN</td>
      <td>36200.0</td>
      <td>64300.0</td>
      <td>68900.0</td>
      <td>121400.0</td>
      <td>103300.0</td>
      <td>79600.0</td>
      <td>110100.0</td>
      <td>69200.0</td>
      <td>119100.0</td>
      <td>88200.0</td>
      <td>102700.0</td>
      <td>85200.0</td>
      <td>126100.0</td>
      <td>134700.0</td>
      <td>130500.0</td>
      <td>91000.0</td>
      <td>NaN</td>
      <td>104300.0</td>
      <td>86800.0</td>
      <td>90700.0</td>
      <td>112100.0</td>
      <td>133800.0</td>
      <td>90400.0</td>
      <td>NaN</td>
      <td>113400.0</td>
      <td>60800.0</td>
      <td>87500.0</td>
      <td>118700.0</td>
      <td>143300.0</td>
      <td>118300.0</td>
      <td>126400.0</td>
      <td>132700.0</td>
      <td>118900.0</td>
      <td>177000.0</td>
      <td>184500.0</td>
      <td>122900.0</td>
      <td>115500.0</td>
      <td>149000.0</td>
      <td>150700.0</td>
      <td>166300.0</td>
      <td>116900.0</td>
      <td>111800.0</td>
      <td>134800.0</td>
      <td>118500.0</td>
      <td>122800.0</td>
      <td>127200.0</td>
      <td>112800.0</td>
      <td>114400.0</td>
      <td>90600.0</td>
      <td>104700.0</td>
      <td>105500.0</td>
      <td>110400.0</td>
      <td>164400.0</td>
      <td>153500.0</td>
      <td>117400.0</td>
    </tr>
    <tr>
      <th>1996-08-01</th>
      <td>112100.0</td>
      <td>158000.0</td>
      <td>103900.0</td>
      <td>130800.0</td>
      <td>116800.0</td>
      <td>115100.0</td>
      <td>87400.0</td>
      <td>88200.0</td>
      <td>96600.0</td>
      <td>90700.0</td>
      <td>115100.0</td>
      <td>126700.0</td>
      <td>120800.0</td>
      <td>121700.0</td>
      <td>128000.0</td>
      <td>150200.0</td>
      <td>133800.0</td>
      <td>129100.0</td>
      <td>91700.0</td>
      <td>105600.0</td>
      <td>140100.0</td>
      <td>139300.0</td>
      <td>108200.0</td>
      <td>137200.0</td>
      <td>123500.0</td>
      <td>126800.0</td>
      <td>115700.0</td>
      <td>113500.0</td>
      <td>140100.0</td>
      <td>117400.0</td>
      <td>123900.0</td>
      <td>97200.0</td>
      <td>103000.0</td>
      <td>98800.0</td>
      <td>127300.0</td>
      <td>108000.0</td>
      <td>98800.0</td>
      <td>148500.0</td>
      <td>111200.0</td>
      <td>116600.0</td>
      <td>63000.0</td>
      <td>174000.0</td>
      <td>73400.0</td>
      <td>71500.0</td>
      <td>59500.0</td>
      <td>81600.0</td>
      <td>75700.0</td>
      <td>90300.0</td>
      <td>85600.0</td>
      <td>67400.0</td>
      <td>90500.0</td>
      <td>86900.0</td>
      <td>95400.0</td>
      <td>108400.0</td>
      <td>100000.0</td>
      <td>136100.0</td>
      <td>97000.0</td>
      <td>117100.0</td>
      <td>155200.0</td>
      <td>165400.0</td>
      <td>80200.0</td>
      <td>167600.0</td>
      <td>126400.0</td>
      <td>122700.0</td>
      <td>162400.0</td>
      <td>165000.0</td>
      <td>132100.0</td>
      <td>103700.0</td>
      <td>97700.0</td>
      <td>112300.0</td>
      <td>76600.0</td>
      <td>103000.0</td>
      <td>93400.0</td>
      <td>68100.0</td>
      <td>83900.0</td>
      <td>116800.0</td>
      <td>82800.0</td>
      <td>98600.0</td>
      <td>99400.0</td>
      <td>91400.0</td>
      <td>77600.0</td>
      <td>113900.0</td>
      <td>87900.0</td>
      <td>98100.0</td>
      <td>127000.0</td>
      <td>125400.0</td>
      <td>94900.0</td>
      <td>84900.0</td>
      <td>102300.0</td>
      <td>114100.0</td>
      <td>133900.0</td>
      <td>86900.0</td>
      <td>216700.0</td>
      <td>117200.0</td>
      <td>113100.0</td>
      <td>196300.0</td>
      <td>121600.0</td>
      <td>151600.0</td>
      <td>146300.0</td>
      <td>94700.0</td>
      <td>128500.0</td>
      <td>115700.0</td>
      <td>130100.0</td>
      <td>90200.0</td>
      <td>112400.0</td>
      <td>182800.0</td>
      <td>125500.0</td>
      <td>165900.0</td>
      <td>110600.0</td>
      <td>127000.0</td>
      <td>115000.0</td>
      <td>108600.0</td>
      <td>126700.0</td>
      <td>133100.0</td>
      <td>179100.0</td>
      <td>137200.0</td>
      <td>95500.0</td>
      <td>146100.0</td>
      <td>144200.0</td>
      <td>111200.0</td>
      <td>118700.0</td>
      <td>135600.0</td>
      <td>181100.0</td>
      <td>144400.0</td>
      <td>104800.0</td>
      <td>142400.0</td>
      <td>108800.0</td>
      <td>108200.0</td>
      <td>168200.0</td>
      <td>117800.0</td>
      <td>167300.0</td>
      <td>96000.0</td>
      <td>176300.0</td>
      <td>109400.0</td>
      <td>151400.0</td>
      <td>124500.0</td>
      <td>190300.0</td>
      <td>134900.0</td>
      <td>117200.0</td>
      <td>207000.0</td>
      <td>128400.0</td>
      <td>109300.0</td>
      <td>127600.0</td>
      <td>151300.0</td>
      <td>104200.0</td>
      <td>86000.0</td>
      <td>91300.0</td>
      <td>94500.0</td>
      <td>94400.0</td>
      <td>87800.0</td>
      <td>147600.0</td>
      <td>79400.0</td>
      <td>96700.0</td>
      <td>138900.0</td>
      <td>171500.0</td>
      <td>266200.0</td>
      <td>264500.0</td>
      <td>267900.0</td>
      <td>378100.0</td>
      <td>389400.0</td>
      <td>187300.0</td>
      <td>150200.0</td>
      <td>262700.0</td>
      <td>145900.0</td>
      <td>150800.0</td>
      <td>154900.0</td>
      <td>180000.0</td>
      <td>144200.0</td>
      <td>199500.0</td>
      <td>394100.0</td>
      <td>281800.0</td>
      <td>555800.0</td>
      <td>229800.0</td>
      <td>342200.0</td>
      <td>296000.0</td>
      <td>165000.0</td>
      <td>186800.0</td>
      <td>269300.0</td>
      <td>155200.0</td>
      <td>174000.0</td>
      <td>129500.0</td>
      <td>197700.0</td>
      <td>125700.0</td>
      <td>127600.0</td>
      <td>187600.0</td>
      <td>132200.0</td>
      <td>67000.0</td>
      <td>86400.0</td>
      <td>126900.0</td>
      <td>256700.0</td>
      <td>82600.0</td>
      <td>96800.0</td>
      <td>113000.0</td>
      <td>111600.0</td>
      <td>155500.0</td>
      <td>157800.0</td>
      <td>145700.0</td>
      <td>206700.0</td>
      <td>209600.0</td>
      <td>166200.0</td>
      <td>164100.0</td>
      <td>184700.0</td>
      <td>230800.0</td>
      <td>174500.0</td>
      <td>329700.0</td>
      <td>87300.0</td>
      <td>122000.0</td>
      <td>93900.0</td>
      <td>154400.0</td>
      <td>205100.0</td>
      <td>193900.0</td>
      <td>135200.0</td>
      <td>169200.0</td>
      <td>303300.0</td>
      <td>204200.0</td>
      <td>178900.0</td>
      <td>153600.0</td>
      <td>206800.0</td>
      <td>242400.0</td>
      <td>309800.0</td>
      <td>249600.0</td>
      <td>231400.0</td>
      <td>173300.0</td>
      <td>172000.0</td>
      <td>128700.0</td>
      <td>166900.0</td>
      <td>182500.0</td>
      <td>200600.0</td>
      <td>143100.0</td>
      <td>216400.0</td>
      <td>260500.0</td>
      <td>264500.0</td>
      <td>226500.0</td>
      <td>130700.0</td>
      <td>231200.0</td>
      <td>281500.0</td>
      <td>167200.0</td>
      <td>497100.0</td>
      <td>191700.0</td>
      <td>174600.0</td>
      <td>183800.0</td>
      <td>262300.0</td>
      <td>119100.0</td>
      <td>178400.0</td>
      <td>160900.0</td>
      <td>267100.0</td>
      <td>184200.0</td>
      <td>170800.0</td>
      <td>228100.0</td>
      <td>245700.0</td>
      <td>...</td>
      <td>150600.0</td>
      <td>196900.0</td>
      <td>112000.0</td>
      <td>183900.0</td>
      <td>108300.0</td>
      <td>174300.0</td>
      <td>165600.0</td>
      <td>133000.0</td>
      <td>173600.0</td>
      <td>221000.0</td>
      <td>122400.0</td>
      <td>104100.0</td>
      <td>124300.0</td>
      <td>109500.0</td>
      <td>118100.0</td>
      <td>145000.0</td>
      <td>171300.0</td>
      <td>131800.0</td>
      <td>122500.0</td>
      <td>181400.0</td>
      <td>233400.0</td>
      <td>193000.0</td>
      <td>75700.0</td>
      <td>141700.0</td>
      <td>116500.0</td>
      <td>151200.0</td>
      <td>140800.0</td>
      <td>150900.0</td>
      <td>116500.0</td>
      <td>139100.0</td>
      <td>135100.0</td>
      <td>96200.0</td>
      <td>202800.0</td>
      <td>136500.0</td>
      <td>133000.0</td>
      <td>116200.0</td>
      <td>116200.0</td>
      <td>167700.0</td>
      <td>117500.0</td>
      <td>137700.0</td>
      <td>150100.0</td>
      <td>152000.0</td>
      <td>139900.0</td>
      <td>148500.0</td>
      <td>136400.0</td>
      <td>142500.0</td>
      <td>138200.0</td>
      <td>128500.0</td>
      <td>151900.0</td>
      <td>147500.0</td>
      <td>149000.0</td>
      <td>127000.0</td>
      <td>155000.0</td>
      <td>132100.0</td>
      <td>151000.0</td>
      <td>102600.0</td>
      <td>143600.0</td>
      <td>94800.0</td>
      <td>163800.0</td>
      <td>92700.0</td>
      <td>88000.0</td>
      <td>133500.0</td>
      <td>135600.0</td>
      <td>95700.0</td>
      <td>88500.0</td>
      <td>84100.0</td>
      <td>168400.0</td>
      <td>141400.0</td>
      <td>132100.0</td>
      <td>100700.0</td>
      <td>120600.0</td>
      <td>129400.0</td>
      <td>134200.0</td>
      <td>152900.0</td>
      <td>180400.0</td>
      <td>132400.0</td>
      <td>114700.0</td>
      <td>148100.0</td>
      <td>160400.0</td>
      <td>122900.0</td>
      <td>132700.0</td>
      <td>153600.0</td>
      <td>133700.0</td>
      <td>153900.0</td>
      <td>85700.0</td>
      <td>138800.0</td>
      <td>134600.0</td>
      <td>101900.0</td>
      <td>117400.0</td>
      <td>100700.0</td>
      <td>98200.0</td>
      <td>90400.0</td>
      <td>82800.0</td>
      <td>92600.0</td>
      <td>99600.0</td>
      <td>109100.0</td>
      <td>110500.0</td>
      <td>102800.0</td>
      <td>114600.0</td>
      <td>128700.0</td>
      <td>121500.0</td>
      <td>141800.0</td>
      <td>111400.0</td>
      <td>130600.0</td>
      <td>135400.0</td>
      <td>146800.0</td>
      <td>93800.0</td>
      <td>111300.0</td>
      <td>123400.0</td>
      <td>124000.0</td>
      <td>133600.0</td>
      <td>194100.0</td>
      <td>122600.0</td>
      <td>86000.0</td>
      <td>115900.0</td>
      <td>123300.0</td>
      <td>99900.0</td>
      <td>103800.0</td>
      <td>103200.0</td>
      <td>141000.0</td>
      <td>119100.0</td>
      <td>115600.0</td>
      <td>109400.0</td>
      <td>124200.0</td>
      <td>127600.0</td>
      <td>120100.0</td>
      <td>127900.0</td>
      <td>139500.0</td>
      <td>121400.0</td>
      <td>122200.0</td>
      <td>110200.0</td>
      <td>124200.0</td>
      <td>148900.0</td>
      <td>129100.0</td>
      <td>149800.0</td>
      <td>151900.0</td>
      <td>132100.0</td>
      <td>136000.0</td>
      <td>68500.0</td>
      <td>125900.0</td>
      <td>110700.0</td>
      <td>72200.0</td>
      <td>162200.0</td>
      <td>125700.0</td>
      <td>89100.0</td>
      <td>63900.0</td>
      <td>65200.0</td>
      <td>51500.0</td>
      <td>124000.0</td>
      <td>98900.0</td>
      <td>NaN</td>
      <td>57300.0</td>
      <td>99400.0</td>
      <td>115700.0</td>
      <td>87100.0</td>
      <td>82900.0</td>
      <td>96000.0</td>
      <td>133500.0</td>
      <td>129100.0</td>
      <td>134300.0</td>
      <td>125000.0</td>
      <td>79000.0</td>
      <td>71600.0</td>
      <td>103600.0</td>
      <td>92100.0</td>
      <td>129100.0</td>
      <td>82800.0</td>
      <td>125900.0</td>
      <td>79000.0</td>
      <td>86000.0</td>
      <td>71400.0</td>
      <td>73000.0</td>
      <td>99800.0</td>
      <td>92700.0</td>
      <td>128100.0</td>
      <td>114500.0</td>
      <td>144200.0</td>
      <td>100600.0</td>
      <td>101500.0</td>
      <td>117700.0</td>
      <td>155400.0</td>
      <td>132400.0</td>
      <td>118400.0</td>
      <td>113100.0</td>
      <td>115200.0</td>
      <td>119500.0</td>
      <td>128500.0</td>
      <td>47600.0</td>
      <td>NaN</td>
      <td>69000.0</td>
      <td>53600.0</td>
      <td>65600.0</td>
      <td>59500.0</td>
      <td>65700.0</td>
      <td>NaN</td>
      <td>36400.0</td>
      <td>64300.0</td>
      <td>68600.0</td>
      <td>120700.0</td>
      <td>103000.0</td>
      <td>79400.0</td>
      <td>109700.0</td>
      <td>69000.0</td>
      <td>118400.0</td>
      <td>87900.0</td>
      <td>102400.0</td>
      <td>85000.0</td>
      <td>125900.0</td>
      <td>134300.0</td>
      <td>128800.0</td>
      <td>90400.0</td>
      <td>NaN</td>
      <td>104400.0</td>
      <td>87300.0</td>
      <td>90700.0</td>
      <td>111900.0</td>
      <td>133500.0</td>
      <td>90500.0</td>
      <td>NaN</td>
      <td>113600.0</td>
      <td>61300.0</td>
      <td>87700.0</td>
      <td>119000.0</td>
      <td>143700.0</td>
      <td>118700.0</td>
      <td>127300.0</td>
      <td>133500.0</td>
      <td>119200.0</td>
      <td>176400.0</td>
      <td>184600.0</td>
      <td>123400.0</td>
      <td>116000.0</td>
      <td>149500.0</td>
      <td>151700.0</td>
      <td>164800.0</td>
      <td>117900.0</td>
      <td>112300.0</td>
      <td>136000.0</td>
      <td>119200.0</td>
      <td>123700.0</td>
      <td>128600.0</td>
      <td>113000.0</td>
      <td>115100.0</td>
      <td>91000.0</td>
      <td>105400.0</td>
      <td>106200.0</td>
      <td>110800.0</td>
      <td>163500.0</td>
      <td>152800.0</td>
      <td>117500.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 14723 columns</p>
</div>



Now I'm going to throw in the national average column in order to work with the national data a little bit, to establish a baseline and set initial parameters for SARIMAX.


```python
zdf_by_zip['Nationwide'] = zdf_by_zip.mean(axis = 1)
```

# EDA and Visualization


```python
font = {'weight' : 'bold',
        'size'   : 22}

matplotlib.rc('font', **font)
```

Here is just a straight up visual of the national average.


```python
plt.figure(figsize = (12, 5))
plt.plot(zdf_by_zip['Nationwide']);
```


![png](output_49_0.png)


I'll go ahead and decompose it so I can see what the big picture looks like.


```python
rcParams['figure.figsize'] = 14, 5
fig = sm.tsa.seasonal_decompose(zdf_by_zip['Nationwide']).plot();
```


![png](output_51_0.png)


Autocorrelation and partial autocorrelation plots to help with determing AR and MA values.


```python
plt.figure(figsize=(12,5))
pd.plotting.autocorrelation_plot(zdf_by_zip['Nationwide'][-48:]);
```


![png](output_53_0.png)



```python
from statsmodels.graphics.tsaplots import plot_acf
from matplotlib.pylab import rcParams

rcParams['figure.figsize'] = 14, 5

plot_acf(zdf_by_zip['Nationwide'], lags = 100);
```


![png](output_54_0.png)



```python
from statsmodels.graphics.tsaplots import plot_pacf
from matplotlib.pylab import rcParams

rcParams['figure.figsize'] = 14, 5

plot_pacf(zdf_by_zip['Nationwide'], lags = 100);
```


![png](output_55_0.png)


I'll again use the example Portsmouth NH zipcode to see how one part of the picture might look on its own.


```python
plt.figure(figsize = (12, 5))
plt.plot(zdf_by_zip[3801]);
```


![png](output_57_0.png)



```python
rcParams['figure.figsize'] = 14, 5
sm.tsa.seasonal_decompose(zdf_by_zip[3801]).plot();
```


![png](output_58_0.png)



```python
rcParams['figure.figsize'] = 14, 5

plot_acf(zdf_by_zip[3801], lags = 100);
```


![png](output_59_0.png)



```python
plt.figure(figsize=(12,5))
pd.plotting.autocorrelation_plot(zdf_by_zip[3801][-48:]);
```


![png](output_60_0.png)



```python
rcParams['figure.figsize'] = 14, 5

plot_pacf(zdf_by_zip[3801], lags = 100);
```


![png](output_61_0.png)


# ARIMA Modeling

Again, starting with the national average to get a high-level overview.

I'm going to begin by iterating over models to find the version with parameters that enable the lowest aic value.

## Determining Parameters

I'll try to first find the optimal pdq and PDQS values.

The majority of my time was spent determining hyperparameters. 

I found that the lowest AIC figures were not necessarily the best ones to use. For example, many of the restults prescribed a differencing (d) level of 2. However, upon comparing this order with others that had AIC scores nearlly as low, I found that the confidence intervals for the d = 2 orders were massively larger than some of the others. Higher differencing also made for a steeper slope, which I was hesitent to use in real estate since a line curving upward seems overly optimistic, and to stay on the conservative side.

In addition, since I was beginning at the national level, I needed to find a way to model all 14,000 records without necessarily determining unique parameters for each one, which would have taken weeks of computing time. In investigating individual zip codes, I found that the top order sets were very close to each other in terms of AIC and forecasted value, once differencing was taken at a value of 1 as standard. Knowing this, I set out to find the best possible 'universal key' parameters that might work for the highest number of zip codes. It might not be perfect for each zip, but it was the best way of approximating the overall set and finding a strong outcome for the top zips in the nation.

### ARIMA Version

Here, I'm generating 25 random samples with replacement to determine which order sets wind up at the top. I will ultimately take the top 10 scores from 25 different zip codes, keep only the codes they **all** have in common, and then I will run that test 25 times, then tally the results and pick the very best scoring set of parameters.


```python
#Generate a list of all zips and select a sample of 25.
order_list = []

for _ in range(0, 25):
    zip_list = zdf_by_zip.columns
    random_zips = np.random.choice(zip_list, 25)
    zip_dict = dict()
    univ_list = []

    for azip in random_zips: 
        # Running the aic_search function that will populate the dictionary
        # with the top ten order combinations for each selected zip.
        # aic_search is defined above with my functions.
        univ_list, order_df, fails = aic_search(zdf_by_zip[azip], p_max = 2,
                                                d_max = 2,  
                                                q_max = 2,
                                                end = '2017-04-01')
    
    # The next piece really weird. I got a very strange error when running the above
    # code. When inputting the zip directly, it worked great. When inputting
    # from the random list, it failed. Only for a differenccing of 2. 
    # Weird, right? After much attempted debugging with 
    # changing the type etc., I found that running it twice woked fine. Go 
    # figure. I just ran the same line twice. 
    for azip in random_zips:
        univ_list, order_df, fails = aic_search(zdf_by_zip[azip], p_max = 4, 
                                                d_max = 2, 
                                                q_max = 4,
                                                end = '2017-04-01')

        # The loop updates univ_list with the combinations as they are tried.
        # We only use the ones that have 10 or more combinations and take the top 
        # ten from the ones that do.
        univ_list = [x[0] for x in univ_list]
        if len(univ_list) >= 10:
            zip_dict[azip] = univ_list[:10]

    # Initialize the list that will temporarily house the dict values as the 
    # dict is iterated.
    comb_list = []
    
    # Now I have a dict with zips as the keys and lists of 10 top order sets
    # as the values.
    start = 0
    for v in zip_dict.values():
        # If this is the first run, I'll need to move over the values from the 
        # first entry to get things started. Otherwise it has nothing to 
        # compare itself against for paring. And there is no need to compare
        # it against itself.
        if start == 0:
            comb_list = v.copy()
            start = 1
        # This is where I'm paring down the list to only those order sets
        # that the zip codes have in common in their top ten.
        for i in comb_list:
            if i in v:
                # If the value is still here in this next entry, good, move on.
                continue
            else:
                # If it's not, takie it out of the master list.
                comb_list.remove(i)
                continue
    
    # Appending the master list of all the combinations across 25 random runs 
    # of 25 zip codes each.
    order_list.append(comb_list)

```


```python
# Making a copy since that job takes a long time to run and I don't want to 
# lose the info.
order_list2 = order_list.copy()

# Flatten so that all order sets are in one list. Sources don't matter now.
flat_list = [item for sublist in order_list2 for item in sublist]

# Make them unique.
set_list = set(flat_list)

# Setting up the final dictionary for all the order set counts.
counted_orders = dict()

# Tallying the total combinations that were in the most sets throughout the tests.
for orders in set_list:
    counted_orders[orders] = flat_list.count(orders)

Print('After 25 runs of 25 random samples, these are the counts of the top model parameters that appear the most times')
print('-----'*20)
counted_orders
```

Attemping this several times, 1,1,3 turned out to be the top choice every time. That's our winner, and we'll go with that for all future modeling.

Note that I ran grid searches for SARIMAX as well, but since I ultimately did not choose that modeling method, I have since removed those searches for the sake of tidiness. 

## Preliminary Tests and Forecasts

### SARIMAX Modeling

I just mentioned that I wound up not using the SARIMAX modeling, but I am keeping this part for posterity, to demonstrate some of the reasons why I chose not to pursue this method.

#### SARIMAX Nationwide Test


```python
train_test_SARIMAX(zdf_by_zip['Nationwide'],
                   order = (1,1,3),
                   start = '2013-04-01')
```

    ==============================================================================
                     coef    std err          z      P>|z|      [0.025      0.975]
    ------------------------------------------------------------------------------
    ar.L1          0.9885      0.016     63.282      0.000       0.958       1.019
    ma.L1         -0.2688      0.043     -6.253      0.000      -0.353      -0.185
    ma.L2          0.0417      0.051      0.818      0.413      -0.058       0.142
    ma.L3         -0.0374      0.063     -0.590      0.555      -0.162       0.087
    sigma2      9.582e+04   3280.754     29.207      0.000    8.94e+04    1.02e+05
    ==============================================================================
    AIC:  2864.283183277808



![png](output_78_1.png)



![png](output_78_2.png)


#### SARIMAX Example Test

This is where I decided not to use the SARIMAX model. Some of the zip codes I looked at did not seem to work right when the seasonality was removed. They worked ok with the parameters established for seasonality, but then they broke without them. So I changed gears and went with ARIMA. 

Below is the graph for Portsmouth. There is something clearly wrong with it, as it has a strong upward trajectory up until the testing phase, then becomes a straight, flat line. With ARIMA, I didn't have this problem.

I did try this with top order sets from both SARIMAX and ARIMA, and both parameter sets had the same problem.


```python
train_test_SARIMAX(zdf_by_zip[3801],
                   order = (1,1,3),
                   start = '2013-04-01')
```

    ==============================================================================
                     coef    std err          z      P>|z|      [0.025      0.975]
    ------------------------------------------------------------------------------
    ar.L1          0.8858      0.040     22.223      0.000       0.808       0.964
    ma.L1          0.9376      0.060     15.543      0.000       0.819       1.056
    ma.L2          0.1727      0.096      1.797      0.072      -0.016       0.361
    ma.L3         -0.1219      0.048     -2.550      0.011      -0.216      -0.028
    sigma2      2.901e+05   1.69e+04     17.132      0.000    2.57e+05    3.23e+05
    ==============================================================================
    AIC:  3104.844558781295



![png](output_81_1.png)



![png](output_81_2.png)


#### SARIMAX Nationwide Foreccast

I'll put together a complete model including diagnostics and forecasts.

Note that I had once had a different set of numbers, including seasonal figures. I decided to get rid of the seasonal figures once I realized that although the seasonal graph shows a clear seasonal trend, the variance is so little that it isn't worth plugging in the figures.


```python
forecast_SARIMAX(data = zdf_by_zip['Nationwide'], 
          order = (1,1,3),
          start = '2018-04-01',
          end = '2023-04-01')
```

    ==============================================================================
                     coef    std err          z      P>|z|      [0.025      0.975]
    ------------------------------------------------------------------------------
    ar.L1          0.9518      0.007    127.622      0.000       0.937       0.966
    ma.L1         -0.8947      0.010    -91.922      0.000      -0.914      -0.876
    ma.L2         -0.0001      0.005     -0.019      0.985      -0.011       0.010
    ma.L3          0.0095      0.005      1.948      0.051   -6.02e-05       0.019
    sigma2      4.408e+05   2.65e-10   1.66e+15      0.000    4.41e+05    4.41e+05
    ==============================================================================
    AIC:  4356.699513358408



![png](output_84_1.png)



![png](output_84_2.png)


#### SARIMAX Example Forecast

Now that we have starting parameters for a nationwide model, I'll try out the model_zip function on a the same Portsmouth example from earlier.

This is going to be one of the reasons for ultimately picking ARIMA over SARIMAX for my modeling. There is something clearly wrong with the below SARIMAX model, even though order test liked this order. I specifically chose this zipcode because there's something wrong with it, to illustrate my choice of model.


```python
forecast_SARIMAX(zdf_by_zip[60625], 
          order = (1,1,3),
          start = '2018-04-01',
          end = '2028-04-01')
```

    ==============================================================================
                     coef    std err          z      P>|z|      [0.025      0.975]
    ------------------------------------------------------------------------------
    ar.L1          0.9736      0.009    102.953      0.000       0.955       0.992
    ma.L1         -0.9434      0.013    -75.375      0.000      -0.968      -0.919
    ma.L2          0.0443      0.008      5.826      0.000       0.029       0.059
    ma.L3         -0.0484      0.007     -7.271      0.000      -0.061      -0.035
    sigma2      4.535e+06    1.7e-10   2.66e+16      0.000    4.53e+06    4.53e+06
    ==============================================================================
    AIC:  4824.407441928715



![png](output_88_1.png)



![png](output_88_2.png)


### ARIMA Modeling

#### ARIMA Nationwide Test


```python
train_test_ARIMA(zdf_by_zip['Nationwide'], (1,1,3), start = '2013-04-01')
```

                                 ARIMA Model Results                              
    ==============================================================================
    Dep. Variable:           D.Nationwide   No. Observations:                  204
    Model:                 ARIMA(1, 1, 3)   Log Likelihood               -1454.808
    Method:                       css-mle   S.D. of innovations            300.583
    Date:                Tue, 21 Jan 2020   AIC                           2921.616
    Time:                        15:33:03   BIC                           2941.524
    Sample:                    05-01-1996   HQIC                          2929.669
                             - 04-01-2013                                         
    ======================================================================================
                             coef    std err          z      P>|z|      [0.025      0.975]
    --------------------------------------------------------------------------------------
    const                576.1130    663.252      0.869      0.386    -723.837    1876.063
    ar.L1.D.Nationwide     0.9807      0.012     79.712      0.000       0.957       1.005
    ma.L1.D.Nationwide    -0.2665      0.073     -3.639      0.000      -0.410      -0.123
    ma.L2.D.Nationwide     0.0421      0.065      0.646      0.519      -0.086       0.170
    ma.L3.D.Nationwide    -0.0357      0.071     -0.505      0.614      -0.175       0.103
                                        Roots                                    
    =============================================================================
                      Real          Imaginary           Modulus         Frequency
    -----------------------------------------------------------------------------
    AR.1            1.0196           +0.0000j            1.0196            0.0000
    MA.1            2.5519           -0.0000j            2.5519           -0.0000
    MA.2           -0.6870           -3.2391j            3.3111           -0.2833
    MA.3           -0.6870           +3.2391j            3.3111            0.2833
    -----------------------------------------------------------------------------



![png](output_91_1.png)


#### ARIMA Example Test


```python
train_test_ARIMA(zdf_by_zip[3801], (1,1,3), start = '2013-04-01')
```

                                 ARIMA Model Results                              
    ==============================================================================
    Dep. Variable:                 D.3801   No. Observations:                  204
    Model:                 ARIMA(1, 1, 3)   Log Likelihood               -1570.037
    Method:                       css-mle   S.D. of innovations            526.625
    Date:                Tue, 21 Jan 2020   AIC                           3152.073
    Time:                        15:39:57   BIC                           3171.982
    Sample:                    05-01-1996   HQIC                          3160.127
                             - 04-01-2013                                         
    ================================================================================
                       coef    std err          z      P>|z|      [0.025      0.975]
    --------------------------------------------------------------------------------
    const          976.4706    553.550      1.764      0.079    -108.467    2061.408
    ar.L1.D.3801     0.8853      0.040     22.181      0.000       0.807       0.963
    ma.L1.D.3801     0.9973      0.076     13.146      0.000       0.849       1.146
    ma.L2.D.3801     0.1100      0.106      1.042      0.299      -0.097       0.317
    ma.L3.D.3801    -0.3196      0.077     -4.162      0.000      -0.470      -0.169
                                        Roots                                    
    =============================================================================
                      Real          Imaginary           Modulus         Frequency
    -----------------------------------------------------------------------------
    AR.1            1.1296           +0.0000j            1.1296            0.0000
    MA.1           -0.9761           -0.6401j            1.1673           -0.4076
    MA.2           -0.9761           +0.6401j            1.1673            0.4076
    MA.3            2.2963           -0.0000j            2.2963           -0.0000
    -----------------------------------------------------------------------------



![png](output_93_1.png)


#### ARIMA Nationwide Forecast

In contrast to the 3801 example further up, the graph turned out differently here using an ARIMA model. For each model, I used the top order result *for that model*, so they should both be getting optimum treatment. I also tried using the same number for both, with results that were not much different. Not only did the line make more sense, but the confidence intervals are much smaller as well.


```python
forecast_ARIMA(zdf_by_zip['Nationwide'], (1,1,3), forecast_length = 120)
```


![png](output_96_0.png)


#### ARIMA Example Forecast


```python
forecast_ARIMA(zdf_by_zip[60618], (1,1,3), forecast_length = 120)
```


![png](output_98_0.png)


## Modeling All Zipcodes

### Creating Supplementary DataFrame

I first need to set up the beginning of what will be the master dataframe, including all of the zip info (city etc.)


```python
# Creating a basis for the columns I will want to add to the new master 
# dataframe as it is generated.
zip_info = pd.DataFrame(zdf[['RegionName', 'City', 'State', 'Metro', 
                             'CountyName', 'SizeRank']])
```


```python
zip_info = zip_info.sort_values('RegionName')
```


```python
zip_info.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>RegionName</th>
      <th>City</th>
      <th>State</th>
      <th>Metro</th>
      <th>CountyName</th>
      <th>SizeRank</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>5850</th>
      <td>1001</td>
      <td>Agawam</td>
      <td>MA</td>
      <td>Springfield</td>
      <td>Hampden</td>
      <td>5851</td>
    </tr>
    <tr>
      <th>4199</th>
      <td>1002</td>
      <td>Amherst</td>
      <td>MA</td>
      <td>Springfield</td>
      <td>Hampshire</td>
      <td>4200</td>
    </tr>
    <tr>
      <th>11213</th>
      <td>1005</td>
      <td>Barre</td>
      <td>MA</td>
      <td>Worcester</td>
      <td>Worcester</td>
      <td>11214</td>
    </tr>
    <tr>
      <th>6850</th>
      <td>1007</td>
      <td>Belchertown</td>
      <td>MA</td>
      <td>Springfield</td>
      <td>Hampshire</td>
      <td>6851</td>
    </tr>
    <tr>
      <th>14547</th>
      <td>1008</td>
      <td>Blandford</td>
      <td>MA</td>
      <td>Springfield</td>
      <td>Hampden</td>
      <td>14548</td>
    </tr>
  </tbody>
</table>
</div>




```python
zip_info.set_index(['RegionName'], inplace = True)
```

### Building the Model

This is the big job. I'm going to use the ARIMA modling to create a master projection DataFrame of 1-10 year projections for every zipcode in the source. Function is defined under Functions above.

national_models is defined above with the rest of my functions.


```python
# all_zip_models = national_models(source=zdf_by_zip, info=zip_info, 
#                                  params=(1,1,3))
```

This took like 15 hours to run. I don't want to lose it.


```python
# all_zip_models.to_csv('all_zip_models_113.csv')
```

Below is for when I'm running off a csv, so I don't have to re-make that dataframe every time I re-run the notebook.


```python
all_zip_models = pd.read_csv('all_zip_models_113.csv')
```


```python
all_zip_models.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>Zip</th>
      <th>City</th>
      <th>State</th>
      <th>Metro</th>
      <th>CountyName</th>
      <th>SizeRank</th>
      <th>2018-04-01</th>
      <th>1_Year</th>
      <th>1_Gain</th>
      <th>2_Year</th>
      <th>2_Gain</th>
      <th>3_Year</th>
      <th>3_Gain</th>
      <th>4_Year</th>
      <th>4_Gain</th>
      <th>5_Year</th>
      <th>5_Gain</th>
      <th>6_Year</th>
      <th>6_Gain</th>
      <th>7_Year</th>
      <th>7_Gain</th>
      <th>8_Year</th>
      <th>8_Gain</th>
      <th>9_Year</th>
      <th>9_Gain</th>
      <th>10_Year</th>
      <th>10_Gain</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>1001</td>
      <td>Agawam</td>
      <td>MA</td>
      <td>Springfield</td>
      <td>Hampden</td>
      <td>5851</td>
      <td>223600.0</td>
      <td>228007.34</td>
      <td>0.02</td>
      <td>232698.84</td>
      <td>0.04</td>
      <td>237330.68</td>
      <td>0.06</td>
      <td>241936.88</td>
      <td>0.08</td>
      <td>246532.07</td>
      <td>0.10</td>
      <td>251122.53</td>
      <td>0.12</td>
      <td>255710.95</td>
      <td>0.14</td>
      <td>260298.50</td>
      <td>0.16</td>
      <td>264885.68</td>
      <td>0.18</td>
      <td>269472.69</td>
      <td>0.21</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1002</td>
      <td>Amherst</td>
      <td>MA</td>
      <td>Springfield</td>
      <td>Hampshire</td>
      <td>4200</td>
      <td>353300.0</td>
      <td>371533.37</td>
      <td>0.05</td>
      <td>383994.76</td>
      <td>0.09</td>
      <td>394106.87</td>
      <td>0.12</td>
      <td>403327.09</td>
      <td>0.14</td>
      <td>412208.71</td>
      <td>0.17</td>
      <td>420961.77</td>
      <td>0.19</td>
      <td>429666.04</td>
      <td>0.22</td>
      <td>438351.77</td>
      <td>0.24</td>
      <td>447030.47</td>
      <td>0.27</td>
      <td>455706.50</td>
      <td>0.29</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>1005</td>
      <td>Barre</td>
      <td>MA</td>
      <td>Worcester</td>
      <td>Worcester</td>
      <td>11214</td>
      <td>220700.0</td>
      <td>233987.20</td>
      <td>0.06</td>
      <td>244116.51</td>
      <td>0.11</td>
      <td>252374.41</td>
      <td>0.14</td>
      <td>259605.39</td>
      <td>0.18</td>
      <td>266272.83</td>
      <td>0.21</td>
      <td>272631.05</td>
      <td>0.24</td>
      <td>278819.57</td>
      <td>0.26</td>
      <td>284914.98</td>
      <td>0.29</td>
      <td>290959.29</td>
      <td>0.32</td>
      <td>296975.56</td>
      <td>0.35</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>1007</td>
      <td>Belchertown</td>
      <td>MA</td>
      <td>Springfield</td>
      <td>Hampshire</td>
      <td>6851</td>
      <td>270600.0</td>
      <td>275322.74</td>
      <td>0.02</td>
      <td>280590.95</td>
      <td>0.04</td>
      <td>285931.71</td>
      <td>0.06</td>
      <td>291313.87</td>
      <td>0.08</td>
      <td>296719.63</td>
      <td>0.10</td>
      <td>302138.86</td>
      <td>0.12</td>
      <td>307565.76</td>
      <td>0.14</td>
      <td>312997.04</td>
      <td>0.16</td>
      <td>318430.83</td>
      <td>0.18</td>
      <td>323866.03</td>
      <td>0.20</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>1008</td>
      <td>Blandford</td>
      <td>MA</td>
      <td>Springfield</td>
      <td>Hampden</td>
      <td>14548</td>
      <td>214200.0</td>
      <td>224827.76</td>
      <td>0.05</td>
      <td>230960.68</td>
      <td>0.08</td>
      <td>235983.13</td>
      <td>0.10</td>
      <td>240716.62</td>
      <td>0.12</td>
      <td>245374.91</td>
      <td>0.15</td>
      <td>250013.64</td>
      <td>0.17</td>
      <td>254647.27</td>
      <td>0.19</td>
      <td>259279.58</td>
      <td>0.21</td>
      <td>263911.55</td>
      <td>0.23</td>
      <td>268543.43</td>
      <td>0.25</td>
    </tr>
  </tbody>
</table>
</div>



Getting rid of null values that will mess up my DataFrame.


```python
# Create a list of zipcodes with null values.
null_zips = []

for column in zdf_by_zip.columns:
    if zdf_by_zip[column].isnull().sum() > 0:
        null_zips.append(column)
        
```


```python
# Remove everything from that list.
for zip in null_zips:
    try:
        all_zip_models.drop(all_zip_models[all_zip_models['Zip'] == zip].index,
                           inplace = True)
    except:
        continue
```

Check to see how many records have been cut from the original, after nulls were removed and some models no doubt failed. 


```python
all_zip_models.shape
```




    (13594, 28)



Resetting the index for the sake of the loaded file.


```python
# all_zip_models = all_zip_models.drop(['Unnamed: 0'], axis = 1)
```


```python
all_zip_models.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>Zip</th>
      <th>City</th>
      <th>State</th>
      <th>Metro</th>
      <th>CountyName</th>
      <th>SizeRank</th>
      <th>2018-04-01</th>
      <th>1_Year</th>
      <th>1_Gain</th>
      <th>2_Year</th>
      <th>2_Gain</th>
      <th>3_Year</th>
      <th>3_Gain</th>
      <th>4_Year</th>
      <th>4_Gain</th>
      <th>5_Year</th>
      <th>5_Gain</th>
      <th>6_Year</th>
      <th>6_Gain</th>
      <th>7_Year</th>
      <th>7_Gain</th>
      <th>8_Year</th>
      <th>8_Gain</th>
      <th>9_Year</th>
      <th>9_Gain</th>
      <th>10_Year</th>
      <th>10_Gain</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>1001</td>
      <td>Agawam</td>
      <td>MA</td>
      <td>Springfield</td>
      <td>Hampden</td>
      <td>5851</td>
      <td>223600.0</td>
      <td>228007.34</td>
      <td>0.02</td>
      <td>232698.84</td>
      <td>0.04</td>
      <td>237330.68</td>
      <td>0.06</td>
      <td>241936.88</td>
      <td>0.08</td>
      <td>246532.07</td>
      <td>0.10</td>
      <td>251122.53</td>
      <td>0.12</td>
      <td>255710.95</td>
      <td>0.14</td>
      <td>260298.50</td>
      <td>0.16</td>
      <td>264885.68</td>
      <td>0.18</td>
      <td>269472.69</td>
      <td>0.21</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1002</td>
      <td>Amherst</td>
      <td>MA</td>
      <td>Springfield</td>
      <td>Hampshire</td>
      <td>4200</td>
      <td>353300.0</td>
      <td>371533.37</td>
      <td>0.05</td>
      <td>383994.76</td>
      <td>0.09</td>
      <td>394106.87</td>
      <td>0.12</td>
      <td>403327.09</td>
      <td>0.14</td>
      <td>412208.71</td>
      <td>0.17</td>
      <td>420961.77</td>
      <td>0.19</td>
      <td>429666.04</td>
      <td>0.22</td>
      <td>438351.77</td>
      <td>0.24</td>
      <td>447030.47</td>
      <td>0.27</td>
      <td>455706.50</td>
      <td>0.29</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>1005</td>
      <td>Barre</td>
      <td>MA</td>
      <td>Worcester</td>
      <td>Worcester</td>
      <td>11214</td>
      <td>220700.0</td>
      <td>233987.20</td>
      <td>0.06</td>
      <td>244116.51</td>
      <td>0.11</td>
      <td>252374.41</td>
      <td>0.14</td>
      <td>259605.39</td>
      <td>0.18</td>
      <td>266272.83</td>
      <td>0.21</td>
      <td>272631.05</td>
      <td>0.24</td>
      <td>278819.57</td>
      <td>0.26</td>
      <td>284914.98</td>
      <td>0.29</td>
      <td>290959.29</td>
      <td>0.32</td>
      <td>296975.56</td>
      <td>0.35</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>1007</td>
      <td>Belchertown</td>
      <td>MA</td>
      <td>Springfield</td>
      <td>Hampshire</td>
      <td>6851</td>
      <td>270600.0</td>
      <td>275322.74</td>
      <td>0.02</td>
      <td>280590.95</td>
      <td>0.04</td>
      <td>285931.71</td>
      <td>0.06</td>
      <td>291313.87</td>
      <td>0.08</td>
      <td>296719.63</td>
      <td>0.10</td>
      <td>302138.86</td>
      <td>0.12</td>
      <td>307565.76</td>
      <td>0.14</td>
      <td>312997.04</td>
      <td>0.16</td>
      <td>318430.83</td>
      <td>0.18</td>
      <td>323866.03</td>
      <td>0.20</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>1008</td>
      <td>Blandford</td>
      <td>MA</td>
      <td>Springfield</td>
      <td>Hampden</td>
      <td>14548</td>
      <td>214200.0</td>
      <td>224827.76</td>
      <td>0.05</td>
      <td>230960.68</td>
      <td>0.08</td>
      <td>235983.13</td>
      <td>0.10</td>
      <td>240716.62</td>
      <td>0.12</td>
      <td>245374.91</td>
      <td>0.15</td>
      <td>250013.64</td>
      <td>0.17</td>
      <td>254647.27</td>
      <td>0.19</td>
      <td>259279.58</td>
      <td>0.21</td>
      <td>263911.55</td>
      <td>0.23</td>
      <td>268543.43</td>
      <td>0.25</td>
    </tr>
  </tbody>
</table>
</div>



Since that DataFrame is such a bear to create, I'll copy it before doing any damage.


```python
all_zip_ranked = all_zip_models.sort_values('10_Gain', ascending = False)
```

Adding in a column to display each zip's overall rank.


```python
all_zip_ranked['Rank'] = [i + 1 for i in range(0, len(all_zip_ranked))]
```


```python
state_group = group_areas(all_zip_ranked, 'State', 10)
```


```python
all_zip_ranked[all_zip_ranked['City'] == 'Fort Collins']
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>Zip</th>
      <th>City</th>
      <th>State</th>
      <th>Metro</th>
      <th>CountyName</th>
      <th>SizeRank</th>
      <th>2018-04-01</th>
      <th>1_Year</th>
      <th>1_Gain</th>
      <th>2_Year</th>
      <th>2_Gain</th>
      <th>3_Year</th>
      <th>3_Gain</th>
      <th>4_Year</th>
      <th>4_Gain</th>
      <th>5_Year</th>
      <th>5_Gain</th>
      <th>6_Year</th>
      <th>6_Gain</th>
      <th>7_Year</th>
      <th>7_Gain</th>
      <th>8_Year</th>
      <th>8_Gain</th>
      <th>9_Year</th>
      <th>9_Gain</th>
      <th>10_Year</th>
      <th>10_Gain</th>
      <th>Rank</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>11086</th>
      <td>11086</td>
      <td>80524</td>
      <td>Fort Collins</td>
      <td>CO</td>
      <td>Fort Collins</td>
      <td>Larimer</td>
      <td>1827</td>
      <td>411600.0</td>
      <td>436037.45</td>
      <td>0.06</td>
      <td>459438.50</td>
      <td>0.12</td>
      <td>480748.72</td>
      <td>0.17</td>
      <td>500394.98</td>
      <td>0.22</td>
      <td>518716.99</td>
      <td>0.26</td>
      <td>535985.10</td>
      <td>0.30</td>
      <td>552414.46</td>
      <td>0.34</td>
      <td>568176.32</td>
      <td>0.38</td>
      <td>583406.95</td>
      <td>0.42</td>
      <td>598214.79</td>
      <td>0.45</td>
      <td>990</td>
    </tr>
    <tr>
      <th>11085</th>
      <td>11085</td>
      <td>80521</td>
      <td>Fort Collins</td>
      <td>CO</td>
      <td>Fort Collins</td>
      <td>Larimer</td>
      <td>2492</td>
      <td>369500.0</td>
      <td>391598.25</td>
      <td>0.06</td>
      <td>411250.92</td>
      <td>0.11</td>
      <td>428810.41</td>
      <td>0.16</td>
      <td>444864.29</td>
      <td>0.20</td>
      <td>459835.20</td>
      <td>0.24</td>
      <td>474027.13</td>
      <td>0.28</td>
      <td>487658.74</td>
      <td>0.32</td>
      <td>500887.33</td>
      <td>0.36</td>
      <td>513826.02</td>
      <td>0.39</td>
      <td>526556.19</td>
      <td>0.43</td>
      <td>1162</td>
    </tr>
    <tr>
      <th>11088</th>
      <td>11088</td>
      <td>80526</td>
      <td>Fort Collins</td>
      <td>CO</td>
      <td>Fort Collins</td>
      <td>Larimer</td>
      <td>928</td>
      <td>369000.0</td>
      <td>387726.86</td>
      <td>0.05</td>
      <td>404297.93</td>
      <td>0.10</td>
      <td>419177.56</td>
      <td>0.14</td>
      <td>432861.53</td>
      <td>0.17</td>
      <td>445700.30</td>
      <td>0.21</td>
      <td>457941.61</td>
      <td>0.24</td>
      <td>469760.58</td>
      <td>0.27</td>
      <td>481281.01</td>
      <td>0.30</td>
      <td>492590.39</td>
      <td>0.33</td>
      <td>503750.59</td>
      <td>0.37</td>
      <td>2321</td>
    </tr>
    <tr>
      <th>11087</th>
      <td>11087</td>
      <td>80525</td>
      <td>Fort Collins</td>
      <td>CO</td>
      <td>Fort Collins</td>
      <td>Larimer</td>
      <td>312</td>
      <td>408900.0</td>
      <td>423621.35</td>
      <td>0.04</td>
      <td>436133.53</td>
      <td>0.07</td>
      <td>447892.11</td>
      <td>0.10</td>
      <td>459415.83</td>
      <td>0.12</td>
      <td>470866.36</td>
      <td>0.15</td>
      <td>482294.09</td>
      <td>0.18</td>
      <td>493714.70</td>
      <td>0.21</td>
      <td>505133.10</td>
      <td>0.24</td>
      <td>516550.81</td>
      <td>0.26</td>
      <td>527968.30</td>
      <td>0.29</td>
      <td>4921</td>
    </tr>
    <tr>
      <th>11089</th>
      <td>11089</td>
      <td>80528</td>
      <td>Fort Collins</td>
      <td>CO</td>
      <td>Fort Collins</td>
      <td>Larimer</td>
      <td>5637</td>
      <td>482300.0</td>
      <td>489571.02</td>
      <td>0.02</td>
      <td>501608.11</td>
      <td>0.04</td>
      <td>513972.52</td>
      <td>0.07</td>
      <td>526358.26</td>
      <td>0.09</td>
      <td>538745.40</td>
      <td>0.12</td>
      <td>551132.62</td>
      <td>0.14</td>
      <td>563519.85</td>
      <td>0.17</td>
      <td>575907.08</td>
      <td>0.19</td>
      <td>588294.31</td>
      <td>0.22</td>
      <td>600681.54</td>
      <td>0.25</td>
      <td>7398</td>
    </tr>
  </tbody>
</table>
</div>



# Assessment

My goal here is going to be to drill down to the state, county, and city levels to find the top area before determining the top zip codes, in order to identify an area that is more secure in the surrounding areas.


```python
state_group = group_areas(all_zip_ranked, 'State', 10)
```


```python
# state_group["PercentGain"] = state_group['10_Gain'] * 100
# state_group['PercentGain'] = state_group['PercentGain'].astype(int)
```


```python
state_group.head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>10_Gain</th>
      <th>Rank</th>
    </tr>
    <tr>
      <th>State</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>CO</th>
      <td>0.376599</td>
      <td>1</td>
    </tr>
    <tr>
      <th>WA</th>
      <td>0.376518</td>
      <td>2</td>
    </tr>
    <tr>
      <th>FL</th>
      <td>0.366317</td>
      <td>3</td>
    </tr>
    <tr>
      <th>NV</th>
      <td>0.364545</td>
      <td>4</td>
    </tr>
    <tr>
      <th>SD</th>
      <td>0.348000</td>
      <td>5</td>
    </tr>
    <tr>
      <th>OR</th>
      <td>0.342077</td>
      <td>6</td>
    </tr>
    <tr>
      <th>TN</th>
      <td>0.330711</td>
      <td>7</td>
    </tr>
    <tr>
      <th>HI</th>
      <td>0.323529</td>
      <td>8</td>
    </tr>
    <tr>
      <th>MN</th>
      <td>0.316356</td>
      <td>9</td>
    </tr>
    <tr>
      <th>CA</th>
      <td>0.313181</td>
      <td>10</td>
    </tr>
  </tbody>
</table>
</div>




```python
state_group.index[0]
```




    'CO'



Below was an ill-fated attempt to create a map of Colorado showing ROI by county, before I realized we didn't have enough data. Keeping for posterity.


```python
# colorado = all_zip_ranked[all_zip_ranked['State'] == 'CO']
# co_counties = group_areas(colorado, 'CountyName', 10)
# co_counties.to_csv('co_counties.csv')

# !pip install geopandas
# !pip install pyshp
# !pip install shapely
# !pip install plotly-geo

# import plotly.figure_factory as ff

# fips = ['06021', '06023', '06027',
#         '06029', '06033', '06059',
#         '06047', '06049', '06051',
#         '06055', '06061']
# values = [1,2,3,4,5,6,7,8,9,10,11]

# fig = ff.create_choropleth(fips=fips, values=values)
# fig.layout.template = None
# fig.show()
```

Below code was used to generate a US map for the presenation, through plotly's Web interface.


```python
# import plotly.plotly as py
# import chart_studio

# chart_studio.tools.set_credentials_file(username='terryollila', api_key='JG9BIy9Sn131zwKgTQGW')

# scl = [[0.0, 'rgb(242,240,247)'],[0.2, 'rgb(218,218,235)'],[0.4, 'rgb(188,189,220)'],\
#             [0.6, 'rgb(158,154,200)'],[0.8, 'rgb(117,107,177)'],[1.0, 'rgb(84,39,143)']]

# data = [ dict(
#         type='choropleth',
#         colorscale = scl,
#         autocolorscale = False,
#         locations = state_group.index,
#         z = state_group['PercentGain'],
#         locationmode = 'USA-states',
#         text = state_group.index,
#         marker = dict(
#             line = dict (
#                 color = 'rgb(255,255,255)',
#                 width = 2
#             )
#         ),
#         colorbar = dict(
#             title = "% Gain"
#         )
#     ) ]

# layout = dict(
#         title = 'Forecasted Real Estate Growth by State\n2018 - 2028',
#         geo = dict(
#             scope='usa',
#             projection=dict( type='albers usa' ),
#             showlakes = True,
#             lakecolor = 'rgb(255, 255, 255)',
#         ),
#     )

# fig = dict( data=data, layout=layout )

# url = py.plot( fig, filename='d3-cloropleth-map' )
```

## Find the Top Cities in Each County

This is going to be the final result. I'm going to take the top zip code, and from there I'll take the top county, then from inside that county I will take the top city, and within the top city, I will take 5 top zip codes as the final recommendation.

The below function does all the heavy of the drill-down from state to zip level. It's defined with my other functions above.


```python
top_zips = drill_for_zips('CO', data=all_zip_ranked, num_results = 10, years_ahead = 10)
```


```python
top_zips
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>Zip</th>
      <th>City</th>
      <th>State</th>
      <th>Metro</th>
      <th>CountyName</th>
      <th>SizeRank</th>
      <th>2018-04-01</th>
      <th>1_Year</th>
      <th>1_Gain</th>
      <th>2_Year</th>
      <th>2_Gain</th>
      <th>3_Year</th>
      <th>3_Gain</th>
      <th>4_Year</th>
      <th>4_Gain</th>
      <th>5_Year</th>
      <th>5_Gain</th>
      <th>6_Year</th>
      <th>6_Gain</th>
      <th>7_Year</th>
      <th>7_Gain</th>
      <th>8_Year</th>
      <th>8_Gain</th>
      <th>9_Year</th>
      <th>9_Gain</th>
      <th>10_Year</th>
      <th>10_Gain</th>
      <th>Rank</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>11012</th>
      <td>11012</td>
      <td>80204</td>
      <td>Denver</td>
      <td>CO</td>
      <td>Denver</td>
      <td>Denver</td>
      <td>2156</td>
      <td>371600.0</td>
      <td>417634.06</td>
      <td>0.12</td>
      <td>460441.08</td>
      <td>0.24</td>
      <td>499683.23</td>
      <td>0.34</td>
      <td>535891.86</td>
      <td>0.44</td>
      <td>569519.10</td>
      <td>0.53</td>
      <td>600949.71</td>
      <td>0.62</td>
      <td>630511.10</td>
      <td>0.70</td>
      <td>658481.88</td>
      <td>0.77</td>
      <td>685099.13</td>
      <td>0.84</td>
      <td>710564.58</td>
      <td>0.91</td>
      <td>32</td>
    </tr>
    <tr>
      <th>11011</th>
      <td>11011</td>
      <td>80203</td>
      <td>Denver</td>
      <td>CO</td>
      <td>Denver</td>
      <td>Denver</td>
      <td>1941</td>
      <td>571500.0</td>
      <td>640146.30</td>
      <td>0.12</td>
      <td>699335.96</td>
      <td>0.22</td>
      <td>750624.50</td>
      <td>0.31</td>
      <td>795887.03</td>
      <td>0.39</td>
      <td>836553.68</td>
      <td>0.46</td>
      <td>873715.16</td>
      <td>0.53</td>
      <td>908203.32</td>
      <td>0.59</td>
      <td>940652.60</td>
      <td>0.65</td>
      <td>971546.88</td>
      <td>0.70</td>
      <td>1001255.20</td>
      <td>0.75</td>
      <td>102</td>
    </tr>
    <tr>
      <th>11023</th>
      <td>11023</td>
      <td>80219</td>
      <td>Denver</td>
      <td>CO</td>
      <td>Denver</td>
      <td>Denver</td>
      <td>611</td>
      <td>315700.0</td>
      <td>348074.83</td>
      <td>0.10</td>
      <td>376272.10</td>
      <td>0.19</td>
      <td>400569.75</td>
      <td>0.27</td>
      <td>421963.48</td>
      <td>0.34</td>
      <td>441194.74</td>
      <td>0.40</td>
      <td>458815.70</td>
      <td>0.45</td>
      <td>475237.50</td>
      <td>0.51</td>
      <td>490766.34</td>
      <td>0.55</td>
      <td>505630.22</td>
      <td>0.60</td>
      <td>519998.91</td>
      <td>0.65</td>
      <td>190</td>
    </tr>
    <tr>
      <th>11027</th>
      <td>11027</td>
      <td>80223</td>
      <td>Denver</td>
      <td>CO</td>
      <td>Denver</td>
      <td>Denver</td>
      <td>5869</td>
      <td>331000.0</td>
      <td>364571.67</td>
      <td>0.10</td>
      <td>393687.79</td>
      <td>0.19</td>
      <td>418903.88</td>
      <td>0.27</td>
      <td>441196.73</td>
      <td>0.33</td>
      <td>461298.50</td>
      <td>0.39</td>
      <td>479757.98</td>
      <td>0.45</td>
      <td>496986.49</td>
      <td>0.50</td>
      <td>513292.35</td>
      <td>0.55</td>
      <td>528906.65</td>
      <td>0.60</td>
      <td>544002.58</td>
      <td>0.64</td>
      <td>198</td>
    </tr>
    <tr>
      <th>11028</th>
      <td>11028</td>
      <td>80224</td>
      <td>Denver</td>
      <td>CO</td>
      <td>Denver</td>
      <td>Denver</td>
      <td>5211</td>
      <td>500700.0</td>
      <td>542332.61</td>
      <td>0.08</td>
      <td>581473.44</td>
      <td>0.16</td>
      <td>617373.03</td>
      <td>0.23</td>
      <td>650558.40</td>
      <td>0.30</td>
      <td>681470.90</td>
      <td>0.36</td>
      <td>710480.09</td>
      <td>0.42</td>
      <td>737895.45</td>
      <td>0.47</td>
      <td>763976.15</td>
      <td>0.53</td>
      <td>788939.19</td>
      <td>0.58</td>
      <td>812966.32</td>
      <td>0.62</td>
      <td>253</td>
    </tr>
    <tr>
      <th>11039</th>
      <td>11039</td>
      <td>80236</td>
      <td>Denver</td>
      <td>CO</td>
      <td>Denver</td>
      <td>Denver</td>
      <td>6891</td>
      <td>382500.0</td>
      <td>423570.84</td>
      <td>0.11</td>
      <td>457365.24</td>
      <td>0.20</td>
      <td>485565.91</td>
      <td>0.27</td>
      <td>509788.20</td>
      <td>0.33</td>
      <td>531180.99</td>
      <td>0.39</td>
      <td>550561.37</td>
      <td>0.44</td>
      <td>568510.48</td>
      <td>0.49</td>
      <td>585441.64</td>
      <td>0.53</td>
      <td>601648.81</td>
      <td>0.57</td>
      <td>617341.07</td>
      <td>0.61</td>
      <td>267</td>
    </tr>
    <tr>
      <th>11022</th>
      <td>11022</td>
      <td>80218</td>
      <td>Denver</td>
      <td>CO</td>
      <td>Denver</td>
      <td>Denver</td>
      <td>3055</td>
      <td>773400.0</td>
      <td>857125.55</td>
      <td>0.11</td>
      <td>922077.80</td>
      <td>0.19</td>
      <td>975320.98</td>
      <td>0.26</td>
      <td>1020974.67</td>
      <td>0.32</td>
      <td>1061709.05</td>
      <td>0.37</td>
      <td>1099254.87</td>
      <td>0.42</td>
      <td>1134733.96</td>
      <td>0.47</td>
      <td>1168873.45</td>
      <td>0.51</td>
      <td>1202144.64</td>
      <td>0.55</td>
      <td>1234853.03</td>
      <td>0.60</td>
      <td>292</td>
    </tr>
    <tr>
      <th>11019</th>
      <td>11019</td>
      <td>80212</td>
      <td>Denver</td>
      <td>CO</td>
      <td>Denver</td>
      <td>Denver</td>
      <td>4845</td>
      <td>510000.0</td>
      <td>557325.90</td>
      <td>0.09</td>
      <td>598413.86</td>
      <td>0.17</td>
      <td>633836.33</td>
      <td>0.24</td>
      <td>665228.48</td>
      <td>0.30</td>
      <td>693753.56</td>
      <td>0.36</td>
      <td>720239.05</td>
      <td>0.41</td>
      <td>745273.62</td>
      <td>0.46</td>
      <td>769276.04</td>
      <td>0.51</td>
      <td>792544.20</td>
      <td>0.55</td>
      <td>815290.03</td>
      <td>0.60</td>
      <td>293</td>
    </tr>
    <tr>
      <th>11018</th>
      <td>11018</td>
      <td>80211</td>
      <td>Denver</td>
      <td>CO</td>
      <td>Denver</td>
      <td>Denver</td>
      <td>1388</td>
      <td>546600.0</td>
      <td>594501.05</td>
      <td>0.09</td>
      <td>637074.86</td>
      <td>0.17</td>
      <td>674311.10</td>
      <td>0.23</td>
      <td>707684.86</td>
      <td>0.29</td>
      <td>738263.55</td>
      <td>0.35</td>
      <td>766819.63</td>
      <td>0.40</td>
      <td>793912.05</td>
      <td>0.45</td>
      <td>819945.30</td>
      <td>0.50</td>
      <td>845212.10</td>
      <td>0.55</td>
      <td>869924.26</td>
      <td>0.59</td>
      <td>320</td>
    </tr>
    <tr>
      <th>11042</th>
      <td>11042</td>
      <td>80239</td>
      <td>Denver</td>
      <td>CO</td>
      <td>Denver</td>
      <td>Denver</td>
      <td>3344</td>
      <td>298800.0</td>
      <td>324306.36</td>
      <td>0.09</td>
      <td>346878.56</td>
      <td>0.16</td>
      <td>366328.38</td>
      <td>0.23</td>
      <td>383529.94</td>
      <td>0.28</td>
      <td>399112.64</td>
      <td>0.34</td>
      <td>413529.69</td>
      <td>0.38</td>
      <td>427107.42</td>
      <td>0.43</td>
      <td>440080.80</td>
      <td>0.47</td>
      <td>452619.03</td>
      <td>0.51</td>
      <td>464843.91</td>
      <td>0.56</td>
      <td>392</td>
    </tr>
  </tbody>
</table>
</div>



# Conclusions

Analyzing an entire nation of zipcodes is not going to be as precise as measuring a handful of zip codes at once. However, with a little work we can create some modeling that can, at the very least, drill into the zip codes that are, if not at the very top, at least near the top in terms of percent growth over the coming years.

The thing I did *not* do was to simply take the five top zip codes in the country. There is too much possibility for error with that tactic, and the areas picked would lack the strength of the surrounding areas to mitigate risk. Also, those zipcodes could be isolted flukes.

Instead, in trying to mitigate risk and hone in on the best overall area, and then the best zipcodes within that area, I began at the state level and picked the state with the top average percent gain over ten years. I then drilled down to the county level and picked the top county. From there, I went to the top city, and picked zipcodes from there. This process should insulate the picks from risk as it accounts for strength not only in its own area, but in the surrounding area as well. The strongest zips from the strongest cities from the strongest counties from the strongest state.

In the end, Colorado was the top state, with Denver being the top country, Denver also being the top state, and the five zip codes listed below as the top 5 recommended zip codes for real estate investment nationwide. Since all five were in close proximity to each other, I feel that this reflects great strength in the area and the chances of any one of these zipcodes having come from any sort of anomaly is extremely low.

Oh yes, and for all you sports fans, the top zip encompasses both Mile High Stadium and the Pepsi center.

## Growth Line Comparisons

The top zip code of all was 80204:


```python
forecast_ARIMA(zdf_by_zip[80204], order=(1,1,3), forecast_length=120, 
               diagnostics=False, denver=True)
```


![png](output_150_0.png)


Denver has been experiencing explosive growth in recent years as a magnet for technology companies new and old to gather. 

You don't necessarily need to buy right close to the stadiums, though, and fight all that traffic. These other zip codes are also very strong in their own right, and none of them is far from downtown.

80203, 80219, 80223, 80224.


```python
# Creating subplots for each of the other 4 Denver zip codes in top 5.

fig, axes = plt.subplots(2,2, figsize=(16,10))

forecast_ARIMA_sub(zdf_by_zip[80203], ax=axes[0,0], order=(1,1,3), 
                   forecast_length=120, diagnostics=False)
forecast_ARIMA_sub(zdf_by_zip[80219], ax=axes[0,1], order=(1,1,3), 
                   forecast_length=120, diagnostics=False)
forecast_ARIMA_sub(zdf_by_zip[80223], ax=axes[1,0], order=(1,1,3), 
                   forecast_length=120, diagnostics=False)
forecast_ARIMA_sub(zdf_by_zip[80224], ax=axes[1,1], order=(1,1,3), 
                   forecast_length=120, diagnostics=False)

plt.tight_layout()
plt.show();
```


![png](output_153_0.png)


Plotting them all together for comparison.


```python
fig, axes = plt.subplots(1,1,figsize=(16,10))

ax = axes

order = (1,1,3)

forecast_ARIMA_sub(zdf_by_zip[80203], ax=ax, order=(order), 
                   forecast_length=120, diagnostics=False, intervals=False,
                   lw=2)
forecast_ARIMA_sub(zdf_by_zip[80219], ax=ax, order=(order), 
                   forecast_length=120, diagnostics=False, intervals=False,
                   lw=2)
forecast_ARIMA_sub(zdf_by_zip[80223], ax=ax, order=(order), 
                   forecast_length=120, diagnostics=False, intervals=False,
                   lw=2)
forecast_ARIMA_sub(zdf_by_zip[80224], ax=ax, order=(order), 
                   forecast_length=120, diagnostics=False, intervals=False,
                   lw=2)
forecast_ARIMA_sub(zdf_by_zip[80204], ax=ax, order=(order), 
                   forecast_length=120, diagnostics=False, 
                   intervals=False, denver = True, lw = 4)
forecast_ARIMA_sub(zdf_by_zip['Nationwide'], ax=ax, order=(order), 
                   forecast_length=120, diagnostics=False, intervals=False,
                   lw=5, ls='--', color = 'red')
plt.title('Top Five In Denever', fontsize = 25)

plt.show();
```


![png](output_155_0.png)


Getting 10 year forecast figure for Nationwide model since that information doesn't yet exist outside of a function.


```python
# Same model used for everything else.
model = statsmodels.tsa.arima_model.ARIMA(zdf_by_zip['Nationwide'], 
                                          order=(1,1,3)) 
fitted = model.fit()  
fc, se, conf = fitted.forecast(120, alpha=0.05)
# Calculating the % increase over 10 years and confidence intervals.
nat_roi = round((fc[-1] - fc[0]) / fc[0],2)
nat_conf = (conf[-1,1] - conf[-1,0]) / conf[-1,1]

nat_roi = all_zip_ranked['10_Gain'].mean()
```

## Growth Bar Comparisons

Bar graphs to demonstrate return on investment over the next ten years.


```python
top_list = list(top_zips['Zip'][0:5])
# top_list = [80204, 80203, 80219, 80223, 80224]

# Since I made 80204 oroange before, it will be too confusing to make it 
# orange now, and I don't want to go into making a specific assignment again.
temp = top_list[0]
top_list[0] = top_list[1]
top_list[1] = temp

plt.figure(figsize = (12,8))
plt.title('% Return on Investment Over Ten Years', fontsize = 28)
for azip in top_list:
    gain = top_zips[top_zips['Zip'] == azip]['10_Gain']
    plt.bar(str(azip), gain*100)
plt.bar('Nationwide', nat_roi*100)
plt.ylabel('% Return', fontsize = 20)
plt.xlabel('Zip Code', fontsize = 20)
plt.yticks(ticks = [25, 50, 75, 100], labels = ('25%', '50%', '75%', '100%'))
plt.tick_params(axis="x", labelsize=16)
plt.tick_params(axis="y", labelsize=16)
plt.show()
```


![png](output_160_0.png)


Similarly represnting the dollar amonunt of gain over ten years given a purchase price of the current average. 80203 takes top honors there, but you'll need a fatter pocket book to make that money.


```python
# Since I made 80204 oroange before, it will be too confusing to make it 
# orange now, and I don't want to go into making a specific assignment again.

# Define the national increase
# nat_net = fc[-1] - fc[0]
nat_net = (all_zip_ranked['10_Year'].mean() - all_zip_ranked['2018-04-01'].mean())

# Create the graph
fig, ax = plt.subplots(figsize = (12,8))
plt.title('$ Return on Investment Over Ten Years\n(based on average purchase price for zip)',
          fontsize = 23)
i=0
for azip in top_list:
    gain_start = top_zips[top_zips['Zip'] == azip]['2018-04-01']
    gain_end = top_zips[top_zips['Zip'] == azip]['10_Year']
    dol_gain = top_zips[top_zips['Zip'] == azip]['10_Year'] - top_zips[
        top_zips['Zip'] == azip]['2018-04-01']
    ax.bar(str(azip), dol_gain)
    plt.annotate(str(f'purchase\n${int(gain_start)}'), (i-.28,6000), 
                 color = 'white', fontsize = 12)
    i += 1

plt.bar('Nationwide', nat_net)
plt.yticks(ticks = [100000, 200000, 300000, 400000], labels = ('$100,000',
                                                               '$200,000',
                                                               '$300,000',
                                                               '$400,000'))
plt.ylim(0)
plt.ylabel('$ Return', fontsize = 20)
plt.xlabel('Zip Code', fontsize = 20)
plt.annotate(f'purchase\n${int(fc[0])}', (5-.28,6000), 
             fontsize = 12, color = 'white')
plt.tick_params(axis="x", labelsize=16)
plt.tick_params(axis="y", labelsize=16)
plt.show()
```


![png](output_162_0.png)


## Summary Table

Finally, I'll put together a final table to display 5 and 10 year growth figures for the 5 top zip codes.


```python
# Create lists to concatenate into a DataFrame.
start_list = []
roi_list5 = []
dol_list5 = []
roi_list10 = []
dol_list10 = []

for zipcode in top_list:
    start = all_zip_ranked[all_zip_ranked.Zip == zipcode]['2018-04-01']
    start_list.append(float(start))
    
    # First add the 5 year numbers to the 5 year lists.
    roi_gain5 = all_zip_ranked[all_zip_ranked.Zip == zipcode]['5_Gain']
    end5 = all_zip_ranked[all_zip_ranked.Zip == zipcode]['5_Year']
    roi_list5.append(float(roi_gain5))
    dol_list5.append(int(end5 - start))
    
    # And now the 10 year numbers.
    roi_gain10 = all_zip_ranked[all_zip_ranked.Zip == zipcode]['10_Gain']
    end10 = all_zip_ranked[all_zip_ranked.Zip == zipcode]['10_Year']
    roi_list10.append(float(roi_gain10))
    dol_list10.append(int(end10 - start))
```


```python
# Lists must be converted to DataFrames to concatenate.
start_list = pd.DataFrame([start_list], columns = top_list)
roi_list5 = pd.DataFrame([roi_list5], columns = top_list)
roi_list10 = pd.DataFrame([roi_list10], columns = top_list)
dol_list5 = pd.DataFrame([dol_list5], columns = top_list)
dol_list10 = pd.DataFrame([dol_list10], columns = top_list)
```


```python
summary_table = pd.DataFrame(columns=top_list)
summary_table = pd.concat([summary_table, start_list, roi_list5, dol_list5,
                           roi_list10, dol_list10], axis = 0)
summary_table['Return'] = ['start_list', '5 Yr Return %', '5 Yr Returns $',
                           '10 Yr Return %', '10 Yr Return $']
summary_table = summary_table.set_index('Return')
```

This wraps up the analysis with a succint summary of figures for the top 5 zip codes in my analysis.


```python
summary_table
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>80203</th>
      <th>80204</th>
      <th>80219</th>
      <th>80223</th>
      <th>80224</th>
    </tr>
    <tr>
      <th>Return</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>start_list</th>
      <td>571500.00</td>
      <td>371600.00</td>
      <td>315700.00</td>
      <td>331000.00</td>
      <td>500700.00</td>
    </tr>
    <tr>
      <th>5 Yr Return %</th>
      <td>0.46</td>
      <td>0.53</td>
      <td>0.40</td>
      <td>0.39</td>
      <td>0.36</td>
    </tr>
    <tr>
      <th>5 Yr Returns $</th>
      <td>265053.00</td>
      <td>197919.00</td>
      <td>125494.00</td>
      <td>130298.00</td>
      <td>180770.00</td>
    </tr>
    <tr>
      <th>10 Yr Return %</th>
      <td>0.75</td>
      <td>0.91</td>
      <td>0.65</td>
      <td>0.64</td>
      <td>0.62</td>
    </tr>
    <tr>
      <th>10 Yr Return $</th>
      <td>429755.00</td>
      <td>338964.00</td>
      <td>204298.00</td>
      <td>213002.00</td>
      <td>312266.00</td>
    </tr>
  </tbody>
</table>
</div>




```python

```
