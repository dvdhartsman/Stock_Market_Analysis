# Stock_Market_Analysis
##### Analyzing the Stock Market with Time Series Models and Classification Models

By: David Hartsman

![Prices Being Charted in Real Time](./Files/stock_header.jpg)
### Overview
In this project, I took a multi-pronged approach to predicting performance of the stock market. I analyzed price movement at the index level and sector level. I first developed predictive models using a qualitative target: whether or not an asset appreciated over a one, three, six, or twelve month period. My second approach to predicting the stock market utilized time series modeling on the index level. 


### Data Understanding
For all of the project avenues, I downloaded data from yfinance, the Yahoo! Finance api. The data was initially fairly straightforward, containing features for the ticket, adjusted close, close, high, low, open, and volume. I supplemented that data by downloading data from Federal Reserve Economic Data by using pandas_datareader, adding historical data for interest rates and GDP. I also added several technical indicators to the data by using the pandas_ta library. That library is filled with functions that calculate many frequently used technical indicators such as ATR, MACD, RSI, etc. I then added to my data the 5 Fama-French indicators maintained and shared by Kenneth French from Dartmouth University, an accomplished and respected professor. The final component of data preparation involved me finding dates in the data that corresponded to the time horizons I previously described. I then created feature columns for each time horizon: one that was binary indicating whether the asset went up in value or not, and another set of features that displayed the percent change in the asset's value in the future. The latter features were created for inferential purposes.

### Evaluation
The predictions of models created using binary targets were extremely impressive. The best-performing models were consistently tree based models, and the ExtraTreesClassifier in particular frequently produced the most accurate test scores. Here you can see just one of the hundred estimators in one such model. Seeing it helps to understand how its many layers of depth can help generate such accurate predictions. 
![This ExtraTreesClassifier Estimator is quite complicated...](./Files/Energy_Grid_12m_tree.jpg)



### Conclusion

