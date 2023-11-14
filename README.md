# Stock Market Analysis
##### Analyzing the Stock Market with Time Series Models and Classification Models

By: David Hartsman

##### Instructions for navigating the repo:

[1_Index_Profitability_Classification](https://github.com/dvdhartsman/Stock_Market_Analysis/blob/main/1_Index_Profitability_Classification.ipynb) - Modeling and observations for the Dow Jones, S&P 500, Nasdaq, and Russell 2000
[2a_SPDR_ETF_modeling](https://github.com/dvdhartsman/Stock_Market_Analysis/blob/main/2a_SPDR_ETF_modeling.ipynb) - Overview of sector SPDRs and is primarily for model iterations
[2b_SPDR_Eval](https://github.com/dvdhartsman/Stock_Market_Analysis/blob/main/2b_SPDR_Eval.ipynb) - Evaluation and summary of the SPDR modeling
[3_S&P_Time_Series](https://github.com/dvdhartsman/Stock_Market_Analysis/blob/main/3_S%26P_Time_Series.ipynb) - ARIMA modeling
[4_LSTM_Neural_Networks](https://github.com/dvdhartsman/Stock_Market_Analysis/blob/main/4_LSTM_Neural_Networks.ipynb) - LSTM neural network

##### Disclaimer: This notebook should not be considered any kind of financial advice. It exists only for the purposes of practicing modeling and making predictions

![Prices Being Charted in Real Time](./Files/stock_header.jpg)
### Overview
In this project, I took a multi-pronged approach to predicting performance of the stock market. I analyzed price movement at the index level and sector level. I first developed predictive models using a qualitative target: whether or not an asset appreciated over a one, three, six, or twelve month period. My second approach to predicting the stock market utilized time series modeling on the index level. 


### Data Understanding
For all of the project avenues, a similar process was used to [gather and prepare](https://github.com/dvdhartsman/Stock_Market_Analysis/blob/main/Data_Prep/Data_Downloads_and_Processing.ipynb) the data. I downloaded data from yfinance, the Yahoo! Finance api. The data was initially fairly straightforward, containing features for the ticket, adjusted close, close, high, low, open, and volume. I supplemented that data by downloading data from Federal Reserve Economic Data by using pandas_datareader, adding historical data for interest rates and GDP. I also added several technical indicators to the data by using the pandas_ta library. That library is filled with functions that calculate many frequently used technical indicators such as ATR, MACD, RSI, etc. I then added to my data the 5 Fama-French indicators maintained and shared by Kenneth French from Dartmouth University, an accomplished and respected professor. The final component of data preparation involved me finding dates in the data that corresponded to the time horizons I previously described. I then created feature columns for each time horizon: one that was binary indicating whether the asset went up in value or not, and another set of features that displayed the percent change in the asset's value in the future. The latter features were created for inferential purposes. 

### Evaluation
I modeled [major index](https://github.com/dvdhartsman/Stock_Market_Analysis/blob/main/1_Index_Profitability_Classification.ipynb) returns, as well as returns for [SPDR ETFs](https://github.com/dvdhartsman/Stock_Market_Analysis/blob/main/2a_SPDR_ETF_modeling.ipynb) that represent the 11 sectors of the stock market. The targets of the models were binary: whether the price went up or down over different periods of time. The predictions of models created using binary targets were rather impressive. Precision was the target metric, as the practical application of these models would be to make correct "buy" recommendations, and not to have false positives. The best-performing models were consistently tree based models. Here you can see just one of the hundred estimators in one such model.  
![This ExtraTreesClassifier Estimator is quite complicated...](./Files/extra_tree.jpg)


One of the most attractive features of the models I produced for different sectors was their ability to minimize downside risk over longer time horizons. Downside risk was more difficult to manage for one and three month targets. There are many box plots in the [SPDR evaluation notebook](https://github.com/dvdhartsman/Stock_Market_Analysis/blob/main/2b_SPDR_Eval.ipynb) that demonstrate the models' ability to contain downside risk. 

I also utilized two different methods of uni-variate time series modeling. I first attempted to create ARIMA models using only the past closing prices of the S&P 500. ARIMA stands for "auto-regressive integrated moving average". The work with ARIMA is found in [this notebook](https://github.com/dvdhartsman/Stock_Market_Analysis/blob/main/3_S%26P_Time_Series.ipynb).

The final approach to modeling that I took was using a Long Short-Term Memory neural network to predict stock prices. This is another form of uni-variate modeling. The only features I provided to the neural network were the past 15 days' closing prices. This model is particularly adept at storing memories from previous data and determining how much relevance that data has to the future predictions of the model.  

[LSTM Predictions vs True Values](./Files/LSTM_preds.jpg)


### Conclusion

The binary target models produced very unique modeling results. For example, when modeling 12-month Tech, I had two different models get 0% precision scores on unseen test data! That was one of two occasions that I failed to improve on the dummy model's precision score, the other being 3-month Communication Services, where the data were extremely sparse and imbalanced.

Many of the models' ROC curves and AUC scores were very different from other modeling projects I have done in the past. This seems to indicate how intrinsically difficult it is to predict the vicissitudes of the stock market. 

[CRAZY Curves](.Files/drunk_Curve.jpg) 

As I said earlier, tree based models were the most frequent representatives for each subset of the market that I modeled, though Logistic Regression models were occassionally the best candidates depending on use case. Generally speaking, the longer the time horizon, the more reliably the models could minimize risk. That does not mean that they produced the most gains, however.

With precision as my target metric, the positive rate of many of these models was very low. In order to focus on "useful" models, I had to inspect the positive prediction rates along with the precision scores. Also, my efforts to tune models to precision often times had very surprising outcomes, causing these models to behave strangely. Suffice it to say, that the models did not always provide signals worth acting upon. Still, the mean performance when they did predict "buys" was generally quite strong.

For Uni-Variate Time Series Modeling, the ARIMA model was the least impressive or instructive part of this modeling project. The model did not seem well suited to predicting stock prices. The LSTM neural network, on the other hand, managed remarkably accurate predictions. There did tend to be a slight lag, and under-representation of the full measure of volatile moves, however the contours of the chart were quite well captured over a surprisingly long period of time. 

These models as a whole, can provide a supplementary source of information, but none of them should be trusted to outperform on their own. The market is extremely difficult to predict, and will move irrationally. Please be judicious in your decisions, and enjoy surveying the results of my research.

