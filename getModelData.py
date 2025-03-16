import pandas as pd
import numpy as np
import datetime
import yfinance as yf
from finrl.meta.preprocessor.yahoodownloader import YahooDownloader
from finrl.meta.preprocessor.preprocessors import FeatureEngineer, data_split
from finrl import config_tickers
from finrl.config import INDICATORS
import itertools

def get_model_data(ticker_list = config_tickers.DOW_30_TICKER,
        TRAIN_START_DATE = '2009-01-01',
    TRAIN_END_DATE = '2020-07-01',
    TRADE_START_DATE = '2020-07-01',
    TRADE_END_DATE = '2021-10-29'):


    df_raw = YahooDownloader(start_date = TRAIN_START_DATE,
                         end_date = TRADE_END_DATE,
                         ticker_list = config_tickers.DOW_30_TICKER).fetch_data()

    df_raw.head()

    # # Part 3: Preprocess Data
    # We need to check for missing data and do feature engineering to convert the data point into a state.
    # * **Adding technical indicators**. In practical trading, various information needs to be taken into account, such as historical prices, current holding shares, technical indicators, etc. Here, we demonstrate two trend-following technical indicators: MACD and RSI.
    # * **Adding turbulence index**. Risk-aversion reflects whether an investor prefers to protect the capital. It also influences one's trading strategy when facing different market volatility level. To control the risk in a worst-case scenario, such as financial crisis of 2007–2008, FinRL employs the turbulence index that measures extreme fluctuation of asset price.

    # Hear let's take **MACD** as an example. Moving average convergence/divergence (MACD) is one of the most commonly used indicator showing bull and bear market. Its calculation is based on EMA (Exponential Moving Average indicator, measuring trend direction over a period of time.)

    fe = FeatureEngineer(use_technical_indicator=True,
                         tech_indicator_list = INDICATORS,
                         use_vix=True,
                         use_turbulence=True,
                         user_defined_feature = False)

    processed = fe.preprocess_data(df_raw)

    list_ticker = processed["tic"].unique().tolist()
    list_date = list(pd.date_range(processed['date'].min(),processed['date'].max()).astype(str))
    combination = list(itertools.product(list_date,list_ticker))

    processed_full = pd.DataFrame(combination,columns=["date","tic"]).merge(processed,on=["date","tic"],how="left")
    processed_full = processed_full[processed_full['date'].isin(processed['date'])]
    processed_full = processed_full.sort_values(['date','tic'])

    processed_full = processed_full.fillna(0)

    processed_full.head()

    # # Part 4: Save the Data

    # ### Split the data for training and trading

    train = data_split(processed_full, TRAIN_START_DATE,TRAIN_END_DATE)
    trade = data_split(processed_full, TRADE_START_DATE,TRADE_END_DATE)
    print(len(train))
    print(len(trade))

    trade.reset_index(drop=True, inplace=True)

    train.reset_index(drop=True, inplace=True)

    # ### Save data to csv file

    # For Colab users, you can open the virtual directory in colab and manually download the files.
    #
    # For users running on your local environment, the csv files should be at the same directory of this notebook.

    train.to_csv('model_data/train_data.csv')
    trade.to_csv('model_data/trade_data.csv')