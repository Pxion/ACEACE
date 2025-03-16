import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from stable_baselines3 import A2C, DDPG, PPO, SAC, TD3
import sys
import os
from finrl.agents.stablebaselines3.models import DRLAgent
from finrl.config import INDICATORS, TRAINED_MODEL_DIR
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from finrl.meta.preprocessor.yahoodownloader import YahooDownloader
from pypfopt.efficient_frontier import EfficientFrontier


def backtest_stock_trading(
    trade, train, trained_model_dir="trained_models", models_to_use=None,
    initial_cash=1000000, buy_cost_pct=0.001, sell_cost_pct=0.001,
    start_date='2020-07-01', end_date='2021-10-29', tech_indicator_list=None
):


    # 筛选指定日期内的数据
    # 复制 date 列并转换为 datetime 格式
    train['date_copy'] = pd.to_datetime(train['date'])
    trade['date_copy'] = pd.to_datetime(trade['date'])

    # 筛选指定日期内的数据
    # train = train[(train["date_copy"].dt.date >= start_date) & (train["date_copy"].dt.date <= end_date)]
    trade = trade[(trade["date_copy"].dt.date >= start_date) & (trade["date_copy"].dt.date <= end_date)]

    # 删除复制的列
    train.drop(columns=['date_copy'], inplace=True)
    trade.drop(columns=['date_copy'], inplace=True)
    train = train.set_index(train.columns[0])
    train.index.names = ['']
    trade = trade.set_index(trade.columns[0])
    trade.index.names = ['']
    # 查看数据形状
    print("Train data shape:", train.shape)
    print("Trade data shape:", trade.shape)


    # Set up models
    if models_to_use is None:
        models_to_use = ['a2c', 'ddpg', 'ppo', 'td3', 'sac']
    # 模型优化效果有限，强制使用cpu加载模型以提高性能
    trained_models = {}
    for model_name in models_to_use:
        if model_name == 'a2c':
            trained_models['a2c'] = A2C.load(f"{trained_model_dir}/agent_a2c", device='cpu')
        elif model_name == 'ddpg':
            trained_models['ddpg'] = DDPG.load(f"{trained_model_dir}/agent_ddpg", device='cpu')
        elif model_name == 'ppo':
            trained_models['ppo'] = PPO.load(f"{trained_model_dir}/agent_ppo", device='cpu')
        elif model_name == 'td3':
            trained_models['td3'] = TD3.load(f"{trained_model_dir}/agent_td3", device='cpu')
        elif model_name == 'sac':
            trained_models['sac'] = SAC.load(f"{trained_model_dir}/agent_sac", device='cpu')

    # Prepare stock trading environment
    stock_dimension = len(trade.tic.unique())
    state_space = 1 + 2 * stock_dimension + len(INDICATORS) * stock_dimension

    num_stock_shares = [0] * stock_dimension
    buy_cost_list = sell_cost_list = [buy_cost_pct] * stock_dimension

    env_kwargs = {
        "hmax": 100,
        "initial_amount": initial_cash,
        "num_stock_shares": num_stock_shares,
        "buy_cost_pct": buy_cost_list,
        "sell_cost_pct": sell_cost_list,
        "state_space": state_space,
        "stock_dim": stock_dimension,
         # "tech_indicator_list": tech_indicator_list or INDICATORS,
        "tech_indicator_list": INDICATORS,
        "action_space": stock_dimension,
        "reward_scaling": 1e-4
    }

    # Create environment
    e_trade_gym = StockTradingEnv(df=trade, turbulence_threshold=70, risk_indicator_col='vix', **env_kwargs)

    # Run DRL agents
    results = {}
    for model_name, model in trained_models.items():
        df_account_value, df_actions = DRLAgent.DRL_prediction(model=model, environment=e_trade_gym)
        if df_account_value is not None:
            df_account_value = df_account_value.set_index(df_account_value.columns[0])
            results[model_name] = df_account_value['account_value']

    # Mean Variance Optimization (MVO)
    def process_df_for_mvo(df):
        return df.pivot(index="date", columns="tic", values="close")

    def StockReturnsComputing(StockPrice, Rows, Columns):
        StockReturn = np.zeros([Rows - 1, Columns])
        for j in range(Columns):
            for i in range(Rows - 1):
                StockReturn[i, j] = ((StockPrice[i + 1, j] - StockPrice[i, j]) / StockPrice[i, j]) * 100
        return StockReturn

    StockData = process_df_for_mvo(train)
    TradeData = process_df_for_mvo(trade)

    arStockPrices = np.asarray(StockData)
    [Rows, Cols] = arStockPrices.shape
    arReturns = StockReturnsComputing(arStockPrices, Rows, Cols)

    meanReturns = np.mean(arReturns, axis=0)
    covReturns = np.cov(arReturns, rowvar=False)

    ef_mean = EfficientFrontier(meanReturns, covReturns, weight_bounds=(0, 0.5))
    raw_weights_mean = ef_mean.max_sharpe()
    cleaned_weights_mean = ef_mean.clean_weights()

    mvo_weights = np.array([1000000 * cleaned_weights_mean[i] for i in range(len(cleaned_weights_mean))])
    LastPrice = np.array([1 / p for p in StockData.tail(1).to_numpy()[0]])
    Initial_Portfolio = np.multiply(mvo_weights, LastPrice)
    Portfolio_Assets = TradeData @ Initial_Portfolio
    MVO_result = pd.DataFrame(Portfolio_Assets, columns=["Mean Var"])

    # DJIA index
    df_dji = YahooDownloader(
        start_date=start_date, end_date=end_date, ticker_list=["dji"]
    ).fetch_data()
    df_dji = df_dji[["date", "close"]]
    fst_day = df_dji["close"][0]
    dji = pd.merge(
        df_dji["date"],
        df_dji["close"].div(fst_day).mul(1000000),
        how="outer",
        left_index=True,
        right_index=True,
    ).set_index("date")

    # Compile results
    result = pd.DataFrame(
        {model_name: results.get(model_name) for model_name in models_to_use}
    )

    # Add MVO and DJIA
    result['mvo'] = MVO_result["Mean Var"]
    result['dji'] = dji["close"]

    # Plot results
    # fig, ax = plt.subplots(figsize=(15, 5))
    # result.plot(ax=ax)
    # ax.set_title("Backtest Results")
    # fig.savefig('backtest_results.png')
    return result


# Example usage:
# trade_data_path = 'trade_data.csv'
# train_data_path = 'train_data.csv'
# trained_model_dir = "D:/互联网+，大创/大创--多智能体市场分析/FinRL-master/examples/trained_models"
# backtest_result = backtest_stock_trading(
#     trade_data_path, train_data_path, trained_model_dir,
#     models_to_use=['a2c', 'sac', 'ppo'], initial_cash=1000000
# )
# print(backtest_result)
# backtest_stock_trading('model_data/trade_data.csv','model_data/train_data.csv')