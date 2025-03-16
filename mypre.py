from __future__ import annotations

import argparse
import numpy as np
from datetime import datetime, timedelta
from finrl.meta.preprocessor.yahoodownloader import YahooDownloader
from finrl.meta.preprocessor.preprocessors import FeatureEngineer
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from finrl.agents.stablebaselines3.models import DRLAgent
from finrl.config import INDICATORS, TRAINED_MODEL_DIR
from finrl.config import RESULTS_DIR
import warnings
# 使用 Baostock 替换 YahooDownloader
import baostock as bs
import pandas as pd

warnings.filterwarnings('ignore')


def predict_next_day(
        ticker: str = "AAPL",
        cost_price: float = 0,
        shares: int = 0,
        train_start: str = "2020-01-01",
        use_saved_model: bool = False,  # 添加是否使用保存模型的参数
) :
    """
    预测股票下一个交易日的涨跌

    参数:
        ticker: 股票代码，例如 "AAPL"
        cost_price: 成本价
        shares: 持仓股数
        train_start: 训练数据开始日期
        train_end: 训练数据结束日期
        use_saved_model: 是否使用保存的模型，默认False
    """

    # 1. 获取数据
    today = datetime.now().strftime('%Y-%m-%d')

    # 登录系统
    bs.login()

    try :
        # 转换股票代码格式
        if ticker.endswith('.SS') :
            bs_code = f"sh.{ticker[:-3]}"
        elif ticker.endswith('.SZ') :
            bs_code = f"sz.{ticker[:-3]}"
        else :
            bs_code = ticker

        stock_info = bs.query_stock_basic(code=bs_code)
        if stock_info.error_code == '0' and stock_info.next() :
            stock_name = stock_info.get_row_data()[1]  # 获取股票名称

        # 获取数据
        rs = bs.query_history_k_data_plus(
            code=bs_code,
            fields="date,code,open,high,low,close,volume,amount,turn,pctChg",
            start_date=train_start,
            end_date=today,
            frequency="d",
            adjustflag="3"  # 前复权
        )

        data_list = []
        while (rs.error_code == '0') & rs.next() :
            data_list.append(rs.get_row_data())

        # 转换为 DataFrame
        df = pd.DataFrame(data_list, columns=rs.fields)

        # 处理数据格式以匹配 YahooDownloader 的输出格式
        df['tic'] = ticker
        df['date'] = pd.to_datetime(df['date'])
        numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'amount', 'turn', 'pctChg']
        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric)

        # 确保列名匹配
        df = df.rename(columns={
            'volume' : 'volume',
            'amount' : 'amount'
        })

    finally :
        bs.logout()

    # 2. 特征工程
    fe = FeatureEngineer(
        use_technical_indicator=True,
        tech_indicator_list=INDICATORS,
        use_turbulence=False,
        use_vix=False
    )
    df.to_csv(f"{RESULTS_DIR}/{ticker}df.csv" )
    processed = fe.preprocess_data(df)
    processed = processed.fillna(0)
    processed = processed.replace(np.inf, 0)
    processed.to_csv(f"{RESULTS_DIR}/{ticker}processed.csv")
    # 3. 设置训练环境
    stock_dimension = len(processed.tic.unique())
    state_space = 1 + 2 * stock_dimension + len(INDICATORS) * stock_dimension

    # 保存成本价和持仓信息，但不传入环境
    position_info = {
        "cost_price" : cost_price,
        "holding_shares" : shares,
        "available_cash" : 15000
    }

    env_kwargs = {
        "hmax" : 100,  # 单次最大交易数量限制
        "initial_amount" : position_info["available_cash"],  # 使用实际可用资金
        "num_stock_shares" : [position_info["holding_shares"]],  # 实际持仓数量
        "buy_cost_pct" : [0.00025],  # A股印花税+手续费，买入约万分之2.5
        "sell_cost_pct" : [0.00125],  # A股印花税+手续费，卖出约千分之1.25
        "state_space" : state_space,
        "stock_dim" : 1,  # 因为只交易单个股票，所以设为1
        "tech_indicator_list" : INDICATORS,
        "action_space" : 1,  # 因为只交易单个股票，所以设为1
        "reward_scaling" : 1e-4,
        "print_verbosity" : 1
    }

    # 4. 训练或加载模型
    # 根据训练截止日期筛选训练数据
    # train_data = processed[processed.date <= train_end]

    # 创建股票交易训练环境
    # 使用之前定义的env_kwargs参数字典初始化环境
    # env_kwargs包含了最大持仓数量、初始资金、交易成本等关键参数
    env_train = StockTradingEnv(df=processed, **env_kwargs)

    # 初始化DRL智能体
    # DRLAgent封装了强化学习算法的训练和预测功能
    # 将训练环境传入智能体用于后续训练
    agent = DRLAgent(env=env_train)

    model_name = "sac"
    train_timesteps = 100000
    try :
        if use_saved_model :
            print("正在加载保存的模型...")
            try :
                # 根据选择的模型类型加载对应模型
                if model_name == 'a2c' :
                    trained_model = agent.get_model("a2c").load(f"{RESULTS_DIR}/a2c/a2c_model_{ticker}")
                elif model_name == 'ppo' :
                    trained_model = agent.get_model("ppo").load(f"{RESULTS_DIR}/ppo/ppo_model_{ticker}")
                elif model_name == 'ddpg' :
                    trained_model = agent.get_model("ddpg").load(f"{RESULTS_DIR}/ddpg/ddpg_model_{ticker}")
                elif model_name == 'td3' :
                    trained_model = agent.get_model("td3").load(f"{RESULTS_DIR}/td3/td3_model_{ticker}")
                elif model_name == 'sac' :
                    trained_model = agent.get_model("sac").load(TRAINED_MODEL_DIR + "/agent_sac")
                print(f"成功加载保存的{model_name.upper()}模型")
            except Exception as e :
                print(f"加载模型失败：{str(e)}")
                print("将重新训练模型...")
                use_saved_model = False

        if not use_saved_model :
            print(f"开始训练新{model_name.upper()}模型...")
            # 为不同模型设置参数
            model_params = {
                'a2c' : {
                    "n_steps" : 5,
                    "ent_coef" : 0.005,
                    "learning_rate" : 0.0007
                },
                'ppo' : {
                    "n_steps" : 2048,
                    "ent_coef" : 0.01,
                    "learning_rate" : 0.0003,
                    "batch_size" : 64
                },
                'ddpg' : {
                    "buffer_size" : 100000,
                    "learning_rate" : 0.0005,
                    "batch_size" : 64
                },
                'td3' : {
                    "buffer_size" : 100000,
                    "learning_rate" : 0.0005,
                    "batch_size" : 64
                },
                'sac' : {
                    "buffer_size" : 100000,
                    "batch_size" : 64,
                    "ent_coef" : "auto",
                    "learning_rate" : 0.0001,
                    "learning_starts" : 100,

                }
            }

            # 获取对应模型的参数
            current_model_params = model_params.get(model_name, {})

            # 创建模型
            model = agent.get_model(
                model_name,
                model_kwargs=current_model_params,
                verbose=1
            )

            # 训练模型
            trained_model = agent.train_model(
                model=model,
                tb_log_name=model_name,
                total_timesteps=train_timesteps
            )

            # 保存训练好的模型
            print("保存模型...")
            save_path = f"{RESULTS_DIR}/{model_name}/{model_name}_model_{ticker}"
            trained_model.save(save_path)
            print(f"模型保存成功: {save_path}")

    except Exception as e :
        print(f"模型训练/加载过程中发生错误：{str(e)}")
        return None

    # 5. 获取最新状态并预测
    latest_data = processed[processed.date == processed.date.max()]
    latest_data.to_csv(f"{RESULTS_DIR}/{ticker}latest_data.csv")
    # 构造状态向量
    state_space = 1 + 2 * stock_dimension + len(INDICATORS) * stock_dimension
    latest_state = np.zeros(state_space)

    try :
        # 填充状态向量
        latest_state[0] = env_kwargs['initial_amount']

        tic_data = latest_data.iloc[0]
        latest_state[1] = tic_data['close']
        latest_state[2] = 0  # 初始持仓量

        # 技术指标
        for j, indicator in enumerate(INDICATORS) :
            latest_state[3 + j] = tic_data[indicator]

        # reshape状态向量
        latest_state = latest_state.reshape(1, -1)
        print("状态向量：")
        print(latest_state)
        # 预测
        action, _states = trained_model.predict(latest_state)
        # 保存action到本地
        action_file = f"{RESULTS_DIR}/{ticker}action.txt"
        with open(action_file, 'w') as f :
            f.write(str(action[0]))
        # 帮我实现打印最近5天的最新交易数据
        print("\n=== 最近5天的最新交易数据 ===")
        print(df.tail(5))
        # 解释预测结果
        print("\n=== 股票预测结果 ===")
        print(f"股票: {ticker}")
        print(f"股票名称: {stock_name}")
        print(f"日期: {latest_data.date.iloc[0]}")
        print(f"当前价格: {latest_data.close.iloc[0]:.2f}")

        signal = "看涨" if action[0] > 0.3 else ("看跌" if action[0] < -0.3 else "持平")
        print(f"预测信号: {signal}")

        signal_type = "强烈看涨" if action[0] > 0.7 else (
            "中度看涨" if action[0] > 0.5 else (
                "轻度看涨" if action[0] > 0.3 else (
                    "强烈看跌" if action[0] < -0.7 else (
                        "中度看跌" if action[0] < -0.5 else (
                            "轻度看跌" if action[0] < -0.3 else "观望")))))
        print(f"信号类型: {signal_type}")
        print(
            f"信号强度: {abs(float(action[0])):.4f} (信号强度表示预测的置信度,范围0-1,数值越大表示AI模型对预测结果越有信心,但不代表预测一定准确)")

        print("\n=== 技术指标分析 ===")
        print("以下是各项技术指标的最新值及其含义:")
        for indicator in INDICATORS :
            value = float(latest_data[indicator].iloc[0])
            print(f"\n{indicator}: {value:.4f}")

            # 为不同指标添加详细解释
            if indicator == 'macd' :
                print("MACD指标(移动平均收敛/发散)详细说明:")
                print("- MACD是最流行的趋势跟踪指标之一")
                print("- 正值表示上升趋势，负值表示下降趋势")
                print("- 数值的绝对值越大表示趋势越强")
                print("- MACD金叉(DIFF上穿DEA)是买入信号")
                print("- MACD死叉(DIFF下穿DEA)是卖出信号")
                print("- MACD柱状图可以直观反映趋势强弱变化")
            elif indicator == 'boll_ub' :
                print("布林带上轨(Bollinger Bands Upper Band)详细说明:")
                print("- 布林带由中轨(20日移动平均线)和上下轨组成")
                print("- 上轨=中轨+2倍标准差")
                print("- 股价突破上轨表示超买，上涨动能较强")
                print("- 此时应警惕可能出现的回调风险")
                print("- 股价长期在上轨运行表明强势特征明显")
            elif indicator == 'boll_lb' :
                print("布林带下轨(Bollinger Bands Lower Band)详细说明:")
                print("- 下轨=中轨-2倍标准差")
                print("- 股价突破下轨表示超卖，下跌动能较强")
                print("- 此时可能出现技术性反弹机会")
                print("- 股价长期在下轨运行表明弱势特征明显")
                print("- 布林带收窄表示即将突破，扩张表示趋势加强")
            elif indicator == 'rsi_30' :
                print("RSI(30日)相对强弱指标详细说明:")
                print("- RSI用于衡量价格动量，取值范围0-100")
                print("- RSI>80表示超买区，股价可能面临回调压力")
                print("- RSI<20表示超卖区，股价可能出现反弹机会")
                print("- RSI上穿50分界线显示上升动能加强")
                print("- RSI下穿50分界线显示下跌动能加强")
                print("- RSI背离可以提前预警价格走势转折")
            elif indicator == 'cci_30' :
                print("CCI(30日)顺势指标详细说明:")
                print("- CCI用于判断股价是否超买或超卖")
                print("- CCI>100进入超买区，>200极度超买")
                print("- CCI<-100进入超卖区，<-200极度超卖")
                print("- CCI上穿/下穿±100常被用作买入/卖出信号")
                print("- CCI也可以通过背离形态预测价格走势转折")
                print("- CCI的波动范围越大，市场活跃度越高")
            elif indicator == 'dx_30' :
                print("DX(30日)动向指标详细说明:")
                print("- DX用于衡量价格趋势的强度")
                print("- 取值范围0-100，数值越大趋势越明显")
                print("- 一般认为DX>25表示存在明显趋势")
                print("- DX>40表示强趋势，>60表示极强趋势")
                print("- DX上升表示趋势增强，下降表示趋势减弱")
                print("- 可与ADX、ADXR等指标配合使用增强可靠性")
            elif indicator == 'close_30_sma' :
                print("30日简单移动平均线(SMA)详细说明:")
                print("- 反映30个交易日内股价的平均水平")
                print("- 股价上穿30日均线是短期买入信号")
                print("- 股价下穿30日均线是短期卖出信号")
                print("- 均线向上倾斜表示上升趋势，向下倾斜表示下降趋势")
                print("- 股价与均线的距离可以衡量股价的超买超卖程度")
                print("- 常与其他均线(如60日)结合判断趋势拐点")
            elif indicator == 'close_60_sma' :
                print("60日简单移动平均线(SMA)详细说明:")
                print("- 反映60个交易日内的中期价格走势")
                print("- 是判断中期趋势的重要参考线")
                print("- 30日均线上穿60日均线形成金叉，看涨信号")
                print("- 30日均线下穿60日均线形成死叉，看跌信号")
                print("- 股价站稳60日均线上方，表明中期趋势向好")
                print("- 股价跌破60日均线下方，表明中期趋势转弱")
                print("- 60日均线的方向对判断大趋势很有帮助")
    except Exception as e :
        print(f"预测过程中发生错误：{str(e)}")
        return None


def parse_args() :
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='股票预测程序')

    parser.add_argument('--symbol',
                        type=str,
                        default='601928',
                        help='股票代码')

    parser.add_argument('--cost_price', type=float, default=0,
                        help='成本价')

    parser.add_argument('--shares', type=int, default=26100,
                        help='持仓股数')

    parser.add_argument('--need_train',
                        type=lambda x : x.lower() == 'true',  # 将字符串转换为布尔值
                        default=True,
                        help='是否需要训练 (true/false)')

    parser.add_argument('--start_date',
                        type=str,
                        default=(datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d'),
                        help='开始日期 (YYYY-MM-DD)')

    parser.add_argument('--end_date',
                        type=str,
                        default=datetime.now().strftime('%Y-%m-%d'),
                        help='结束日期 (YYYY-MM-DD)')

    return parser.parse_args()


if __name__ == "__main__" :
    # 使用示例
    args = parse_args()

    # 添加.SS后缀（如果是上证股票）
    if args.symbol.startswith('6') :
        ticker = f"{args.symbol}.SS"
    # 添加.SZ后缀（如果是深证股票）
    elif args.symbol.startswith('0') or args.symbol.startswith('3') :
        ticker = f"{args.symbol}.SZ"
    else :
        ticker = args.symbol

    print(f"\n=== 股票预测配置 ===")
    print(f"股票代码: {ticker}")
    print(f"成本价: {args.cost_price}")
    print(f"持仓股数: {args.shares}")
    print(f"开始日期: {args.start_date}")
    print(f"是否训练: {args.need_train}")
    print("==================\n")
    # 使用新训练的模型
    # prediction_new = predict_next_day(
    #     ticker="AAPL",
    #     train_start="2023-01-01",
    #     train_end="2023-12-31",
    #     use_saved_model=False  # 不使用保存的模型，重新训练
    # )

    # 使用保存的模型
    prediction_saved = predict_next_day(
        ticker=ticker,
        cost_price=args.cost_price,
        shares=args.shares,
        train_start=args.start_date,
        use_saved_model=not args.need_train  # need_train为True时不使用保存的模型
    )