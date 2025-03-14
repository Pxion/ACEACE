import yfinance as yf
from datetime import datetime, timedelta

# 获取今天的日期
today = datetime.today()
today_str = today.strftime('%Y-%m-%d')

# 获取昨天的日期
yesterday = today - timedelta(days=1)
yesterday_str = yesterday.strftime('%Y-%m-%d')

def get_now_data(symbol, yesterday_str, today_str=None):
    # 下载数据
    dji_data = yf.download(
        symbol,
        start=yesterday_str,
        end=today_str,
        progress=True,     # 显示下载进度条
        auto_adjust=True   # 使用调整后收盘价（可选）
    )
    # 保存为 CSV 文件
    dji_data.to_csv(f"{symbol}TodayData.csv")


symbol="000001.SS"
get_now_data(symbol, yesterday_str)




# def get_data(symbol, start_date, end_date):
#     # 下载数据
#     dji_data = yf.download(
#         symbol,
#         start=start_date,
#         end=end_date,
#         progress=True,     # 显示下载进度条
#         auto_adjust=True   # 使用调整后收盘价（可选）
#     )
#     # 保存为 CSV 文件
#     dji_data.to_csv("testData.csv")
#     df = pd.read_csv("testData.csv")
#     df.drop(0,inplace=True)
#     # dji_data.drop(['Price'], axis=1, inplace=True)
#     df.rename(columns={'Price' : 'Date'}, inplace=True)
#     df.drop(1,inplace=True)
#
#     # 保存为 CSV 文件
#     df.to_csv(f"{symbol}_new.csv")
#
# # 设置参数
# symbol = "000001.SS"        # 道琼斯指数代码
# start_date = "2012-01-01"
# end_date = "2025-3-11"
#
# get_data(symbol, start_date, end_date)
# df=pd.read_csv("testData.csv")
#
# # debug
# df.drop(0, inplace=True)
