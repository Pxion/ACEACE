import yfinance as yf
import pandas as pd
def get_data(symbol, start_date, end_date):
    # 下载数据
    dji_data = yf.download(
        symbol,
        start=start_date,
        end=end_date,
        progress=True,     # 显示下载进度条
        auto_adjust=True   # 使用调整后收盘价（可选）
    )
    # 保存为 CSV 文件
    dji_data.to_csv("testData.csv")
    df = pd.read_csv("testData.csv")
    df.drop(0,inplace=True)
    # dji_data.drop(['Price'], axis=1, inplace=True)
    df.rename(columns={'Price' : 'Date'}, inplace=True)
    df.drop(1,inplace=True)

    # 保存为 CSV 文件
    df.to_csv(f"{symbol}_new.csv")

# 设置参数
symbol = "000001.SS"        # 道琼斯指数代码
start_date = "2012-01-01"
end_date = "2025-3-11"

get_data(symbol, start_date, end_date)
# df=pd.read_csv("testData.csv")
#
# # debug
# df.drop(0, inplace=True)
