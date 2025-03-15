import yfinance as yf


def get_stock_info(stock_name) :
    """
    获取指定股票或指数的最新股价和当前涨跌幅。

    参数:
      stock_name: 股票名称或指数名称，例如 "上证指数"、"道琼斯指数" 或直接使用股票代码，如 "000001.SS"、"^DJI"。

    返回:
      一个元组 (price, change_percent)，其中 price 是最新股价，
      change_percent 是当前涨跌幅（百分比）。
    """
    # 定义中文名称与对应股票代码的映射
    mapping = {
        "上证指数" : "000001.SS",  # 上证指数对应的雅虎财经代码
        "道琼斯指数" : "^DJI",  # 道琼斯指数对应的雅虎财经代码
    }

    # 如果 stock_name 在映射中，则使用对应代码，否则直接使用传入的 stock_name
    symbol = mapping.get(stock_name, stock_name)

    # 利用 yfinance 获取数据
    ticker = yf.Ticker(symbol)
    info = ticker.info

    # 从 info 中获取最新价格和涨跌幅
    price = info.get('regularMarketPrice')
    change_percent = info.get('regularMarketChangePercent')

    return price, change_percent


# 示例使用
# def get_new_stock_info():
#     # 查询上证指数
#     price, change = get_stock_info("上证指数")
#     print("上证指数 最新价格: ", price, "当前涨跌幅: ", change, "%")
#
#     # 查询道琼斯指数
#     price, change = get_stock_info("道琼斯指数")
#     print("道琼斯指数 最新价格: ", price, "当前涨跌幅: ", change, "%")
