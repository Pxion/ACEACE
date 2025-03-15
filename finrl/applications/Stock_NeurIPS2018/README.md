我们展示了一个在算法交易中应用强化学习（RL）的工作流程，这是对 [NeurIPS 2018 论文](https://arxiv.org/abs/1811.07522) 中过程的重现和改进。

# 使用方法

## 第一步：数据

首先，运行笔记本文件：*Stock_NeurIPS2018_1_Data.ipynb*。

它下载并预处理股票的OHLCV（开盘价、最高价、最低价、收盘价、交易量）数据。

它生成两个CSV文件：*train.csv* 和 *trade.csv*。您可以查看提供的两个示例文件。

## 第二步：训练交易代理

其次，运行笔记本文件：*Stock_NeurIPS2018_2_Train.ipynb*。

它展示了如何将数据处理成OpenAI Gym风格的环境，然后训练一个深度强化学习（DRL）代理。

它会生成一个训练好的RL模型的 .zip 文件。这里，我们也提供了一个训练好的A2C模型的 .zip 文件。

## 第三步：回测

最后，运行笔记本文件：*Stock_NeurIPS2018_3_Backtest.ipynb*。

它对训练好的代理进行回测，并与两个基准进行比较：均值-方差优化（Mean-Variance Optimization）和市场DJIA指数。

最终，它将绘制出回测过程中投资组合价值的图表。