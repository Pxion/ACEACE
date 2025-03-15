# 投资组合优化代理

该目录包含投资组合优化代理中常用架构和算法。

为了实例化模型，需要有一个 [PortfolioOptimizationEnv](/finrl/meta/env_portfolio_optimization/) 的实例。在下面的示例中，我们使用 `DRLAgent` 类实例化一个策略梯度（"pg"）模型。通过字典 `model_kwargs`，我们可以设置 `PolicyGradient` 类的参数，并且通过字典 `policy_kwargs`，可以更改所选架构的参数。

```python
from finrl.agents.portfolio_optimization.models import DRLAgent
from finrl.agents.portfolio_optimization.architectures import EIIE

# 设置 PolicyGradient 算法参数
model_kwargs = {
    "lr": 0.01,  # 学习率
    "policy": EIIE,  # 使用 EIIE 架构
}

# 设置 EIIE 架构参数
policy_kwargs = {
    "k_size": 4  # 卷积核大小
}

model = DRLAgent(train_env).get_model("pg", model_kwargs, policy_kwargs)
```


在下面的示例中，模型在 5 个回合中进行训练（我们将一个回合定义为使用的环境的一个完整周期）。

```python
DRLAgent.train_model(model, episodes=5)
```


重要的是，架构和环境必须定义相同的 `time_window`。默认情况下，两者都使用 50 个时间步作为 `time_window`。有关时间窗口的更多详细信息，请参阅此[文章](https://doi.org/10.5753/bwaif.2023.231144)。

### 策略梯度算法

类 `PolicyGradient` 实现了 *Jiang et al* 论文中使用的策略梯度算法。该算法受到 DDPG（深度确定性策略梯度）的启发，但有一些不同之处：
- DDPG 是一种演员-评论家算法，因此它具有演员和评论家神经网络。然而，下面的算法没有评论家神经网络，而是使用投资组合价值作为价值函数：策略将被更新以最大化投资组合价值。
- DDPG 通常在训练期间使用动作中的噪声参数来创建探索行为。而 PG 算法则采用完全利用的方法。
- DDPG 随机从其回放缓冲区中采样经验。然而，实现的策略梯度按时间顺序采样一批经验，以便能够计算批次中投资组合价值的变化并将其用作价值函数。

算法实现如下：
1. 初始化策略网络和回放缓冲区；
2. 对于每个回合，执行以下操作：
    1. 对于每个 `batch_size` 时间步长周期，执行以下操作：
        1. 对于每个时间步长，定义要执行的动作，模拟时间步长并将经验保存到回放缓冲区。
        2. 在模拟完 `batch_size` 个时间步长后，采样回放缓冲区。
        3. 计算价值函数：$V = \sum\limits_{t=1}^{batch\_size} ln(\mu_{t}(W_{t} \cdot P_{t}))$，其中 $W_{t}$ 是时间步长 t 执行的动作，$P_{t}$ 是时间步长 t 的价格变化向量，$\mu_{t}$ 是时间步长 t 的交易剩余因子。更多细节请参阅 *Jiang et al* 论文。
        4. 在策略网络中执行梯度上升。
    2. 如果在回合结束时，回放缓冲区中有剩余经验序列，则对剩余经验执行步骤 1 到 5。

### 参考文献

如果您在研究中使用了其中之一，可以使用以下参考文献。

#### EIIE 架构和策略梯度算法

[用于金融投资组合管理问题的深度强化学习框架](https://doi.org/10.48550/arXiv.1706.10059)
```
@misc{jiang2017deep,
      title={A Deep Reinforcement Learning Framework for the Financial Portfolio Management Problem},
      author={Zhengyao Jiang and Dixing Xu and Jinjun Liang},
      year={2017},
      eprint={1706.10059},
      archivePrefix={arXiv},
      primaryClass={q-fin.CP}
}
```


#### EI3 架构

[用于投资组合管理的多尺度时间特征聚合卷积神经网络](https://doi.org/10.1145/3357384.3357961)
```
@inproceedings{shi2018multiscale,
               author = {Shi, Si and Li, Jianjun and Li, Guohui and Pan, Peng},
               title = {A Multi-Scale Temporal Feature Aggregation Convolutional Neural Network for Portfolio Management},
               year = {2019},
               isbn = {9781450369763},
               publisher = {Association for Computing Machinery},
               address = {New York, NY, USA},
               url = {https://doi.org/10.1145/3357384.3357961},
               doi = {10.1145/3357384.3357961},
               booktitle = {Proceedings of the 28th ACM International Conference on Information and Knowledge Management},
               pages = {1613–1622},
               numpages = {10},
               keywords = {portfolio management, reinforcement learning, inception network, convolution neural network},
               location = {Beijing, China},
               series = {CIKM '19} }
```
