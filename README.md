# V2Sim-RL 在V2Sim平台上进行强化学习

内部介绍，请勿外传

从前有一个平台，叫V2Sim，用于交通网和配电网的联合仿真。

现在，要搞一个深度强化学习算法，为充电站自动定价，从而防止电压电流越限、减少排队长度、缓解路网拥堵。

### 前置知识

最基本的，需要会用Git、GitHub、Python和某种编辑器（例如VSCode、Anaconda或PyCharm）。

然后，需要了解深度学习(Deep Learning, DL)、强化学习(Reinforcement Learning, RL)和深度强化学习(Deep RL, DRL)，了解常用的算法。本项目目前使用Soft Actor-Critic (SAC)算法。

其次，需要了解强化学习常用的Python库Gymnaisum（简称Gym）。本项目目前把V2Sim平台上的12nodes的例子封装成了Gym环境。

### 环境配置
需要Python 3.12版本（必须恰好是3.12.x版本，不能是3.13这样的更高版本），因此建议开一个虚拟环境。

然后安装v2sim-rl的依赖包：
```bash
# 安装v2sim-rl的依赖。执行以下命令之前，确保你在V2Sim-RL的文件夹里面
pip install -r requirements.txt
```

接下来，还需要安装PyTorch、SUMO和V2Sim

+ PyTorch可以到`https://pytorch.org/`上查看安装教程，注意选择对应的cuda版本；

+ SUMO可以到`https://eclipse.dev/sumo/`下载；

+ V2Sim请从`https://github.com/fmy-xfk/v2sim`下载，然后使用以下命令安装：

```bash
# 安装v2sim。执行以下命令之前，确保你在V2Sim的文件夹里面
pip install -r requirements.txt
python setup.py develop
```

### 项目结构
+ 主程序是`sac.py`(从spinningup里面copy过来修改的)，输入`python sac.py`以运行
+ V2Sim环境是`env.py`
+ SAC的核心程序在`core.py`(从spinningup里面copy过来的)
+ 其他杂项在`utils.py`里面(从spinningup里面copy过来修改的)

记观测数据维度为$N_o$，动作维度为$N_a$，则

+ Actor网络：
+ Critic网络：包括2个结构相同Q网络。每个Q网络都是一个多层感知器(MLP)，默认维度依次为$(N_o+N_a,256,256,1)$，这里两个256是两个hidden layer，可以通过命令行调整，比如变成3个128的命令行为`python sac.py -hid 128 -l 3`，此时维度依次为$(N_o+N_a,128,128,128,1)$