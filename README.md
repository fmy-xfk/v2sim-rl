# V2Sim-RL 在V2Sim平台上进行强化学习

<p style="text-align:center;font-weight:bold">内部介绍，请勿外传</p>


从前有一个平台，叫V2Sim，用于交通网和配电网的联合仿真。

现在，要搞一个深度强化学习算法，为充电站自动定价，从而防止电压电流越限、减少排队长度、缓解路网拥堵。

### 前置知识

最基本的，需要会用Git、GitHub、Python和某种编辑器（例如VSCode、Anaconda或PyCharm）。

然后，需要了解深度学习(Deep Learning, DL)、强化学习(Reinforcement Learning, RL)和深度强化学习(Deep RL, DRL)，了解常用的算法。本项目目前使用Soft Actor-Critic (SAC)算法，可以参考[OpenAI的教程](https://spinningup.openai.com/en/latest/algorithms/sac.html)

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
+ 绘图程序是`plot.py`
+ 其他杂项在`utils.py`里面(从spinningup里面copy过来修改的)

### 参数说明
+ observation: 所有道路的车辆密度(`车辆数/给定容量`) + 所有充电站排队情况(`总车辆数/充电桩数-1`)
+ action: 12维向量，每一维从0.0~5.0，表示充电站的定价。注意：每一维依次对应CS1,CS10,CS11,CS12,CS2,CS3,...,CS9，这是由于字典序导致的。
+ reward: `-(总电压越限*1e5 + 各站排队比例之和*100 + 正在行驶的车辆数/100)`, 其中`排队比例 = 等待车辆数/充电桩数`

Actor网络和Critic网络的结构都是多层感知器(MLP)。可以通过命令行修改MLP的隐含层的维度和层数，例如`python sac.py -hid 128 -l 3`表示隐含层数为3，每层128个神经元。默认为2层，每层256个神经元。

V2Sim仿真步长15s，作为DRL的环境的步长60s(即V2Sim走4步)。

### 现有问题和可能的改进方向
SAC只跑50个epoch效果太差，跑的多了速度太慢。

可以考虑并行DRL，需要调研一下。