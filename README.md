# 这是《最优化理论及其在电力系统中的应用》的课程作业！

# V2Sim-RL 在V2Sim平台上进行强化学习

从前有一个平台，叫V2Sim，用于交通网和配电网的联合仿真。

现在，要搞一个深度强化学习算法，为充电站自动定价，从而使得充电负荷均匀分布。
这个仓库就是为了这个目的设计的。

### 环境配置
需要Python 3.12版本，然后安装v2sim-rl的依赖包：
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
+ 主程序是`sac.py`(从spinningup里面copy过来修改的)，使用了Soft Actor-Critic算法：
  - 输入`python sac.py`以运行;
  - `sac.py`包含2种模型，一种是标准SAC：`python sac.py -m MLP`；另一种是LSTM-SAC：`python sac.py -m LSTM`；
  - `d`参数可以切换算例，例如：`python -d drl_2cs`
  - `epochs`参数可以改变epoch数量
+ V2Sim环境是`env.py`，直接运行可以看到价格恒为1.0时的return；
  - 使用`search.py`进行网格搜索法，不建议在超过3个充电站的时候使用，因为可能要几个月甚至几年才能跑完；
+ 绘图程序是`plot.py`和`plot_all.py`
  - 运行`plot.py`就会为所有仿真结果绘图，绘图就在结果文件夹里面，请手动打开；
  - 运行`plot_all.pu`会绘制不同仿真结果的对比图；
+ 其他杂项在`utils.py`里面。

### 参数说明
+ observation: 请看`env.py`第230行的函数；
+ action: 一个向量，每一维从0.0~5.0，表示充电站的定价；
+ reward: `-(充电站排队数量极差 + 10 * 充电站负荷极差)`。

V2Sim仿真步长15s，作为DRL的环境的步长600s(即V2Sim走20步)。