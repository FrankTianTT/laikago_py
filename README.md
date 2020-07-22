# laikago_py



## 文件目录

- `envs`目录下的是环境文件;
- `algorithms`目录下单是算法文件，主要是不同DRL算法的网络模型;
- `tasks`目录下的是任务文件.



### envs

部分代码来自开源项目[motion_imitation](https://github.com/google-research/motion_imitation).

```
├─build_envs
│  ├─env_wrappers
│  ├─sensors
│  └─utilities
├─laikago_model
├─robots
└─utilities
```

`laikago_model`目录下的是laikago的模型文件，不需要任何改动。
`robots`目录下的是laikago的仿真模型，包括动力学参数。
`build_envs`负责将模型封装为gym模型。使用`locomotion_gym_env`封装。

### algorithms

部分代码来自[Deep-Reinforcement-Learning-Hands-On](https://github.com/Shmuma/Deep-Reinforcement-Learning-Hands-On).

### tasks

主要的训练文件，每个task代表一个任务，理论上每个任务对应一套reward机制。

## 训练

新建每个task，你需要:

- 在task目录下新建对应的文件夹；
- 建立对应的_task文件，用于配置强化学习算法的逻辑；
- 建立对应的_env_builder文件，用于链接_task文件和`locomotion_gym_env`文件，以及完成其他更高级的配置。

### 配置_task文件

你需要新建一个Task类，例如`StandupTask`，你可以继承自`envs/build_envs/laikago_task`，也可以自己写。

无论如何，为了强化学习算法运行正常，请确保你的类中包括以下函数:

- reward(self, env):返回一个数值；
- done(self, env)：返回bool；

以及
```python
def __call__(self, env):
    return self.reward(env)
```

如果你想在环境中添加其他功能，例如增加随机扰动，可以在重载`update`函数，例如:

```python
def update(self, env):
    if not self.force:
        return
    force = self._give_force()
    self.body_pos = env._pybullet_client.getBasePositionAndOrientation(self.quadruped)[0]
    env._pybullet_client.applyExternalForce(objectUniqueId=self.quadruped, linkIndex=-1,
                         forceObj=force, posObj=self.body_pos, flags=env._pybullet_client.WORLD_FRAME)
```

### 配置_env_builder文件


