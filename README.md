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

- 在`tasks`目录下新建对应的文件夹；
- 建立对应的`_task`文件，用于配置强化学习算法的逻辑；
- 建立对应的`xxx_env_builder.py`文件，用于链接`xxx_task.py`文件和`locomotion_gym_env.py`文件，以及完成其他更高级的配置。

`_task`目录的文件如下：

```
├─play
├─saves
├─train
│  ├─train_xxx.py
│  ├─...
│  └─runs
├─xxx_env_builder.py
└─xxx_task.py

```
### 配置_task文件

你需要新建一个Task类，例如`StandupTask`，你可以继承自`envs/build_envs/laikago_task.py`，也可以自己写。

无论如何，为了强化学习算法运行正常，请确保你的类中包括以下函数:

- `reward(self, env)`:返回一个数值；
- `done(self, env)`：返回bool；

以及
```python
def __call__(self, env):
    return self.reward(env)
```

如果你想在环境中添加其他功能，例如增加随机扰动，可以在重载`update(self, env)`函数，例如:

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

在这个文件中定义`build_env()`函数，返回一个从gym继承的`ENV`类，你可以直接使用locomotion_gym_env文件的`LocomotionGymEnv`完成这个功能，但是这个函数存在的意义在于完成更高级的设置。

在这个函数中完成task和env的绑定，例如
```python
task = runstraight_task.RunstraightTask(mode = mode)
env = locomotion_gym_env.LocomotionGymEnv(gym_config=gym_config, robot_class=robot_class,
                                              env_randomizers=randomizers, robot_sensors=sensors, task=task)
```

也可以在这个函数中添加符合gym规范的wrapper，例如

```python
env = observation_dictionary_to_array_wrapper.ObservationDictionaryToArrayWrapper(env)
```

通过引入`locomotion_gym_config.py`文件进行环境的基本参数设置。

### 开始训练

在`train_`开头的文件中执行训练。

不同的任务只需要更改开头定义的常量内容，例如：

```python
################################
#change these when changing task
import runstraight.runstraight_env_builder as env_builder
TASK_NAME = "runstraight"
FILE_NAME = 'runslow_ppo_128.dat'
################################
```

默认将训练保存的模型存储在相应`task`目录下的`saves`目录里。

默认将训练日志存储在相应`task`目录下的`trian/runs`目录里。

### 测试

在`task/play`目录里的`play_xxx.py`文件中测试，只需要更改`FILE_NAME`即可，`FILE_NAME`即储存在`saves`目录下的`.dat`文件。

例如：

```python
################################
#change these when changing task
import runstraight.runstraight_env_builder as env_builder
TASK_NAME = "runstraight"
FILE_NAME = "best_-31.401_40000.dat"
DONE = True
HID_SIZE = 256
################################
```