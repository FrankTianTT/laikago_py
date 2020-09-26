# laikago_py

## 文件目录

- `envs`目录下的是环境文件;
- `deeprl`目录下的是算法的测试文件，使用stable baselines 3在几个连续状态动作环境中进行了测试;
- `tasks`目录下的是任务文件.

### envs

部分代码来自开源项目[motion_imitation](https://github.com/google-research/motion_imitation).

```
├─build_envs
│  ├─env_wrappers
│  ├─sensors
│  ├─utilities
│  └─laikago_task.py
├─laikago_model
├─robots
└─utilities
```

- `laikago_model`目录下的是laikago的模型文件，不需要任何改动。
- `robots`目录下的是laikago的仿真模型，包括动力学参数。
- `build_envs`负责将模型封装为gym模型。使用`locomotion_gym_env`封装。

`laikago_task.py`是task类的基类，提供了各种计算reward的函数，之后在介绍tasks的部分详细介绍。

### deeprl

[这个项目](https://github.com/FrankTianTT/laikago_py) 使用[stable-baseline3](https://github.com/DLR-RM/stable-baselines3) 作为训练算法。

我们在HalfCheetah, Hopper和Walker三个环境中对算法进行了测试。

### tasks

主要的训练文件，每个task代表一个任务，理论上每个任务对应一套reward机制，但是在实践中，为了尽可能对找到合适的reward，我们往往会设计多种reward机制。


新建每个task，你需要:

- 在`tasks`目录下新建对应的文件夹；
- 建立对应的`xxx_task`文件，用于配置强化学习算法的逻辑；
- 建立对应的`xxx_env_builder.py`文件，用于链接`xxx_task.py`文件和`locomotion_gym_env.py`文件，以及完成其他更高级的配置。

`task`目录的文件如下：

```
├─log&model
├─env_builder.py
└─task.py

```
#### 配置task文件


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

#### 配置env_builder文件


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

## 重要的类

### LaikagoTask

laikago_task.py文件下LaikagoTask类的封装了基本的reward机制和done机制，直接继承这个类可以让你的task更简洁。


- `_reward_of_toe_collision(self)`给出**鼓励**机器人脚趾与地面接触的奖赏
- `_reward_of_leg_collision(self)`给出**阻止**机器人大腿与地面接触的奖赏
- `_reward_of_upward_ori(self)`给出**鼓励**机器人身体端正的奖赏
- `_reward_of_stand_height(self, max_height=0.4)`给出**鼓励**机器人身体高度
- `_reward_of_energy(self)`给出**阻止**机器人消耗能量
- `_reward_of_sum_vel(self)`给出**阻止**机器人关节角速度过快的奖赏
- `_reward_of_toes_height(self)`给出**阻止**机器人脚趾位置过高
- `_reward_of_toe_upper_distance(self)`给出**鼓励**机器人脚趾与大腿尽可能远的奖赏

上述的奖赏都经过了`normalize_reward`函数归一化，近似分布在(0,1)之间。

同时集成了一些`done`和`not_done`的机制。

一个继承自这个类的reward函数可以写作：

```python
def reward(self, env):
    del env
    sum_vel_r = self._reward_of_sum_vel()
    collision_r = self._reward_of_toe_collision()
    height_r = self._reward_of_stand_height()
    toe_upper_r = self._reward_of_toe_upper_distance()

    reward = sum_vel_r + collision_r + height_r + toe_upper_r
    return reward/4
```

一个继承自这个类的done函数可以写作：

```python
def done(self, env):
    del env
    if self._not_done_of_too_short() or self._not_done_of_mode(self.mode):
        return False
    else:
        return self._done_of_wrong_stand_ori() or self._done_of_low_height() or self._done_of_too_long()
```

### Sensor

Sensor类下有许多可以用来作为环境obs的传感器。

可以分为environment-sensor和robot-sensor，environment-sensor是环境记录下来的信息，robot-sensor是robot自身读取的信息。

具体的sensor主要有：
- MotorAngleSensor
- MotorVelocitiySensor
- ToeTouchSensor
- IMUSensor
- LastActionSensor

也可以继承`BoxSpaceSensor`类写自己的Sensor，通过xxx_env_builder.py文件绑定到env中。

### SensorWrapper

SensorWrapper类可以对sensor的数据进行处理，例如NormalizeSensorWrapper将obs做了归一化，使算法训练的难度减小。

## 训练

进入根目录，执行`train.py`文件并确定task的名称和版本，即可开始训练。

例如
```
python train.py -n standup -v 0
```

也可以设置SAC的算法参数，例如
```
python train.py -n standup -v 0 --ent_coef auto_0.1
```
或者在之前best model的基础之上继续训练

```
python train.py -n standup -v 0 -l True
```
## 测试

进入根目录，执行`play.py`文件并确定task的名称和版本，即可开始测试。

例如
```
python play.py -n standup -v 0
```

可以选择让agent永远不会死亡，例如
```
python play.py -n standup -m never_done -v 0
```

## 
项目依赖

```
conda install pytorch torchvision cudatoolkit=10.1 -c pytorch
```