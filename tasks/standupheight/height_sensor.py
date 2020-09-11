from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import typing
import random
from build_envs.sensors import sensor

_ARRAY = typing.Iterable[float]
_FLOAT_OR_ARRAY = typing.Union[float, _ARRAY]
_DATATYPE_LIST = typing.Iterable[typing.Any]


class HeightSensor(sensor.BoxSpaceSensor):
  """A sensor that reads motor angles from the robot."""

  def __init__(self,
               lower_bound: _FLOAT_OR_ARRAY = 0.2,
               upper_bound: _FLOAT_OR_ARRAY = 0.5,
               name: typing.Text = "HeightSensor",
               dtype: typing.Type[typing.Any] = np.float64,
               random_height=True) -> None:
    """Constructs MotorAngleSensor.

    Args:
      num_motors: the number of motors in the robot
      noisy_reading: whether values are true observations
      observe_sine_cosine: whether to convert readings to sine/cosine values for
        continuity
      lower_bound: the lower bound of the motor angle
      upper_bound: the upper bound of the motor angle
      name: the name of the sensor
      dtype: data type of sensor value
    """
    super(HeightSensor, self).__init__(
      name=name,
      shape=(1,),
      lower_bound=lower_bound,
      upper_bound=upper_bound,
      dtype=dtype)
    self.random_height = random_height
    self.set_random_height('init sensor')
    self.timer = 0
    self.fit_timer = 0
    self.time_stamp = 0

  def get_now_height(self):
      quadruped = self._robot.quadruped
      return self._robot._pybullet_client.getBasePositionAndOrientation(quadruped)[0][2]

  def set_random_height(self, why):
      #print(why)
      self.now_expect_height = random.random()*(self._upper_bound - self._lower_bound) + self._lower_bound

  def get_random_observation(self):
      self.timer += 1

      if self._robot._step_counter == 0:
          self.timer = 0
          self.fit_timer = 0
          self.set_random_height('reset env')

      if self.timer > 50:
          self.set_random_height('time too long')
          self.timer = 0

      if abs(self.get_now_height() - self.now_expect_height) < 0.03:
          self.fit_timer += 1
          if self.fit_timer > 10:
              self.set_random_height('fit enough time' + str((self.get_now_height() - self.now_expect_height)**2))
              self.timer = 0
              self.fit_timer = 0
      else:
          self.fit_timer = 0

  def _get_observation(self) -> _ARRAY:
      # 防止被多次调用时timer计时
      if self._robot._step_counter == self.time_stamp:
          return self.now_expect_height

      if self.random_height:
        self.get_random_observation()
      self.time_stamp = self._robot._step_counter
      return self.now_expect_height