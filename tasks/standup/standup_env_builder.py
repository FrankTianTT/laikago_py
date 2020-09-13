#!/usr/bin/env python3
# by frank tian on 7.9.2020

from build_envs import locomotion_gym_env
from build_envs import locomotion_gym_config
from build_envs.env_wrappers import observation_dictionary_to_array_wrapper
from standup import standup_task
from build_envs.sensors import environment_sensors
from build_envs.sensors import sensor_wrappers
from build_envs.sensors import robot_sensors
from build_envs.utilities import controllable_env_randomizer_from_config
from robots import laikago


def build_env(enable_randomizer, enable_rendering, version=0, mode='train', max_episode_steps=300):

    sim_params = locomotion_gym_config.SimulationParameters()
    sim_params.enable_rendering = enable_rendering
    gym_config = locomotion_gym_config.LocomotionGymConfig(simulation_parameters=sim_params)

    robot_class = laikago.Laikago
    '''
    MotorAngleSensor
    MotorVelocitiySensor
    IMUSensor
    ToeTouchSensor
    LastActionSensor
    '''
    sensors = [
        sensor_wrappers.HistoricSensorWrapper(
            wrapped_sensor=robot_sensors.MotorAngleSensor(num_motors=laikago.NUM_MOTORS), num_history=3),
        sensor_wrappers.HistoricSensorWrapper(
            wrapped_sensor=robot_sensors.MotorVelocitiySensor(num_motors=laikago.NUM_MOTORS), num_history=3),
        sensor_wrappers.HistoricSensorWrapper(wrapped_sensor=robot_sensors.IMUSensor(), num_history=3),
        sensor_wrappers.HistoricSensorWrapper(wrapped_sensor=robot_sensors.ToeTouchSensor(laikago.NUM_LEGS), num_history=3),
        sensor_wrappers.HistoricSensorWrapper(
            wrapped_sensor=environment_sensors.LastActionSensor(num_actions=laikago.NUM_MOTORS), num_history=3)
    ]


    task = eval('standup_task.StandupTaskV{}'.format(version))(mode=mode, max_episode_steps=max_episode_steps)

    randomizers = []
    if enable_randomizer:
        randomizer = controllable_env_randomizer_from_config.ControllableEnvRandomizerFromConfig(verbose=False)
        randomizers.append(randomizer)

    env = locomotion_gym_env.LocomotionGymEnv(gym_config=gym_config, robot_class=robot_class,
                                              env_randomizers=randomizers, robot_sensors=sensors, task=task)

    env = observation_dictionary_to_array_wrapper.ObservationDictionaryToArrayWrapper(env)
    return env

