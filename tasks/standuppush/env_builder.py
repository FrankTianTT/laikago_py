from build_envs import locomotion_gym_env
from build_envs import locomotion_gym_config
from build_envs.env_wrappers import observation_dictionary_to_array_wrapper
from build_envs.sensors import environment_sensors
from build_envs.sensors import sensor_wrappers
from build_envs.sensors import robot_sensors
from tasks.standupheight import height_sensor
from build_envs.utilities import controllable_env_randomizer_from_config
from robots import laikago
from tasks.standuppush import task

def build_env(enable_randomizer, enable_rendering, mode='train', version=0, force=True):

    sim_params = locomotion_gym_config.SimulationParameters()
    sim_params.enable_rendering = enable_rendering
    gym_config = locomotion_gym_config.LocomotionGymConfig(simulation_parameters=sim_params)

    robot_class = laikago.Laikago

    sensors = [
        sensor_wrappers.HistoricSensorWrapper(
            wrapped_sensor=robot_sensors.MotorAngleSensor(num_motors=laikago.NUM_MOTORS), num_history=3),
        sensor_wrappers.HistoricSensorWrapper(
            wrapped_sensor=robot_sensors.MotorVelocitiySensor(num_motors=laikago.NUM_MOTORS), num_history=3),
        sensor_wrappers.HistoricSensorWrapper(wrapped_sensor=robot_sensors.IMUSensor(), num_history=3),
        sensor_wrappers.HistoricSensorWrapper(
            wrapped_sensor=environment_sensors.LastActionSensor(num_actions=laikago.NUM_MOTORS), num_history=3),
        height_sensor.HeightSensor(random_height=True)
    ]

    task = eval('task.StanduppushTaskV{}'.format(version))(mode=mode, force=force)

    randomizers = []
    if enable_randomizer:
        randomizer = controllable_env_randomizer_from_config.ControllableEnvRandomizerFromConfig(verbose=False)
        randomizers.append(randomizer)

    env = locomotion_gym_env.LocomotionGymEnv(gym_config=gym_config, robot_class=robot_class,
                                              env_randomizers=randomizers, robot_sensors=sensors, task=task)

    env = observation_dictionary_to_array_wrapper.ObservationDictionaryToArrayWrapper(env)
    return env