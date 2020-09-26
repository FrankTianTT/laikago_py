from build_envs import locomotion_gym_env
from build_envs import locomotion_gym_config
from build_envs.env_wrappers import observation_dictionary_to_array_wrapper
from build_envs.sensors import environment_sensors
from build_envs.sensors.sensor_wrappers import HistoricSensorWrapper, NormalizeSensorWrapper
from build_envs.sensors import robot_sensors
from build_envs.utilities import controllable_env_randomizer_from_config
from robots import laikago
from envs.robots.robot_config import MotorControlMode
from tasks.standuppush import task

def build_env(enable_randomizer, enable_rendering, version=0, mode='train', control_mode='torque', force=True):

    sim_params = locomotion_gym_config.SimulationParameters()
    sim_params.enable_rendering = enable_rendering
    gym_config = locomotion_gym_config.LocomotionGymConfig(simulation_parameters=sim_params)

    robot_class = laikago.Laikago

    sensors = [
        HistoricSensorWrapper(NormalizeSensorWrapper(
            wrapped_sensor=robot_sensors.MotorAngleSensor(num_motors=laikago.NUM_MOTORS)), num_history=3),
        HistoricSensorWrapper(NormalizeSensorWrapper(
            wrapped_sensor=robot_sensors.MotorVelocitiySensor(num_motors=laikago.NUM_MOTORS)), num_history=3),
        HistoricSensorWrapper(NormalizeSensorWrapper(wrapped_sensor=robot_sensors.IMUSensor()), num_history=3),
        HistoricSensorWrapper(NormalizeSensorWrapper(wrapped_sensor=robot_sensors.ToeTouchSensor(laikago.NUM_LEGS)), num_history=3),
        HistoricSensorWrapper(NormalizeSensorWrapper(
            wrapped_sensor=environment_sensors.LastActionSensor(num_actions=laikago.NUM_MOTORS)), num_history=3)
    ]


    task = eval('task.StanduppushTaskV{}'.format(version))(mode=mode, force=force)

    randomizers = []
    if enable_randomizer:
        randomizer = controllable_env_randomizer_from_config.ControllableEnvRandomizerFromConfig(verbose=False)
        randomizers.append(randomizer)


    if control_mode == 'torque':
        motor_control_mode = MotorControlMode.TORQUE
    else:
        motor_control_mode = MotorControlMode.POSITION

    init_pose = 'stand'
    env = locomotion_gym_env.LocomotionGymEnv(gym_config=gym_config, robot_class=robot_class,
                                              motor_control_mode=motor_control_mode,
                                              init_pose=init_pose,
                                              env_randomizers=randomizers, robot_sensors=sensors, task=task)

    env = observation_dictionary_to_array_wrapper.ObservationDictionaryToArrayWrapper(env)
    return env