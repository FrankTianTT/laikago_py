import os
import shutil

TASK_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(TASK_DIR, 'log')


def print_tensoroard(port=None):
    if port is not None:
        print('conda activate laikago_py')
        print('tensorboard --logdir ' + LOG_DIR + ' --port ' + str(port))
    else:
        print('tensorboard --logdir ' + LOG_DIR)



def delete_log():
    shutil.rmtree(LOG_DIR)


if __name__ == "__main__":
    #delete_log()
    print_tensoroard()