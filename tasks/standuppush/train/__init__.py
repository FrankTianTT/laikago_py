import os
import shutil

TASK_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(TASK_DIR, 'runs')

def print_tensoroard():
    print('tensorboard --logdir ' + LOG_DIR)

def delete_log():
    shutil.rmtree(LOG_DIR)

if __name__ == "__main__":
    #delete_log()
    print_tensoroard()