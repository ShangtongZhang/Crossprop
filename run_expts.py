import os
import time
import numpy as np

def main():
	sleep_time = 120  # 2 mins
	step_size_list = np.power(2.0, np.arange(-16, 0))
	for ss in step_size_list:
		cmd_exec = 'python tfTrainMNIST.py {}'.format(ss)
		os.system(cmd_exec)
		time.sleep(sleep_time)

if __name__ == '__main__':
	main()