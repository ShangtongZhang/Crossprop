import os

jobs = 70
for i in range(1, jobs + 1):
    cmd = 'qsub -l walltime=72:00:00,mem=2048mb ' + 'train' + str(i) + '.sh'
    os.system(cmd)
