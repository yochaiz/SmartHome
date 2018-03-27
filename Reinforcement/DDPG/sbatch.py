import sys
import time
from subprocess import Popen
from datetime import datetime

now = datetime.now()
outputFile = 'D-{}-{}-H-{}-{}-{}.out'.format(now.day, now.month, now.hour, now.minute, now.second)

commands = [
    [sys.executable, './train.py', '0', '--random', '--k', '32', '--desc',
     '"knn over all possible discrete actions, ResNet architecture with ~34 layers"'],
    [sys.executable, './train.py', '0', '--random', '--desc', '"default knn size, ResNet architecture with ~34 layers"']
]

# calc GPU fraction
nProc = len(commands)
gpuFrac = 0.90 / nProc

# add gpu fraction to commands
for cmd in commands:
    cmd.append('--gpuFrac')
    cmd.append('{:.3f}'.format(gpuFrac))

procs = []

with open(outputFile, mode='w') as out:
    # print commands
    out.write('{}'.format(commands))
    out.write('***')
    # run processes
    for cmd in commands:
        p = Popen(cmd, stdout=out, stderr=out)
        procs.append(p)
        time.sleep(10)
		

for p in procs:
    p.wait()

# files = ['j.py', 'j2.py']
#
# for file in files:
#     with io.open('{}.txt'.format(file), mode='w') as out:
#         p = Popen([sys.executable, './{}'.format(file), '400', '--k', '3'], stdout=out, stderr=out)
#         procs.append(p)
#
# for p in procs:
#     p.wait()
