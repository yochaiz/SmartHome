import sys
import time
from subprocess import Popen
from datetime import datetime

now = datetime.now()
outputFile = 'D-{}-{}-H-{}-{}-{}.out'.format(now.day, now.month, now.hour, now.minute, now.second)

commands = [
    [sys.executable, './train.py', '0', '--sequential',
     '--k', '32',
     '--desc', '"deep model **sequential** scoreRatio = 0.9"',
     '--settings', '/home/yochaiz/SmartHome/Reinforcement/settings2.json'],
    [sys.executable, './train.py', '0', '--sequential',
     '--k', '32',
     '--rewardScale', '0.5',
     '--desc', '"deep model **sequential** scoreRatio = 0.9 with rewardScaleFactor = 0.5"',
     '--settings', '/home/yochaiz/SmartHome/Reinforcement/settings2.json'],
    [sys.executable, './train.py', '0', '--sequential',
     '--k', '32',
     '--rewardScale', '0.1',
     '--desc', '"deep model **sequential** scoreRatio = 0.9 with rewardScaleFactor = 0.1"',
     '--settings', '/home/yochaiz/SmartHome/Reinforcement/settings2.json']
]

# dstFile = 'ReplayBuffer.py'
# commands = [
#     ([sys.executable, './train.py', '0', '--random', '--k', '32', '--desc', '"step 13 with critic main model"'],
#      'ReplayBuffer-1.py'),
#     ([sys.executable, './train.py', '0', '--random', '--k', '32', '--desc', '"step 13 with critic & actor main model"'],
#      'ReplayBuffer-2.py')
# ]

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
    out.write('***{}***'.format(commands))
    # run processes
    for cmd in commands:
        # copy2(file, dstFile)
        # out.write('copied [{}] to [{}]'.format(file, dstFile))
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
