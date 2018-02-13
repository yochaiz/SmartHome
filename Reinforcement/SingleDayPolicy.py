from datetime import time, timedelta, datetime
from random import randint
import numpy as np

class SingleDayPolicy:
    # a policy for a single light over a single day.
    # the policy is that the light is ON only between 6:00-18:00.
    # the rest of the day, the light is OFF.
    # state structure: [Hour,Minute,Light state {1=on,0=off}]
    # possible actions: ['Don't change status' , 'Change light status' (i.e. turn it on if off)]
    # startTime = time(hour=6, minute=0)
    # endTime = time(hour=18, minute=0)

    def __init__(self, startTime, endTime):
        self.stateSize = 3
        self.actionSize = 2
        self.startTime = startTime
        self.endTime = endTime

    # perform action
    def step(self, state, action):
        curDate = datetime(year=2000, month=1, day=1, hour=state[0], minute=state[1])
        nextDate = curDate + timedelta(minutes=1)
        # update light state at newState due to action performed
        nextState = np.array([nextDate.hour, nextDate.minute, (state[2] + action) % self.actionSize])

        # print(state)
        # print(action)
        # print(nextState)
        # print(curDate.time())
        # print(nextDate.time())
        # print(self.startTime)
        # print(self.endTime)

        positiveReward = 1
        negativeReward = -1

        # light is on
        if state[2] == 1:
            # light should be on
            if self.startTime <= curDate.time() <= self.endTime:
                if action == 0:  # agent doesn't change status
                    reward = positiveReward
                else:  # agent changes status (turns off)
                    reward = negativeReward  # * (self.calcMinutesGap(self.endTime, curDate.time()))

            # light should be off
            else:
                if action == 0:  # agent doesn't change status
                    reward = negativeReward  # * (self.calcMinutesGap(self.startTime, curDate.time()))
                else:  # agent changes status (turns off)
                    reward = positiveReward

        # light is off
        else:
            # light should be on
            if self.startTime <= curDate.time() <= self.endTime:
                if action == 0:  # agent doesn't change status
                    reward = negativeReward  # * (self.calcMinutesGap(self.endTime, curDate.time()))
                else:  # agent changes status (turns off)
                    reward = positiveReward

            # light should be off
            else:
                if action == 0:  # agent doesn't change status
                    reward = positiveReward
                else:  # agent changes status (turns off)
                    reward = negativeReward  # * (self.calcMinutesGap(self.startTime, curDate.time()))

        # print('Time:[{}:{}] , Light state:[{}] , Action:[{}] , Reward:[{}]'.format(curDate.hour, curDate.minute, state[2], action, reward))
        return nextState, reward

    def getRandomState(self):
        hour = randint(0, 23)
        minute = randint(0, 59)
        randTime = time(hour=hour, minute=minute)
        state = np.array([hour, minute, 0])
        if self.startTime <= randTime <= self.endTime:
            state[2] = 1  # light is on

        return state

    def toJSON(self):
        res = dict(vars(self)) # make a copy
        res['Labels'] = []
        res['Labels'].append({
            'startTime': str((datetime(year=1900, month=1, day=1, hour=self.startTime.hour, minute=self.startTime.minute) + timedelta(minutes=1)).time()),
            'endTime': str((datetime(year=1900, month=1, day=1, hour=self.endTime.hour, minute=self.endTime.minute) + timedelta(minutes=1)).time()),
            'label': 1
        })
        res['Labels'].append({
            'startTime': str((datetime(year=1900, month=1, day=1, hour=self.endTime.hour, minute=self.endTime.minute) + timedelta(minutes=2)).time()),
            'endTime': str((datetime(year=1900, month=1, day=1, hour=self.startTime.hour, minute=self.startTime.minute) + timedelta(minutes=2)).time()),
            'label': 0
        })

        res['startTime'] = str(res['startTime'])
        res['endTime'] = str(res['endTime'])

        return res


    # @staticmethod
    # def calcMinutesGap(time1, time2):
    #     if time1 > time2:
    #         endTime = time1
    #         startTime = time2
    #     else:
    #         endTime = time2
    #         startTime = time1
    #
    #     duration = datetime.combine(date.min, endTime) - datetime.combine(date.min, startTime)
    #     return duration.seconds / 60