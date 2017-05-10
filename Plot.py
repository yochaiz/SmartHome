class Plot(object):
    @staticmethod
    # Removes too close labels
    def dateWithMinimalGap(lists, compareValFunc, minGap=None, width=50.0):
        x = lists[0]  # assumes all lists are in the same length

        if minGap is None:
            minGap = round((x[len(x) - 1] - x[0]).seconds / width)

        i = 1
        while i < len(x):
            if compareValFunc(i) < minGap:
                for l in lists:
                    del l[i]
            else:
                i += 1

        return lists

    @staticmethod
    # converts time delta object to text of the greater time type (days,hours,min,secs)
    def timedeltaToText(t):
        if t.days > 0:
            return "[%d] Days" % t.days

        hours, remainder_ = divmod(t.seconds, 3600)
        minutes, seconds = divmod(remainder_, 60)
        if hours > 0:
            return "[%d] Hours, [%d] Minutes, [%d] Seconds" % (hours, minutes, seconds)

        if minutes > 0:
            return "[%d] Minutes, [%d] Seconds" % (minutes, seconds)

        if seconds > 0:
            return "[%d] Seconds" % seconds

        return "No data to display"
