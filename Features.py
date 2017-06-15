import xml.etree.ElementTree as Et
from datetime import datetime
import pandas as pd
import os


class Features(object):
    dateFormat = '%Y-%m-%d %H:%M:%S'

    @staticmethod
    def writeToEmptyFile(root, records, headers, i):
        while i < len(root):
            child = root[i]
            dateStr = child.get('Time')[:-3]
            # date = datetime.strptime(dateStr, Features.dateFormat)
            if child.text == 'null':
                child.text = 'NaN'

            if dateStr != headers[-1]:
                headers.append(dateStr)
                records.append(float(child.text))

            i += 1

        df = pd.DataFrame.from_records([records], columns=headers)
        return df

    @staticmethod
    def writeToExistingFile(dstFile, root, records, i):
        df = pd.read_csv(dstFile)
        headers = df.columns.values.tolist()
        print(headers)
        jStart = 1

        j = jStart
        headerDate = datetime.strptime(headers[j], Features.dateFormat)
        # skip columns we can't fill with data because they before 1st valid date
        child = root[i]
        dateStr = child.get('Time')[:-3]
        date = datetime.strptime(dateStr, Features.dateFormat)
        while date > headerDate:
            j += 1
            headerDate = datetime.strptime(headers[j], Features.dateFormat)

        cutColsIndex = j

        prevDate = None
        while i < len(root):
            child = root[i]
            dateStr = child.get('Time')[:-3]
            date = datetime.strptime(dateStr, Features.dateFormat)
            if child.text == 'null':
                child.text = 'NaN'

            # column already exist for this file
            if dateStr == prevDate:
                i += 1
                continue

            prevDate = dateStr

            if date == headerDate:
                j += 1
                if j < len(headers):
                    headerDate = datetime.strptime(headers[j], Features.dateFormat)
            elif date < headerDate:
                try:
                    df.insert(j, dateStr, df.iloc[:, j - 1])
                except ValueError:
                    print('[%s] - [%s]' % (date, headerDate))
                    break
                headers.insert(j, dateStr)
                # print('Added [%s] column at position [%d]' % (dateStr, j))

                j += 1
                headerDate = datetime.strptime(headers[j], Features.dateFormat)
            else:
                while date > headerDate:
                    j += 1
                    # print('[%d] - [%d]' % (j, len(headers)))
                    # print('[%s] - [%s] - [%s]' % (headers[j], date, headerDate))
                    headerDate = datetime.strptime(headers[j], Features.dateFormat)
                    records.append(records[-1])

                if date < headerDate:
                    df.insert(j, dateStr, df.iloc[:, j - 1])
                    headers.insert(j, dateStr)
                    # print('Added [%s] column at position [%d]' % (dateStr, j))

                j += 1
                headerDate = datetime.strptime(headers[j], Features.dateFormat)

            records.append(float(child.text))
            i += 1

        # there are columns to cut
        if cutColsIndex > jStart:
            print(df.shape)
            df = df.drop(df.columns[jStart:cutColsIndex], axis=1)
            del headers[jStart:cutColsIndex]
            print(df.shape)

        dfNew = pd.DataFrame.from_records([records], columns=headers)
        df = df.append(dfNew)

        return df

    @staticmethod
    def writeToCSV(srcFile, dstFile):
        print('writeToCSV:')
        records, headers = [srcFile], ['filename']

        root = Et.parse(srcFile).getroot()
        # at the moment we decide to skip values until the 1st value which isn't NaN or null
        i = 0
        valueFlag = False
        while i < len(root):
            child = root[i]
            valueFlag = (child.text != 'NaN') and (child.text != 'null')
            if valueFlag is True:
                break

            i += 1

        emptyFile = not os.path.isfile(dstFile)
        if emptyFile is True:
            df = Features.writeToEmptyFile(root, records, headers, i)
        else:
            df = Features.writeToExistingFile(dstFile, root, records, i)

        with open(dstFile, 'w') as f:
            print('#headers:[%d] - #records:[%d]' % (len(df.columns.values), len(records)))
            df = df.fillna(method='ffill', axis=1)  # fill with previous value in row
            df.to_csv(f, header=True, index=False)

            print('Done !')


Features.writeToCSV('data/LightPoints/Devices.LightsAndAutomation.LightPoint.1.2.xml', 'data/LP.csv')
Features.writeToCSV('data/LightPoints/Devices.LightsAndAutomation.LightPoint.1.3.xml', 'data/LP.csv')
# Features.writeToCSV('data/ThermalProbe/Devices.ClimateControl.ThermalProbe.1.xml', 'data/features.csv')
# Features.writeToCSV('data/ThermalProbe/Devices.ClimateControl.ThermalProbe.3.xml', 'data/features.csv')
# Features.writeToCSV('data/ThermalProbe/gg.xml', 'data/features.csv')
# Features.writeToCSV('data/ThermalProbe/vv.xml', 'data/features.csv')
# Features.writeToCSV('data/ThermalProbe/Devices.ClimateControl.ThermalProbe.1.xml', 'data/features.csv')


# @staticmethod
# def writeToCSV(srcFile, dstFile):
#     print('writeToCSV:')
#     # df = pd.read_csv(dstFile)
#     with open(dstFile, 'w') as f:
#         root = Et.parse(srcFile).getroot()
#         records, headers = [srcFile], ['filename']
#         for i, child in enumerate(root):
#             date = child.get('Time')[:-3]
#
#             if date != headers[-1]:
#                 records.append(child.text)
#                 headers.append(date)
#             else:
#                 print('[%s] already exists as header' % date)
#
#         # del headers[0]
#         df = pd.DataFrame.from_records([records], columns=headers)
#         df.to_csv(f)
#
#     # df2 = pd.DataFrame.from_records([records], columns=headers)
#     # dfNew = df.append(df2, ignore_index=True)
#     # # print(df)
#     # print(dfNew)
#     # # df.to_csv(dfNew, header=False)
