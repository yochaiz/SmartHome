import xml.etree.ElementTree as Et
from datetime import datetime
import pandas as pd
import os


class Features(object):
    dateFormat = '%Y-%m-%d %H:%M:%S'

    @staticmethod
    def writeToCSV(srcFile, dstFile):
        print('writeToCSV:')
        df = None
        records, headers = [srcFile], ['filename']
        headerDate = None

        emptyFile = not os.path.isfile(dstFile)
        if not emptyFile:
            df = pd.read_csv(dstFile)
            headers = df.columns.values.tolist()
            print(headers)
            j = 1
            headerDate = datetime.strptime(headers[j], Features.dateFormat)

        root = Et.parse(srcFile).getroot()
        # at the moment we decide to skip values until the 1st value which isn't NaN
        i = 0
        while i < len(root):
            child = root[i]
            if child.text != 'NaN':
                break

            i += 1

        while i < len(root):
            # for i, child in enumerate(root):
            child = root[i]
            dateStr = child.get('Time')[:-3]
            date = datetime.strptime(dateStr, Features.dateFormat)

            # print('[%s] - [%s]' % (date, headerDate))

            if headerDate is None:
                # check if header already exist and skip it for does
                if dateStr == headers[-1]:
                    i += 1
                    continue

                headers.append(dateStr)
            elif date == headerDate:
                j += 1
                if j < len(headers):
                    headerDate = datetime.strptime(headers[j], Features.dateFormat)
            elif date < headerDate:
                df.insert(j, dateStr, df.iloc[:, j - 1])
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
                    # print('** [%s] - [%s]' % (date, headerDate))
                    records.append(records[-1])

                if date < headerDate:
                    df.insert(j, dateStr, df.iloc[:, j - 1])
                    headers.insert(j, dateStr)
                    # print('Added [%s] column at position [%d]' % (dateStr, j))

                j += 1
                headerDate = datetime.strptime(headers[j], Features.dateFormat)

            records.append(float(child.text))
            i += 1

        with open(dstFile, 'w') as f:
            if emptyFile:
                df = pd.DataFrame.from_records([records], columns=headers)
            else:
                dfNew = pd.DataFrame.from_records([records], columns=headers)
                df = df.append(dfNew)

            # print(df.columns.values)
            # print(records)
            print('#headers:[%d] - #records:[%d]' % (len(df.columns.values), len(records)))
            df = df.fillna(method='ffill', axis=1)  # fill with previous value in row
            df.to_csv(f, header=True, index=False)

        print('Done !')


# Features.writeToCSV('data/ThermalProbe/Devices.ClimateControl.ThermalProbe.1.xml', 'data/features.csv')
Features.writeToCSV('data/ThermalProbe/Devices.ClimateControl.ThermalProbe.3.xml', 'data/features.csv')
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
