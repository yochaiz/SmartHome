import sys
import re
from Reinforcement.Functions import *
from Reinforcement.Results import Results

# set folder argument
parser = argparse.ArgumentParser(description='Collect results from log file')
parser.add_argument("Folder", type=str, help="Log file folder location")
args = parser.parse_args()

logFname = '{}/info.log'.format(args.Folder)
jsonFname = '{}/info.json'.format(args.Folder)

# define regex
pattern = re.compile('.*:episode: (\d+), score:\[(\d+\.\d+)\], loss:\[(\d+\.\d+)\]')

# check files exist
if not os.path.exists(jsonFname):
    print('path [{}] does not exist'.format(jsonFname))
    sys.exit()

# load log file
if not os.path.exists(logFname):
    print('path [{}] does not exist'.format(logFname))
    sys.exit()

# load JSON file
info, jsonFname = loadInfoFile(args.Folder, None)

# init results object
results = Results()

# add results from log
with open(logFname, 'r') as f:
    for line in f:
        m = pattern.match(line)
        if m:
            results.score.append(int(m.group(2)))
            results.loss.append(float(m.group(3)))

# update info data in JSON, add results
info['results'] = results.toJSON()
saveDataToJSON(info, jsonFname)

print('Done !')