from Reinforcement.Functions import *

# set folder argument
parser = argparse.ArgumentParser(description='Collect results from log file')
parser.add_argument("Folder", type=str, help="Log file folder location")
args = parser.parse_args()

CollectResultsFromLog(args.Folder)
print('Done !')
