from recbole.trainer import HyperTuning
from recbole.quick_start import objective_function
import numpy as np
import argparse
from datetime import datetime as dt

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default="ml-100k_custom")
parser.add_argument("--model_name")
parser.add_argument("--params")
parser.add_argument("--config")
args = parser.parse_args()

params_file = "hp/" + args.params + ".hyper"
config_file = "hp/" + args.config + ".yaml"
output_file = "hp/" + args.dataset + "/" + args.model_name + ".results"

starttime = dt.now()
hp = HyperTuning(objective_function=objective_function, algo='exhaustive', early_stop=int(1e6),
                 params_file=params_file, fixed_config_file_list=[config_file])

# run
hp.run()
# export result to the file
hp.export_result(output_file=output_file)
# print best parameters
print('best params: ', hp.best_params)
# print best result
print('best result: ')
print(hp.params2result[hp.params2str(hp.best_params)])

endtime = dt.now()
print(endtime - starttime)