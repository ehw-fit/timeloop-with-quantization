* `timeloop.py` - This has a function called `run_timeloop(dirname, configfile, logfile='timeloop.log', workload_bounds=None)` which
    * copies `configfile` to `dirname`
    * modifies the copied configuration file with `workload_bounds`, if given
    * runs timeloop
    * puts all results in `dirname`. `dirname` will contain:
        * `logfile` which is the text log from timeloop
        * the (possibly modified) configuration file copy
        * the xml output file from timeloop

* `parse_timeloop_output.py` - This has a function called `parse_timeloop_stats(path)` which looks for `timeLoopOutput.xml` at `path` (can be a full file path or just a path to the directory) and parses it and returns a python dictionary with the statistics we care about.
This file is also a command-line tool that uses this functionality to produce pickle files of these dictionaries, which can be used to store and compare parsed outputs over time.

* The `construct_workload` folder contains scripts for automatic creation of problem conv layers into Timeloop's workload yaml format
    - `construct_workloads.py` script is used for generating your own workloads in Timeloop format
    - first you can look into the `temps/cnn_layers.yaml` file to see example descriptions of model's conv layers
    - than you can use the `parse_model.py` script to load in Keras or PyTorch model and parse it into the list of workload layers used by the `construct_workloads.py`
    - examle run may be: `python3 parse_model.py --api_name keras --model mobilenet -o keras_mobilenet`
    - this will create the `keras_mobilenet.yaml` file containing the model's conv layers shapes into the `parsed_models` folder
    - then proceed to run the main constructor script `python3 construct_workloads.py <my_dnn_model_name>.yaml`
    - for additional scripts arguments examine their help messages