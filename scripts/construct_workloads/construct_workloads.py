# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import functools
import yaml
import argparse
from argparse import RawTextHelpFormatter
import json
import os
import inspect
import sys


def json_file_to_dict(file_path):
    # Check if the file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file '{file_path}' does not exist.")

    with open(file_path, 'r') as f:
        return json.load(f)


def rewrite_workload_bounds(src, dst, workload_bounds, verbose, bitwidth_setting, uniform_width, non_uniform_width):
    w, h, c, n, m, s, r, wpad, hpad, wstride, hstride = workload_bounds
    q = int((w - s + 2 * wpad) / wstride) + 1
    p = int((h - r + 2 * hpad) / hstride) + 1

    if verbose:
        print('Workload Dimensions:')
        print('  W        =', w)
        print('  H        =', h)
        print('  C        =', c)
        print('  M        =', m)
        print('  S        =', s)
        print('  R        =', r)
        print('  P        =', p)
        print('  Q        =', q)
        print('  N        =', n)
        print('  W-pad    =', wpad)
        print('  H-pad    =', hpad)
        print('  W-stride =', wstride)
        print('  H-stride =', hstride)
        print()

    with open(src, "r") as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    config['problem']['instance']['R'] = r
    config['problem']['instance']['S'] = s
    config['problem']['instance']['P'] = p
    config['problem']['instance']['Q'] = q
    config['problem']['instance']['C'] = c
    config['problem']['instance']['M'] = m
    config['problem']['instance']['N'] = n
    config['problem']['instance']['Wstride'] = wstride
    config['problem']['instance']['Hstride'] = hstride
    config['problem']['instance']['Wdilation'] = 1
    config['problem']['instance']['Hdilation'] = 1

    if bitwidth_setting == 'uniform':
        bitwidths_dict = {
            'Inputs': uniform_width[0],
            'Weights': uniform_width[1],
            'Outputs': uniform_width[2]
        }
        config['problem']['instance']['bitwidths'] = bitwidths_dict
    elif bitwidth_setting == 'non-uniform':
        # Convert dictionary keys to lowercase (preventing possible errors due to case sensitivity)
        non_uniform_width = {k.lower(): v for k, v in non_uniform_width.items()}
        bitwidths_dict = {
            'Inputs': non_uniform_width["inputs"],
            'Weights': non_uniform_width["weights"],
            'Outputs': non_uniform_width["outputs"]
        }
        config['problem']['instance']['bitwidths'] = bitwidths_dict

    with open(dst, "w") as f:
        f.write(yaml.dump(config))


def create_folder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print('ERROR: Creating directory. ' + directory)
        sys.exit()


def construct_workloads(model, bitwidth_setting, out_file, out_dir, verbose=False, uniform_width_set=None, non_uniform_width_set=None):
    # Get current directory name
    this_file_path = os.path.abspath(inspect.getfile(inspect.currentframe()))
    this_directory = os.path.dirname(this_file_path)

    model_file = model
    if not model_file.endswith('.yaml'):
        print(f"The input dnn model '{model_file}' is expected to be a yaml file")
        sys.exit(0)

    if bitwidth_setting == 'uniform' and uniform_width_set is None:
        print('The uniform-width argument must be set when using the uniform bitwidth option')
        sys.exit(0)
    elif bitwidth_setting != 'uniform' and uniform_width_set is not None:
        print('The uniform bitwidth option must be chosen if using the uniform-width argument')
        sys.exit(0)
    elif bitwidth_setting == 'uniform' and uniform_width_set is not None and len(uniform_width_set) != 3:
        print('The uniform-width argument should have three values set for the individual tensors, i.e. 8,4,8 for Inputs, Weights, Outputs')
        sys.exit(0)

    if bitwidth_setting == 'non-uniform' and non_uniform_width_set is None:
        print('The non-uniform-width argument must be set when using the non-uniform bitwidth option')
        sys.exit(0)
    elif bitwidth_setting != 'non-uniform' and non_uniform_width_set is not None:
        print('The non-uniform bitwidth option must be chosen if using the non-uniform-width argument')
        sys.exit(0)
    elif bitwidth_setting == 'non-uniform' and (non_uniform_width_set is not None and not isinstance(non_uniform_width_set, dict)):
        print('The non-uniform-width argument must be a dictionary')
        sys.exit(0)

    # Construct appropriate folder and file paths
    if not os.path.exists(out_dir):
        create_folder(out_dir)
    config_abspath = os.path.join(this_directory, 'temps/sample.yaml')

    # Just test that path points to a valid config file.
    with open(config_abspath, "r") as f:
        yaml.load(f, Loader=yaml.SafeLoader)

    # Load the model from the YAML file
    with open((model_file), 'r') as f:
        model = yaml.load(f, Loader=yaml.SafeLoader)

    # Check if the number of layer and the number of desired non-uniform bitwidths to be applied match
    if bitwidth_setting == 'non-uniform' and len(non_uniform_width_set) != len(model['layers']):
        print('The number of layers in the model and the number of non-uniform bitwidths to be applied must match')
        sys.exit(0)

    # Construct problem shapes for each layer
    if bitwidth_setting == 'non-uniform':
        workload_keys = list(non_uniform_width_set.keys())
    non_uniform_width = {}

    for i, layer in enumerate(model['layers']):
        problem = layer
        file_name = out_file + '_' + 'layer' + str(i+1) + '.yaml'
        file_path = os.path.abspath(os.path.join(out_dir, file_name))
        if bitwidth_setting == 'non-uniform':
            non_uniform_width = non_uniform_width_set[workload_keys[i]]
        rewrite_workload_bounds(src=config_abspath, dst=file_path, workload_bounds=problem, verbose=verbose, bitwidth_setting=bitwidth_setting, uniform_width=uniform_width_set, non_uniform_width=non_uniform_width)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter, prog="construct_workloads", description="Constructor of Timeloop layer workloads for desired model")
    parser.add_argument('-m', '--model', type=str, required=True, help='path to a yaml file containing model layer descriptions (look within the parsed_models directory)')
    parser.add_argument('-b', '--bitwidth', type=str, default="native", choices=["native", "uniform", "non-uniform"], help='choice of data tensor bitwidths')
    parser.add_argument('-u', '--uniform-width', type=lambda x: tuple(map(int, x.split(','))), help='uniform bitwidth option, a comma-separated tuple of 3 input sizes for the individual tensors (Inputs, Weights, Outputs), e.g 8,4,8 or "8,4,8"')
    parser.add_argument('-n', '--non-uniform-width', type=json_file_to_dict, help='non-uniform bitwidth option, path to a JSON file representing non-uniform bitwidths for each layer\n' + \
                        'For example: \'{"layer_1": {"Inputs": 8, "Weights": 4, "Outputs": 8}, "layer_2": {"Inputs": 5, "Weights": 2, "Outputs": 3}}\'')
    parser.add_argument('-O', '--outdir', type=str, default=f"workload_shapes", help='output directory')
    parser.add_argument('-o', '--outfile', type=str, default=f"model", help='output workload name')
    parser.add_argument('-v', '--verbose', default=False, action='store_true')

    # Parse arguments
    opt = parser.parse_args()
    construct_workloads(opt.model, opt.bitwidth, opt.uniform_width, opt.non_uniform_width, opt.verbose, opt.outfile, opt.outdir)
