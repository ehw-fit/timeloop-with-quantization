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

import os
import inspect
import sys

def prod (l):
    return functools.reduce(lambda x, y: x * y, l)

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
        config = yaml.load(f, Loader = yaml.SafeLoader)

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
        config['problem']['instance']['commonBitwidth'] = uniform_width
    elif bitwidth_setting == 'non-uniform':
        bitwidths_dict = {
        'Inputs': non_uniform_width[0],
        'Weights': non_uniform_width[1],
        'Outputs': non_uniform_width[2]
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


if __name__=="__main__":
    parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter, description="Constructor of Timeloop layer workloads for desired model")
    parser.add_argument('-m', '--model', type=str, required=True, help='yaml file containing model layer descriptions, EXPECTED TO BE INSIDE parsed_models folder')
    parser.add_argument('-b', '--bitwidth', type=str, default="native", choices=["native", "uniform", "non-uniform"], help='choice of data tensor bitwidths')
    parser.add_argument('-u', '--uniform-size', type=int, help='integer input size for uniform bitwidth option')
    parser.add_argument('-n', '--non-uniform-size', type=lambda x: tuple(map(int, x.split(','))), help='comma-separated tuple of 3 input sizes for the individual tensors (Inputs, Weights, Outputs), e.g 8,4,8 or "8,4,8"')
    parser.add_argument('-O', '--outdir', type=str, default=f"workload_shapes", help='output directory')
    parser.add_argument('-o', '--outfile', type=str, default=f"model", help='output workload name')
    parser.add_argument('-v', '--verbose', default=False, action='store_true')
    
    # Parse arguments   
    opt = parser.parse_args()
    
    # Get current directory name
    this_file_path = os.path.abspath(inspect.getfile(inspect.currentframe()))
    this_directory = os.path.dirname(this_file_path)

    model_file = opt.model
    if not model_file.endswith('.yaml'):
        print(f"The input dnn model '{model_file}' is expected to be a yaml file")
        sys.exit(0)
    
        
    if opt.bitwidth == 'uniform' and opt.uniform_size is None:
        print('The uniform-size argument must be set when using the uniform bitwidth option')
        sys.exit(0)
    elif opt.bitwidth != 'uniform' and opt.uniform_size is not None:
        print('The uniform bitwidth option must be chosen if using the uniform-size argument')
        sys.exit(0)
    
    if opt.bitwidth == 'non-uniform' and (opt.non_uniform_size is None or len(opt.non_uniform_size) != 3):
        print('The non-uniform-size argument must be set when using the non-uniform bitwidth option')
        sys.exit(0)
    elif opt.bitwidth != 'non-uniform' and opt.non_uniform_size is not None:
        print('The non-uniform bitwidth option must be chosen if using the non-uniform-size argument')
        sys.exit(0)
        
    


    # Construct appropriate folder and file paths
    if not os.path.exists(opt.outdir):
        os.makedirs(opt.outdir)
    create_folder(os.path.abspath(os.path.join(this_directory, opt.outdir)))
    config_abspath = os.path.join(this_directory, 'temps/sample.yaml')

    # Just test that path points to a valid config file.
    with open(config_abspath, "r") as f:
        yaml.load(f, Loader=yaml.SafeLoader)
        
    # Load the model from the YAML file
    in_dir = "parsed_models"
    with open(os.path.join(in_dir, model_file), 'r') as f:
        model = yaml.load(f, Loader=yaml.SafeLoader)
    
    # Construct problem shapes for each layer
    for i, layer in enumerate(model['layers']):
        problem = layer
        file_name = opt.outfile + '_' + 'layer' + str(i+1) + '.yaml'
        file_path = os.path.abspath(os.path.join(this_directory, opt.outdir, file_name))
        rewrite_workload_bounds(config_abspath, file_path, problem, opt.verbose, opt.bitwidth, opt.uniform_size, opt.non_uniform_size)
