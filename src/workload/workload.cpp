/* Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <string>
#include <cstring>
#include <fstream>

#include "workload/shape-models/problem-shape.hpp"
#include "workload/workload.hpp"

namespace problem
{

const Shape* GetShape()
{
  return Workload::current_shape_;
}

// ======================================== //
//                 Workload                 //
// ======================================== //

bool Workload::workload_alive_ = false;
const Shape* Workload::current_shape_ = nullptr;

std::string ShapeFileName(const std::string shape_name)
{
  std::string shape_file_path;

  // Prepare list of dir paths where problem shapes may be located.
  std::list<std::string> shape_dir_list;

  const char* shape_dir_env = std::getenv("TIMELOOP_PROBLEM_SHAPE_DIR");
  if (shape_dir_env)
  {
    shape_dir_list.push_back(std::string(shape_dir_env));
  }

  const char* shape_dirs_env = std::getenv("TIMELOOP_PROBLEM_SHAPE_DIRS");
  if (shape_dirs_env)
  {
    std::string shape_dirs_env_cpp = std::string(shape_dirs_env);
    std::stringstream ss(shape_dirs_env_cpp);
    std::string token;
    while (std::getline(ss, token, ':'))
    {
      shape_dir_list.push_back(token);
    }
  }

  const char* timeloopdir = std::getenv("TIMELOOP_DIR");
  if (!timeloopdir)
  {
    timeloopdir = BUILD_BASE_DIR;
  }
  shape_dir_list.push_back(std::string(timeloopdir) + "/problem-shapes");

  // Append ".cfg" extension if not provided.
  std::list<std::string> shape_file_name_list;
  if (std::strstr(shape_name.c_str(), ".yml") || std::strstr(shape_name.c_str(), ".yaml") || std::strstr(shape_name.c_str(), ".cfg"))
  {
    shape_file_name_list = { shape_name };
  }
  else
  {
    shape_file_name_list = { shape_name + ".yaml", shape_name + ".cfg" };
  }

  bool found = false;
  for (auto& shape_dir: shape_dir_list)
  {
    for (auto& shape_file_name: shape_file_name_list)
    {
      shape_file_path = shape_dir + "/" + shape_file_name;

      // Check if file exists.
      std::ifstream f(shape_file_path);
      if (f.good())
      {
        found = true;
        break;
      }
    }

    if (found)
      break;
  }

  if (!found)
  {
    std::cerr << "ERROR: problem shape files: ";
    for (auto& shape_file_name: shape_file_name_list)
      std::cerr << shape_file_name << " ";
    std::cerr << "not found in problem dir search path:" << std::endl;
    for (auto& shape_dir: shape_dir_list)
      std::cerr << "    " << shape_dir << std::endl;
    exit(1);
  }
  
  std::cerr << "MESSAGE: attempting to read problem shape from file: " << shape_file_path << std::endl;

  return shape_file_path;
}

void ParseWorkload(config::CompoundConfigNode config, Workload& workload)
{
  std::string shape_name;
  if (!config.exists("shape"))
  {
    std::cerr << "WARNING: found neither a problem shape description nor a string corresponding to a to a pre-existing shape description. Assuming shape: cnn_layer." << std::endl;
    config::CompoundConfig shape_config(ShapeFileName("cnn_layer").c_str());
    auto shape = shape_config.getRoot().lookup("shape");
    workload.ParseShape(shape);    
  }
  else if (config.lookupValue("shape", shape_name))
  {    
    config::CompoundConfig shape_config(ShapeFileName(shape_name).c_str());
    auto shape = shape_config.getRoot().lookup("shape");
    workload.ParseShape(shape);    
  }
  else
  {
    auto shape = config.lookup("shape");
    workload.ParseShape(shape);
  }

  // Bounds may be specified directly (backwards-compat) or under a subkey.
  if (config.exists("instance"))
  {
    auto factorized_bounds = config.lookup("instance");
    ParseWorkloadInstance(factorized_bounds, workload);
  }
  else
  {
    ParseWorkloadInstance(config, workload);
  }
}
  
void ParseWorkloadInstance(config::CompoundConfigNode config, Workload& workload)
{
  // Loop bounds for each factorized problem dimension.
  Workload::FactorizedBounds factorized_bounds;
  for (unsigned i = 0; i < GetShape()->NumFactorizedDimensions; i++)
    assert(config.lookupValue(GetShape()->FactorizedDimensionIDToName.at(i),
                              factorized_bounds[i]));
  workload.SetFactorizedBounds(factorized_bounds);

  // Translate into flattened bounds.
  workload.DeriveFlattenedBounds();

  Workload::Coefficients coefficients;
  for (unsigned i = 0; i < GetShape()->NumCoefficients; i++)
  {
    coefficients[i] = GetShape()->DefaultCoefficients.at(i);
    config.lookupValue(GetShape()->CoefficientIDToName.at(i), coefficients[i]);
  }
  workload.SetCoefficients(coefficients);

  Workload::Densities densities;
  std::string density_distribution;

  // shared pointer for parsed density distribution specs
  std::shared_ptr<DensityDistributionSpecs> density_distribution_specs;
  YAML::Node ynode;

  // 1) shared density specification for all dataspaces
  double common_avg_density;
  if (config.exists("commonDensity")){
    config::CompoundConfigNode density_config;
    if (! config.lookup("commonDensity").isMap()){
      config.lookupValue("commonDensity", common_avg_density);
      ynode["distribution"] = "fixed_structured";
      ynode["density"] = common_avg_density;
      density_config = config::CompoundConfigNode(nullptr, ynode, new config::CompoundConfig("dummy.yaml"));
    } else {
      density_config = config.lookup("commonDensity");
    }
    auto density_specs = DensityDistributionFactory::ParseSpecs(density_config);
    // assign all dataspaces the same density value
    for (unsigned i = 0; i < GetShape()->NumDataSpaces; i++){
      densities[i]= DensityDistributionFactory::Construct(density_specs);
      // make sure the density model is correctly set
      assert (densities[i] != NULL);
    }
  }

  // 2) density specifications for each dataspace
  else if (config.exists("densities"))
  {
    auto config_densities = config.lookup("densities");
    for (unsigned i = 0; i < GetShape()->NumDataSpaces; i++){
      double dataspace_avg_density;
      config::CompoundConfigNode density_config;
      std::string dataspace_name = GetShape()->DataSpaceIDToName.at(i);

      if (config_densities.exists(GetShape()->DataSpaceIDToName.at(i)))
      {
		config_densities.lookupValue(GetShape()->DataSpaceIDToName.at(i), dataspace_avg_density);
        
		// if the specific dataspace's density is specified
        if (!config_densities.lookup(GetShape()->DataSpaceIDToName.at(i)).isMap())
        {
          // single number for density is given, default to fixed density distribution
          assert(config_densities.lookupValue(GetShape()->DataSpaceIDToName.at(i), dataspace_avg_density));
          ynode["distribution"] = "fixed_structured";
          ynode["density"] = dataspace_avg_density;
          density_config = config::CompoundConfigNode(nullptr, ynode, new config::CompoundConfig("dummy.yaml"));
        } else
        {
          density_config = config_densities.lookup(GetShape()->DataSpaceIDToName.at(i));
        }
        auto density_specs = DensityDistributionFactory::ParseSpecs(density_config);
        densities[i] = DensityDistributionFactory::Construct(density_specs);
      }
      else
      {
        // no density specified, roll back to default
        ynode["distribution"] = "fixed_structured";
        ynode["density"] = 1.0;
        density_config = config::CompoundConfigNode(nullptr, ynode, new config::CompoundConfig("dummy.yaml"));
        auto density_specs = DensityDistributionFactory::ParseSpecs(density_config);
        densities[i]= DensityDistributionFactory::Construct(density_specs);
      }

      // make sure the density model is correctly set
      assert (densities[i] != NULL);
    }

    // 3) no density specification -> dense workload tensors
  } else {
    config::CompoundConfigNode density_config;
    for (unsigned i = 0; i < GetShape()->NumDataSpaces; i++){
      ynode["distribution"] = "fixed_structured";
      ynode["density"] = 1.0;
      density_config = config::CompoundConfigNode(nullptr, ynode, new config::CompoundConfig("dummy.yaml"));
      auto density_specs = DensityDistributionFactory::ParseSpecs(density_config);
      densities[i]= DensityDistributionFactory::Construct(density_specs);

      // make sure the density model is correctly set
      assert (densities[i] != NULL);
    }
  }
  workload.SetDensities(densities);

  /* Quantization addition */
  Workload::Bitwidths bitwidths;
  Workload::BitwidthsSpecified bitwidths_specified;

  // 1) Shared bitwidth specification for all dataspaces (specified using "commonBitwidth" or "commonDatawidth")
  unsigned common_bitwidth;
  bool common_bw_set = false;
  if (config.exists("commonBitwidth") || config.exists("commonDatawidth") || config.exists("commonWordwidth"))
  {
    std::string bitwidth_key = config.exists("commonBitwidth") ? "commonBitwidth" : (config.exists("commonDatawidth") ? "commonDatawidth" : "commonWordwidth");
    if (! config.lookup(bitwidth_key).isMap())
    {
      config.lookupValue(bitwidth_key, common_bitwidth);
      common_bw_set = true;
    }
    else
    {
      common_bitwidth = 0;
    }
    // Assign the same bitwidth value to all dataspaces
    for (unsigned i = 0; i < GetShape()->NumDataSpaces; i++)
    {
      bitwidths[i] = common_bitwidth;
      bitwidths_specified[i] = common_bw_set == true ? true : false;
    }
  }

  // 2) bitwidth specifications for each dataspace
  else if (config.exists("bitwidths") || config.exists("datawidths") || config.exists("wordwidths"))
  {
    std::string bitwidth_key = config.exists("bitwidths") ? "bitwidths" : (config.exists("datawidths") ? "datawidths" : "wordwidths");
    auto config_bitwidths = config.lookup(bitwidth_key);
    for (unsigned i = 0; i < GetShape()->NumDataSpaces; i++)
    {
      std::string dataspace_name = GetShape()->DataSpaceIDToName.at(i);

      // The dataspace_name exists within the bitwidths specification 
      if (config_bitwidths.exists(dataspace_name))
      {
        // Set the explicit bitwidth specification
		    config_bitwidths.lookupValue(dataspace_name, bitwidths[i]);
        bitwidths_specified[i] = true;
      }
      else
      {
        // No bitwidth specified, roll back to default
        bitwidths[i] = 0;
        bitwidths_specified[i] = false;
      }
    }
  }

  // 3) No bitwidth specification -> assume same bitwidth for all dataspaces
  //    (the data bitwidth will be determined by HW memory specification)
  else
  {
    for (unsigned i = 0; i < GetShape()->NumDataSpaces; i++)
    {
      bitwidths[i] = 0;
      bitwidths_specified[i] = false;
    }
  }

  // Test print if bitwidths are set correctly
  /*
  for (unsigned i = 0; i < GetShape()->NumDataSpaces; i++)
  {
    std::cout << "DataSpace: "<< i << " named: "<< GetShape()->DataSpaceIDToName.at(i) << " Bitwidth: " << bitwidths[i] << " Explicitly set: " << bitwidths_specified[i] << std::endl;
  }
  */
  
  workload.SetBitwidths(bitwidths);
  workload.SetBitwidthsSpecified(bitwidths_specified);
  /*************************/
}

} // namespace problem
