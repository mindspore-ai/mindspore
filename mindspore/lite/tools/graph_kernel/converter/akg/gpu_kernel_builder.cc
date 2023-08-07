/**
 * Copyright 2023 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "tools/graph_kernel/converter/akg/gpu_kernel_builder.h"
#include <vector>
#include <memory>
#include <fstream>
#include <set>
#include "utils/anf_utils.h"
#include "kernel/graph_kernel/graph_kernel_json_flags.h"
#include "tools/graph_kernel/converter/akg/utils.h"
#include "tools/graph_kernel/converter/akg/cpu_kernel_builder.h"
#include "nlohmann/json.hpp"
using json = nlohmann::json;

namespace mindspore::graphkernel {
const int C0 = 0;
const int C1 = 1;
const int C2 = 2;
const int C3 = 3;
const int C4 = 4;
const int C5 = 5;
bool GpuKernelBuilder::CompileJsonsInAnfnodes(const AnfNodePtrList &node_list) {
  std::set<std::string> uniq_info_names;
  dir_path_ =
    SaveNodesInfo(node_list, "./akg_kernel_meta", AkgKernelBuilder::json_option(), &node_info_map_, &uniq_info_names);
  if (dir_path_.empty()) {
    return false;
  }
  auto ret = CompileJsonsInList(dir_path_, std::vector<std::string>(uniq_info_names.begin(), uniq_info_names.end()));
  return ret;
}

std::vector<std::string> GpuKernelBuilder::ReadThreadBlockFromJson(const std::string &dir_name) {
  std::ifstream ifile(dir_name);
  json json_info;
  ifile >> json_info;
  std::vector<std::string> thread_block_info;
  thread_block_info.push_back(std::to_string(static_cast<int>(json_info.at("blockIdx.x"))));
  thread_block_info.push_back(std::to_string(static_cast<int>(json_info.at("blockIdx.y"))));
  thread_block_info.push_back(std::to_string(static_cast<int>(json_info.at("blockIdx.z"))));
  thread_block_info.push_back(std::to_string(static_cast<int>(json_info.at("threadIdx.x"))));
  thread_block_info.push_back(std::to_string(static_cast<int>(json_info.at("threadIdx.y"))));
  thread_block_info.push_back(std::to_string(static_cast<int>(json_info.at("threadIdx.z"))));
  return thread_block_info;
}

AnfNodePtr GpuKernelBuilder::CreateCustomOp(const FuncGraphPtr &func_graph, const CNodePtr &cnode) {
  auto op = std::make_shared<ops::Custom>();
  op->set_type("GraphKernel");
  auto inputs = cnode->inputs();
  auto prim = op->GetPrim();
  inputs[0] = NewValueNode(prim);
  auto custom_cnode = func_graph->NewCNode(inputs);
  custom_cnode->set_fullname_with_scope(cnode->fullname_with_scope());
  custom_cnode->set_abstract(cnode->abstract()->Clone());
  prim->set_attr("func_type", MakeValue("custom_akg_gpu"));
  prim->set_attr("unique_name", MakeValue("CustomAkgGpu"));

  // set attrs
  std::map<std::string, std::vector<uint8_t>> custom_attrs;
  auto kernel_name = node_info_map_[cnode->cast<AnfNodePtr>()];
  custom_attrs["kernel_name"] = std::vector<uint8_t>(kernel_name.begin(), kernel_name.end());

  std::string json_path = dir_path_ + "/" + kernel_name + ".json";
  const std::vector<std::string> thread_block_info = ReadThreadBlockFromJson(json_path);
  std::string ptx_path = dir_path_ + "/" + kernel_name + ".ptx";
  std::string output_shape_str = GetCNodeOutputShapeStr(cnode);
  std::string output_format_str = GetCNodeOutputFormatStr(cnode);
  std::string output_type_str = GetCNodeOutputTypeStr(cnode);
  custom_attrs["outputs_shape"] = std::vector<uint8_t>(output_shape_str.begin(), output_shape_str.end());
  custom_attrs["outputs_format"] = std::vector<uint8_t>(output_format_str.begin(), output_format_str.end());
  custom_attrs["outputs_type"] = std::vector<uint8_t>(output_type_str.begin(), output_type_str.end());
  custom_attrs["GridDimX"] = std::vector<uint8_t>(thread_block_info[C0].begin(), thread_block_info[C0].end());
  custom_attrs["GridDimY"] = std::vector<uint8_t>(thread_block_info[C1].begin(), thread_block_info[C1].end());
  custom_attrs["GridDimZ"] = std::vector<uint8_t>(thread_block_info[C2].begin(), thread_block_info[C2].end());
  custom_attrs["BlockDimX"] = std::vector<uint8_t>(thread_block_info[C3].begin(), thread_block_info[C3].end());
  custom_attrs["BlockDimY"] = std::vector<uint8_t>(thread_block_info[C4].begin(), thread_block_info[C4].end());
  custom_attrs["BlockDimZ"] = std::vector<uint8_t>(thread_block_info[C5].begin(), thread_block_info[C5].end());
  custom_attrs["ptx_path"] = std::vector<uint8_t>(ptx_path.begin(), ptx_path.end());
  std::string info_path = dir_path_ + "/" + kernel_name + ".info";
  std::ifstream ifile(info_path);
  if (ifile.fail()) {
    MS_LOG(ERROR) << "Can not find info at: " << json_path;
    return nullptr;
  }
  json json_info;
  ifile >> json_info;
  auto process = json_info.at(kJsonKeyProcess).get<string>();
  custom_attrs[kJsonKeyProcess] = std::vector<uint8_t>(process.begin(), process.end());
  if (json_info.find(kJsonKeyTargetInfo) != json_info.end()) {
    auto target_info = json_info.at(kJsonKeyTargetInfo);
    auto compute_capability = target_info.at(kJsonKeyComputeCapability).get<string>();
    custom_attrs[kJsonKeyComputeCapability] =
      std::vector<uint8_t>(compute_capability.begin(), compute_capability.end());
    auto sm_count = target_info.at(kJsonKeySmCount).get<int>();
    auto sm_count_str = std::to_string(sm_count);
    custom_attrs[kJsonKeySmCount] = std::vector<uint8_t>(sm_count_str.begin(), sm_count_str.end());
  }
  op->set_attr(custom_attrs);
  return custom_cnode;
}
}  // namespace mindspore::graphkernel
