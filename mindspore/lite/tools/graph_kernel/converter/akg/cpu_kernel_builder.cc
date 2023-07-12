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

#include "tools/graph_kernel/converter/akg/cpu_kernel_builder.h"

#include <dirent.h>
#include <unistd.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <utility>
#include <algorithm>
#include <set>
#include <map>
#include <string>
#include <vector>
#include <memory>

#include "ir/anf.h"
#include "ir/func_graph.h"
#include "utils/anf_utils.h"
#include "utils/log_adapter.h"
#include "utils/system/env.h"
#include "backend/common/graph_kernel/graph_kernel_flags.h"
#include "tools/graph_kernel/converter/akg/utils.h"
#include "kernel/graph_kernel/graph_kernel_json_flags.h"
#include "nlohmann/json.hpp"
using json = nlohmann::json;

namespace mindspore::graphkernel {
AnfNodePtr CpuKernelBuilder::CreateCustomOp(const FuncGraphPtr &func_graph, const CNodePtr &cnode) {
  auto op = std::make_shared<ops::Custom>();
  op->set_type("GraphKernel");
  std::map<std::string, std::vector<uint8_t>> custom_attrs;
  auto fg = GetCNodeFuncGraph(cnode);
  MS_EXCEPTION_IF_NULL(fg);
  auto kernel_name = GetValue<std::string>(fg->get_attr("kernel_name"));
  std::vector<uint8_t> kernel_name_str(kernel_name.begin(), kernel_name.end());
  custom_attrs["kernel_name"] = kernel_name_str;
  if (GraphKernelFlags::GetInstance().enable_dynamic_batch && fg->has_attr("dynamic_input_index")) {
    std::string dynamic_input_index = GetValue<std::string>(fg->get_attr("dynamic_input_index"));
    custom_attrs["dynamic_input_index"] = std::vector<uint8_t>(dynamic_input_index.begin(), dynamic_input_index.end());
  }
  auto kernel_file_name = kernel_name.substr(0, kernel_name.find("_kernel"));
  auto info_path = dir_path_ + "/" + kernel_file_name + ".info";
  std::ifstream ifile(info_path);
  if (ifile.fail()) {
    MS_LOG(ERROR) << "Can not find info at: " << info_path;
    return nullptr;
  }
  json json_info;
  ifile >> json_info;
  auto process = json_info.at(kJsonKeyProcess).get<string>();
  custom_attrs[kJsonKeyProcess] = std::vector<uint8_t>(process.begin(), process.end());
  auto target_info = json_info.at(kJsonKeyTargetInfo);
  auto arch = target_info.at(kJsonKeyArch).get<std::string>();
  custom_attrs[kJsonKeyArch] = std::vector<uint8_t>(arch.begin(), arch.end());
  auto system = target_info.at(kJsonKeySystem).get<std::string>();
  custom_attrs[kJsonKeySystem] = std::vector<uint8_t>(system.begin(), system.end());
  auto feature = target_info.at(kJsonKeyCpuFeature).get<std::string>();
  custom_attrs[kJsonKeyCpuFeature] = std::vector<uint8_t>(feature.begin(), feature.end());
  std::string input_shape_str = GetCNodeInputShapeStr(cnode);
  std::string output_shape_str = GetCNodeOutputShapeStr(cnode);
  std::string output_format_str = GetCNodeOutputFormatStr(cnode);
  std::string output_type_str = GetCNodeOutputTypeStr(cnode);
  custom_attrs["inputs_shape"] = std::vector<uint8_t>(input_shape_str.begin(), input_shape_str.end());
  custom_attrs["outputs_shape"] = std::vector<uint8_t>(output_shape_str.begin(), output_shape_str.end());
  custom_attrs["outputs_format"] = std::vector<uint8_t>(output_format_str.begin(), output_format_str.end());
  custom_attrs["outputs_type"] = std::vector<uint8_t>(output_type_str.begin(), output_type_str.end());
  op->set_attr(custom_attrs);
  auto inputs = cnode->inputs();
  inputs[0] = NewValueNode(op->GetPrim());
  auto custom_cnode = func_graph->NewCNode(inputs);
  custom_cnode->set_fullname_with_scope(cnode->fullname_with_scope());
  custom_cnode->set_abstract(cnode->abstract()->Clone());
  return custom_cnode;
}

bool CpuKernelBuilder::CompileJsonsInAnfnodes(const AnfNodePtrList &node_list) {
  if (GraphKernelFlags::GetInstance().enable_dynamic_batch) {
    for (auto &node : node_list) {
      auto gk_fg = GetCNodeFuncGraph(node);
      MS_EXCEPTION_IF_NULL(gk_fg);
      std::string dynamic_input_index = GetCNodeDynamicInputIndex(node->cast<CNodePtr>());
      if (!dynamic_input_index.empty()) {
        gk_fg->set_attr("dynamic_input_index", MakeValue(dynamic_input_index));
      }
    }
  }
  std::map<AnfNodePtr, std::string> node_info_map;
  std::set<std::string> uniq_info_names;
  dir_path_ =
    SaveNodesInfo(node_list, "./akg_kernel_meta", AkgKernelBuilder::json_option(), &node_info_map, &uniq_info_names);
  if (dir_path_.empty()) {
    return false;
  }
  ExcludeTunedObj(dir_path_, &uniq_info_names, &node_info_map);
  auto res = CompileJsonsInList(dir_path_, std::vector<std::string>(uniq_info_names.begin(), uniq_info_names.end()));
  if (res) {
    std::set<std::string> obj_files;
    std::ostringstream objs;
    for (const auto &iter : node_info_map) {
      AnfUtils::SetNodeAttr("kernel_name", MakeValue(iter.second + "_kernel"), iter.first);
      if (obj_files.insert(iter.second).second) {
        objs << dir_path_ << "/" << iter.second << ".o ";
      }
    }
    return true;
  }
  return false;
}

bool CpuKernelBuilder::GenerateAkgKernelNodes(const FuncGraphPtr &func_graph, const AnfNodePtr &custom_node,
                                              const CNodePtr &old_cnode) {
  auto fg = GetCNodeFuncGraph(old_cnode);
  auto kernel_name = GetValue<std::string>(fg->get_attr("kernel_name")).append(".so");
  auto real_kernel_name = kernel_name.substr(0, kernel_name.find("_kernel")).append(".so");
  auto param_node = CreateAkgKernelParameter(func_graph, "./akg_kernel_meta/" + real_kernel_name, real_kernel_name);
  if (param_node == nullptr) {
    return false;
  }
  auto manager = Manage(func_graph, true);
  manager->AddEdge(custom_node, param_node);
  return true;
}
}  // namespace mindspore::graphkernel
