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
#include <set>
#include "utils/anf_utils.h"
#include "tools/graph_kernel/converter/akg/utils.h"
#include "tools/graph_kernel/converter/akg/cpu_kernel_builder.h"

namespace mindspore::graphkernel {
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

AnfNodePtr GpuKernelBuilder::CreateCustomOp(const FuncGraphPtr &func_graph, const CNodePtr &cnode) {
  auto op = std::make_shared<ops::Custom>();
  op->set_type("GraphKernel");
  auto inputs = cnode->inputs();
  inputs[0] = NewValueNode(op->GetPrim());
  auto custom_cnode = func_graph->NewCNode(inputs);
  custom_cnode->set_fullname_with_scope(cnode->fullname_with_scope());
  custom_cnode->set_abstract(cnode->abstract()->Clone());

  // set attrs
  std::map<std::string, std::vector<uint8_t>> custom_attrs;
  auto kernel_name = node_info_map_[cnode->cast<AnfNodePtr>()];
  custom_attrs["kernel_name"] = std::vector<uint8_t>(kernel_name.begin(), kernel_name.end());
  std::string output_shape_str = GetCNodeOutputShapeStr(cnode);
  std::string output_format_str = GetCNodeOutputFormatStr(cnode);
  std::string output_type_str = GetCNodeOutputTypeStr(cnode);
  custom_attrs["outputs_shape"] = std::vector<uint8_t>(output_shape_str.begin(), output_shape_str.end());
  custom_attrs["outputs_format"] = std::vector<uint8_t>(output_format_str.begin(), output_format_str.end());
  custom_attrs["outputs_type"] = std::vector<uint8_t>(output_type_str.begin(), output_type_str.end());
  op->set_attr(custom_attrs);
  return custom_cnode;
}
}  // namespace mindspore::graphkernel
