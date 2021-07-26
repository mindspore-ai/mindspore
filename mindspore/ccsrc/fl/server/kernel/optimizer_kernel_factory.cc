/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "fl/server/kernel/optimizer_kernel_factory.h"
#include <utility>

namespace mindspore {
namespace fl {
namespace server {
namespace kernel {
bool OptimizerKernelFactory::Matched(const ParamsInfo &params_info, const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  std::string cnode_name = AnfAlgo::GetCNodeName(kernel_node);
  if (kNameToIdxMap.count(cnode_name) == 0) {
    MS_LOG(ERROR) << "Can't find index info for kernel " << cnode_name;
    return false;
  }

  auto input_name_to_idx = kNameToIdxMap.at(cnode_name).at("inputs");
  size_t input_num = params_info.inputs_num();
  for (size_t i = 0; i < input_num; i++) {
    auto one_input_name_type = params_info.inputs_name_type(i);
    std::string name = one_input_name_type.first;
    if (input_name_to_idx.count(name) == 0) {
      MS_LOG(EXCEPTION) << cnode_name << " does not have input named " << name;
      return false;
    }
    size_t input_idx = input_name_to_idx.at(name);
    TypeId kernel_node_input_type = AnfAlgo::GetPrevNodeOutputInferDataType(kernel_node, input_idx);
    TypeId registered_input_type = one_input_name_type.second;
    if (registered_input_type != kernel_node_input_type) {
      return false;
    }
  }

  auto output_name_to_idx = kNameToIdxMap.at(cnode_name).at("outputs");
  size_t output_num = params_info.outputs_num();
  for (size_t i = 0; i < output_num; i++) {
    auto one_output_name_type = params_info.outputs_name_type(i);
    std::string name = one_output_name_type.first;
    if (output_name_to_idx.count(name) == 0) {
      MS_LOG(EXCEPTION) << cnode_name << " does not have output named " << name;
      return false;
    }
    size_t output_idx = output_name_to_idx.at(name);
    TypeId kernel_node_output_type = AnfAlgo::GetOutputInferDataType(kernel_node, output_idx);
    TypeId registered_output_type = one_output_name_type.second;
    if (registered_output_type != kernel_node_output_type) {
      return false;
    }
  }
  return true;
}
}  // namespace kernel
}  // namespace server
}  // namespace fl
}  // namespace mindspore
