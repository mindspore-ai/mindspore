/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include "plugin/device/ascend/kernel/aicpu/aicpu_input_to_attr_registry.h"

#include "include/common/utils/anfalgo.h"
#include "include/common/utils/utils.h"

namespace mindspore {
namespace kernel {
/*
 * Parameter is attr in AICPU, but is input in graph.
 * {
 *   {op_name, {{pos_index， data_type}, ...},
 *   ...
 * }
 */
std::map<string, std::map<size_t, std::string>> AicpuOpInputToAttrMap = {
  {kStridedSliceOpName, {{1, "listInt"}, {2, "listInt"}, {3, "listInt"}}}, {kExpandDimsOpName, {{1, "int"}}}};

bool GetAicpuOpInputToAttrInfo(const CNodePtr &kernel_node, std::map<size_t, std::string> *input_to_attr_info) {
  std::string op_name = common::AnfAlgo::GetCNodeName(kernel_node);
  if (AicpuOpInputToAttrMap.find(op_name) == AicpuOpInputToAttrMap.end()) {
    return false;
  } else {
    *input_to_attr_info = AicpuOpInputToAttrMap[op_name];
    return true;
  }
}
}  // namespace kernel
}  // namespace mindspore
