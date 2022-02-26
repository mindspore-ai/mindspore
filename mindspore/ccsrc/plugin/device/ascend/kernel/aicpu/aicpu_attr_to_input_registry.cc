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

#include "plugin/device/ascend/kernel/aicpu/aicpu_attr_to_input_registry.h"

#include "backend/common/session/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "base/core_ops.h"

namespace mindspore {
namespace kernel {
/*
 * Parameter is input in AICPU, but is attribute in TBE.
 * {
 *   {op_name, {{attr_name, pos_index}, ...},
 *   ...
 * }
 */
std::map<string, std::vector<std::pair<string, size_t>>> AicpuOpAttrToInputMap = {
  {prim::kPrimOneHot->name(), {{"depth", 1}}}};

bool GetAicpuOpAttrToInputInfo(const CNodePtr &kernel_node, std::vector<std::pair<string, size_t>> *info) {
  std::string op_name = common::AnfAlgo::GetCNodeName(kernel_node);
  if (AicpuOpAttrToInputMap.find(op_name) == AicpuOpAttrToInputMap.end()) {
    return false;
  } else {
    *info = AicpuOpAttrToInputMap[op_name];
    return true;
  }
}
}  // namespace kernel
}  // namespace mindspore
