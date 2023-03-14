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
#include "tools/graph_kernel/converter/parameter_to_tensor.h"

namespace mindspore::graphkernel {
bool ParameterToTensor::Run(const FuncGraphPtr &func_graph) {
  auto todos = TopoSort(func_graph->output());
  for (auto &node : todos) {
    auto cnode = node->cast<CNodePtr>();
    if (cnode == nullptr) {
      continue;
    }

    for (size_t idx = 1; idx < cnode->size(); idx++) {
      if (cnode->input(idx)->isa<Parameter>()) {
        auto default_param = cnode->input(idx)->cast<ParameterPtr>()->default_param();
        if (default_param == nullptr) {
          continue;
        }
        auto param_value = default_param->cast<tensor::TensorPtr>();
        if (param_value == nullptr) {
          continue;
        }
        auto value = NewValueNode(param_value);
        value->set_abstract(param_value->ToAbstract());
        cnode->set_input(idx, value);
      }
    }
  }
  return true;
}
}  // namespace mindspore::graphkernel
