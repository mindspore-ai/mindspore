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
#include "tools/optimizer/graph/inputs_adjust_pass.h"
#include <vector>
#include <string>
#include <memory>
#include "src/ops/primitive_c.h"

namespace mindspore::opt {
STATUS InputAdjustPass::AddAttrToInput(const FuncGraphPtr &func_graph, const CNodePtr &cnode, int input_num,
                                       const std::string &attr_name, int flag) {
  MS_ASSERT(cnode != nullptr);
  if (!CheckInputs(cnode)) {
    MS_LOG(ERROR) << "input is invalid.";
    return lite::RET_INPUT_TENSOR_ERROR;
  }
  auto primitive_c = GetValueNode<PrimitiveCPtr>(cnode->input(0));
  auto value_ptr = primitive_c->GetAttr(attr_name);
  if (value_ptr == nullptr) {
    MS_LOG(DEBUG) << "there is no attr :" << attr_name;
    return lite::RET_NO_CHANGE;
  }
  auto inputs = cnode->inputs();
  if (static_cast<int>(inputs.size()) > input_num) {
    primitive_c->EraseAttr(attr_name);
    MS_LOG(DEBUG) << "input num has been meet, which is " << inputs.size();
    return lite::RET_OK;
  } else if (static_cast<int>(inputs.size()) < input_num) {
    MS_LOG(ERROR) << "input num is invalid.";
    return lite::RET_ERROR;
  }
  switch (flag) {
    case 1: {
      auto value_data = GetValue<int32_t>(value_ptr);
      auto param_node =
        BuildIntValueParameterNode(func_graph, value_data, cnode->fullname_with_scope() + "_" + attr_name);
      inputs.push_back(param_node);
      break;
    }
    case 2: {
      auto value_data = GetValue<std::vector<int32_t>>(value_ptr);
      auto param_node =
        BuildIntVecParameterNode(func_graph, value_data, cnode->fullname_with_scope() + "_" + attr_name);
      inputs.push_back(param_node);
      break;
    }
    case 3: {
      auto value_data = GetValue<std::vector<std::vector<int32_t>>>(value_ptr);
      auto param_node =
        BuildIntVec2DParameterNode(func_graph, value_data, cnode->fullname_with_scope() + "_" + attr_name);
      inputs.push_back(param_node);
      break;
    }
    case 4: {
      auto value_data = GetValue<float>(value_ptr);
      auto param_node =
        BuildFloatValueParameterNode(func_graph, value_data, cnode->fullname_with_scope() + "_" + attr_name);
      inputs.push_back(param_node);
      break;
    }
    default: {
      MS_LOG(ERROR) << "Error attr flag";
      return lite::RET_ERROR;
    }
  }
  cnode->set_inputs(inputs);

  return lite::RET_OK;
}

bool InputAdjustPass::Run(const FuncGraphPtr &func_graph) {
  MS_ASSERT(func_graph != nullptr);
  auto manager = Manage(func_graph, true);
  if (manager == nullptr) {
    MS_LOG(ERROR) << "manager is nullptr.";
    return lite::RET_NULL_PTR;
  }
  auto node_list = TopoSort(func_graph->get_return());
  STATUS status = lite::RET_OK;
  for (auto &node : node_list) {
    auto cnode = node->cast<CNodePtr>();
    if (cnode == nullptr) {
      continue;
    }

    if (GetCNodeType(node) == schema::PrimitiveType_Resize) {
      status = AddAttrToInput(func_graph, cnode, 2, "zoom_factor", 1);
    }
    if (status != lite::RET_OK && status != lite::RET_NO_CHANGE) {
      MS_LOG(ERROR) << "adjust input pass is failed.";
      return false;
    }
  }
  return true;
}
}  // namespace mindspore::opt
