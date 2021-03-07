/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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
#include "tools/optimizer/graph/onnx_inputs_adjust_pass.h"
#include <algorithm>
#include <vector>
#include <string>
#include <functional>
#include <memory>
#include "ops/fusion/conv2d_fusion.h"
#include "ops/fusion/conv2d_transpose_fusion.h"
#include "ops/resize.h"
#include "include/errorcode.h"

namespace mindspore::opt {
STATUS OnnxInputAdjustOpPass::AddAttrToInput(const FuncGraphPtr &func_graph, const CNodePtr &cnode, int input_num,
                                             const std::string &attr_name) {
  MS_ASSERT(cnode != nullptr);
  if (!CheckInputs(cnode)) {
    MS_LOG(ERROR) << "input is invalid.";
    return lite::RET_INPUT_TENSOR_ERROR;
  }
  auto primitive_c = GetValueNode<PrimitiveCPtr>(cnode->input(0));
  MS_LOG(INFO) << "supplement " << attr_name << " attr to input";
  auto value_ptr = primitive_c->GetAttr(attr_name);
  auto inputs = cnode->inputs();
  if (static_cast<int>(inputs.size()) > input_num) {
    if (value_ptr != nullptr) {
      primitive_c->EraseAttr(attr_name);
    }
    MS_LOG(DEBUG) << "input num has been meet, which is " << inputs.size();
    return lite::RET_OK;
  } else if (static_cast<int>(inputs.size()) < input_num) {
    MS_LOG(ERROR) << "input num is invalid.";
    return lite::RET_ERROR;
  }
  if (value_ptr != nullptr) {
    auto value_data = GetValue<std::vector<int32_t>>(value_ptr);
    auto param_node = BuildIntVecParameterNode(func_graph, value_data, cnode->fullname_with_scope() + "_" + attr_name);
    inputs.push_back(param_node);
    cnode->set_inputs(inputs);
    primitive_c->EraseAttr(attr_name);
  } else {
    MS_LOG(ERROR) << "there is no attr :" << attr_name;
    return lite::RET_ERROR;
  }

  return lite::RET_OK;
}

STATUS OnnxInputAdjustOpPass::ReplaceInt64ParameterNode(const FuncGraphPtr &func_graph,
                                                        const ParameterPtr &param_node) {
  MS_ASSERT(func_graph != nullptr);
  MS_ASSERT(param_node != nullptr);
  if (param_node->abstract() == nullptr) {
    MS_LOG(ERROR) << "parameter node abstract is invalid.";
    return lite::RET_NULL_PTR;
  }
  auto abstract_tensor = param_node->abstract()->cast<abstract::AbstractTensorPtr>();
  if (abstract_tensor == nullptr) {
    MS_LOG(ERROR) << "param node has no abstract tensor.";
    return lite::RET_NULL_PTR;
  }
  if (abstract_tensor->element() == nullptr || abstract_tensor->element()->GetTypeTrack() == nullptr) {
    MS_LOG(ERROR) << "get typePtr failed.";
    return lite::RET_NULL_PTR;
  }
  if (abstract_tensor->element()->GetTypeTrack()->type_id() != kNumberTypeInt64) {
    MS_LOG(DEBUG) << "don't need to convert to int32.";
    return lite::RET_OK;
  }
  auto manager = func_graph->manager();
  MS_ASSERT(manager != nullptr);
  if (param_node->has_default()) {
    auto default_value = param_node->default_param();
    if (default_value == nullptr) {
      MS_LOG(ERROR) << "default data is nullptr.";
      return lite::RET_NULL_PTR;
    }
    auto param_value = default_value->cast<ParamValueLitePtr>();
    if (param_value == nullptr) {
      MS_LOG(ERROR) << "default data is not paramvaluelite.";
      return lite::RET_NULL_PTR;
    }
    auto param_node_new = BuildParameterNode(func_graph, param_node, param_value);
    manager->Replace(param_node, param_node_new);
  } else {
    // set graph input
    param_node->abstract()->set_type(TypeIdToType(kNumberTypeInt32));
  }
  return lite::RET_OK;
}

STATUS OnnxInputAdjustOpPass::AdjustResize(const CNodePtr &cnode) {
  MS_ASSERT(cnode != nullptr);
  auto node = cnode->input(0);
  MS_ASSERT(node != nullptr);
  auto resize_prim = GetValueNode<std::shared_ptr<ops::Resize>>(node);
  if (resize_prim == nullptr) {
    MS_LOG(ERROR) << "cnode is invalid.";
    return lite::RET_ERROR;
  }
  if (resize_prim->GetAttr(ops::kCoordinateTransformMode) == nullptr) {
    return lite::RET_OK;
  }
  if (cnode->inputs().size() > 4 && resize_prim->get_coordinate_transform_mode() == mindspore::HALF_PIXEL) {
    std::vector<AnfNodePtr> new_resize_inputs;
    new_resize_inputs.push_back(cnode->inputs()[0]);
    new_resize_inputs.push_back(cnode->inputs()[1]);
    new_resize_inputs.push_back(cnode->inputs()[4]);
    cnode->set_inputs(new_resize_inputs);
  } else if (cnode->inputs().size() == 4) {
    auto new_input = cnode->inputs();
    new_input.erase(new_input.begin() + 2);
    cnode->set_inputs(new_input);
  }
  return lite::RET_OK;
}

STATUS OnnxInputAdjustOpPass::ReplaceConstant(const FuncGraphPtr &func_graph, const CNodePtr &cnode) {
  MS_ASSERT(func_graph != nullptr);
  MS_ASSERT(cnode != nullptr);
  if (cnode->inputs().empty() || cnode->input(0) == nullptr) {
    MS_LOG(ERROR) << "constant cnode has no primitive.";
    return lite::RET_ERROR;
  }
  auto value_node = cnode->input(0)->cast<ValueNodePtr>();
  if (value_node == nullptr) {
    MS_LOG(ERROR) << "constant input0 is not valuenode.";
    return lite::RET_ERROR;
  }
  auto value_ptr = value_node->value();
  if (value_ptr == nullptr) {
    MS_LOG(ERROR) << "value node has no value.";
    return lite::RET_ERROR;
  }
  auto primitive_c = value_ptr->cast<PrimitiveCPtr>();
  if (primitive_c == nullptr) {
    MS_LOG(ERROR) << "value is not primitive_c.";
    return lite::RET_ERROR;
  }
  auto param_value = primitive_c->GetAttr("const_data");
  if (param_value == nullptr) {
    MS_LOG(ERROR) << "constant cnode has no data.";
    return lite::RET_ERROR;
  }
  auto param_value_lite = param_value->cast<ParamValueLitePtr>();
  if (param_value_lite == nullptr) {
    MS_LOG(ERROR) << "valueptr is not paramvalueliteptr.";
    return lite::RET_ERROR;
  }
  auto param_node = BuildParameterNode(func_graph, cnode, param_value_lite);
  if (param_node == nullptr) {
    MS_LOG(ERROR) << "convert constant to param node failed.";
    return lite::RET_ERROR;
  }
  auto manager = func_graph->manager();
  MS_ASSERT(manager != nullptr);
  manager->Replace(cnode, param_node);
  return lite::RET_OK;
}

STATUS OnnxInputAdjustOpPass::ReplaceTransposeWithGraphInput(const FuncGraphPtr &func_graph, const CNodePtr &cnode) {
  MS_ASSERT(func_graph != nullptr);
  MS_ASSERT(cnode != nullptr);
  if (cnode->inputs().size() != 3) {
    MS_LOG(ERROR) << "onnx transpose input size is 2, now is " << cnode->inputs().size() - 1;
    return lite::RET_ERROR;
  }
  auto anf_node = cnode->input(1);
  MS_ASSERT(anf_node != nullptr);
  auto param_node = anf_node->cast<ParameterPtr>();
  if (param_node == nullptr || param_node->has_default()) {
    MS_LOG(DEBUG) << "input is not graph input";
    return lite::RET_OK;
  }
  MS_ASSERT(param_node->abstract() != nullptr && param_node->abstract()->GetShapeTrack() != nullptr);
  auto shape_ptr = param_node->abstract()->GetShapeTrack()->cast<abstract::ShapePtr>();
  if (shape_ptr == nullptr) {
    MS_LOG(ERROR) << "shape is nullptr.";
  }
  auto shape_vector = shape_ptr->shape();
  if (shape_vector.size() != 4) {
    MS_LOG(DEBUG) << "only adjust 4 dims graph input.";
    return lite::RET_OK;
  }
  auto perm_anf = cnode->input(2);
  auto perm_param = perm_anf->cast<ParameterPtr>();
  if (perm_param == nullptr || !perm_param->has_default() ||
      !utils::isa<ParamValueLitePtr>(perm_param->default_param())) {
    MS_LOG(DEBUG) << "transpose second input is not parameter node.";
    return lite::RET_OK;
  }
  auto perm_value = perm_param->default_param()->cast<ParamValueLitePtr>();
  if (perm_value->tensor_shape().empty()) {
    MS_LOG(ERROR) << "transpose second input is invalid.";
    return lite::RET_ERROR;
  }
  std::vector<int> perm(perm_value->tensor_shape()[0]);
  if (memcpy_s(perm.data(), perm_value->tensor_size(), perm_value->tensor_addr(), perm_value->tensor_size()) != EOK) {
    MS_LOG(ERROR) << "memcpy data failed.";
    return lite::RET_ERROR;
  }
  std::vector<int> transpose_perm;
  std::transform(perm.begin(), perm.end(), std::back_inserter(transpose_perm),
                 [](const int &val) { return val < 0 ? val + 4 : val; });
  if (transpose_perm[0] == 0 && transpose_perm[1] == 3 && transpose_perm[2] == 1) {
    auto channel = shape_vector[3];
    shape_vector.pop_back();
    shape_vector.insert(shape_vector.begin() + 1, channel);
    param_node->abstract()->set_shape(std::make_shared<abstract::Shape>(shape_vector));
    auto manager = func_graph->manager();
    MS_ASSERT(manager != nullptr);
    manager->Replace(cnode, param_node);
  }
  return lite::RET_OK;
}

STATUS OnnxInputAdjustOpPass::AdjustStridedSlice(const FuncGraphPtr &func_graph, const CNodePtr &cnode) {
  MS_ASSERT(cnode != nullptr);
  if (!CheckInputs(cnode)) {
    MS_LOG(ERROR) << "input is invalid.";
    return lite::RET_INPUT_TENSOR_ERROR;
  }
  if (cnode->inputs().size() == 2) {
    if (AddAttrToInput(func_graph, cnode, 2, "starts") != lite::RET_OK ||
        AddAttrToInput(func_graph, cnode, 3, "ends") != lite::RET_OK ||
        AddAttrToInput(func_graph, cnode, 4, "axes") != lite::RET_OK ||
        AddAttrToInput(func_graph, cnode, 5, "steps") != lite::RET_OK) {
      MS_LOG(ERROR) << "attr to input failed.";
      return lite::RET_ERROR;
    }
  } else if (cnode->inputs().size() <= 3) {
    MS_LOG(ERROR) << "onnx slice's input size need to be >2, now is " << cnode->inputs().size() - 1;
    return lite::RET_INPUT_TENSOR_ERROR;
  }
  int size = 0;
  for (size_t i = 2; i < cnode->inputs().size(); ++i) {
    const auto &param_node = cnode->input(2)->cast<ParameterPtr>();
    if (param_node == nullptr || !param_node->has_default()) {
      continue;
    }
    const auto &default_data = param_node->default_param()->cast<ParamValueLitePtr>();
    if (default_data == nullptr) {
      MS_LOG(ERROR) << "this input is not a paramValueLite.";
      return lite::RET_ERROR;
    }
    auto shape = default_data->tensor_shape();
    size = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
    break;
  }
  auto inputs = cnode->inputs();
  switch (cnode->inputs().size()) {
    case 4: {
      std::vector<int32_t> axes;
      for (int i = 0; i < size; ++i) {
        axes.push_back(i);
      }
      auto new_param_node = BuildIntVecParameterNode(func_graph, axes, cnode->fullname_with_scope() + "_axises");
      if (new_param_node == nullptr) {
        MS_LOG(ERROR) << "new a parameter node failed.";
      }
      inputs.push_back(new_param_node);
    }
    case 5: {
      std::vector<int32_t> steps;
      for (int i = 0; i < size; ++i) {
        steps.push_back(1);
      }
      auto new_param_node = BuildIntVecParameterNode(func_graph, steps, cnode->fullname_with_scope() + "_steps");
      if (new_param_node == nullptr) {
        MS_LOG(ERROR) << "new a parameter node failed.";
      }
      inputs.push_back(new_param_node);
      break;
    }
    default:
      MS_LOG(DEBUG) << "no need to adjust.";
      return lite::RET_NO_CHANGE;
  }
  cnode->set_inputs(inputs);
  return lite::RET_OK;
}

bool OnnxInputAdjustOpPass::Run(const FuncGraphPtr &func_graph) {
  MS_ASSERT(func_graph != nullptr);
  auto manager = Manage(func_graph, true);
  if (manager == nullptr) {
    MS_LOG(ERROR) << "manager is nullptr.";
    return lite::RET_NULL_PTR;
  }
  auto node_list = TopoSort(func_graph->get_return());
  int status = RET_OK;
  for (auto &node : node_list) {
    if (utils::isa<ParameterPtr>(node)) {
      auto param_node = node->cast<ParameterPtr>();
      status = ReplaceInt64ParameterNode(func_graph, param_node);
      if (status != lite::RET_OK) {
        MS_LOG(ERROR) << "replace int64 param node failed.";
        return status;
      }
    }
    auto cnode = node->cast<CNodePtr>();
    if (cnode == nullptr) {
      MS_LOG(DEBUG) << "node is not cnode.";
      continue;
    }
    if (CheckPrimitiveType(node, prim::kPrimConstant)) {
      status = ReplaceConstant(func_graph, cnode);
    } else if (CheckPrimitiveType(node, prim::kPrimTranspose)) {
      status = ReplaceTransposeWithGraphInput(func_graph, cnode);
    } else if (CheckPrimitiveType(node, prim::kPrimStridedSlice)) {
      status = AdjustStridedSlice(func_graph, cnode);
    } else if (CheckPrimitiveType(node, prim::kPrimResize)) {
      status = AdjustResize(cnode);
    } else {
      continue;
    }
    if (status != lite::RET_OK && status != lite::RET_NO_CHANGE) {
      MS_LOG(ERROR) << "adjust input pass is failed.";
      return false;
    }
  }
  return true;
}
}  // namespace mindspore::opt
