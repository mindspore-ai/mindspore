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
#include "tools/converter/parser/onnx/onnx_inputs_adjust.h"
#include <vector>
#include <string>
#include <functional>
#include <algorithm>
#include <memory>
#include "ops/resize.h"
#include "ops/random_normal.h"
#include "include/errorcode.h"
#include "nnacl/op_base.h"
#include "tools/common/tensor_util.h"
#include "tools/optimizer/common/gllo_utils.h"
#include "tools/common/node_util.h"

namespace mindspore::lite {
namespace {
const std::vector<int> kNH2NCPerm = {0, 3, 1, 2};

STATUS AddAttrToInput(const FuncGraphPtr &func_graph, const CNodePtr &cnode, int input_num,
                      const std::string &attr_name) {
  MS_ASSERT(func_graph != nullptr);
  MS_ASSERT(cnode != nullptr);
  MS_CHECK_TRUE_RET(!cnode->inputs().empty(), lite::RET_ERROR);
  if (!opt::CheckInputs(cnode)) {
    MS_LOG(ERROR) << "input is invalid.";
    return lite::RET_INPUT_TENSOR_ERROR;
  }
  auto primitive_c = GetValueNode<PrimitiveCPtr>(cnode->input(0));
  MS_ASSERT(primitive_c != nullptr);
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
    auto param_node =
      opt::BuildIntVecParameterNode(func_graph, value_data, cnode->fullname_with_scope() + "_" + attr_name);
    inputs.push_back(param_node);
    auto manager = func_graph->manager();
    MS_CHECK_TRUE_MSG(manager != nullptr, RET_ERROR, "funcgraph has no manager");
    auto tr = manager->Transact();
    tr.AddEdge(cnode, param_node);
    tr.Commit();
    primitive_c->EraseAttr(attr_name);
  } else {
    MS_LOG(ERROR) << "there is no attr :" << attr_name;
    return lite::RET_ERROR;
  }

  return lite::RET_OK;
}

STATUS ReplaceTypeParameterNode(const FuncGraphPtr &func_graph, const ParameterPtr &param_node, TypeId input,
                                TypeId output) {
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
  if (abstract_tensor->element()->GetTypeTrack()->type_id() != input) {
    MS_LOG(DEBUG) << "The actual type is not the input type, don't need to convert.";
    return lite::RET_OK;
  }
  auto manager = func_graph->manager();
  MS_CHECK_TRUE_MSG(manager != nullptr, RET_ERROR, "funcgraph has no manager");
  if (param_node->has_default()) {
    auto default_value = param_node->default_param();
    MS_ASSERT(default_value != nullptr);
    auto tensor_info = default_value->cast<tensor::TensorPtr>();
    if (tensor_info == nullptr) {
      MS_LOG(ERROR) << "default data is not tensor::Tensor.";
      return lite::RET_NULL_PTR;
    }
    auto param_node_new = opt::BuildParameterNode(func_graph, tensor_info, param_node->fullname_with_scope());
    if (param_node_new == nullptr) {
      MS_LOG(ERROR) << "BuildParameterNode failed.";
      return lite::RET_NULL_PTR;
    }
    if (!manager->Replace(param_node, param_node_new)) {
      MS_LOG(ERROR) << "Replace param node failed.";
      return lite::RET_ERROR;
    }
    func_graph->DropNode(param_node);
  } else {
    // set graph input
    if (abstract_tensor->element()->GetTypeTrack()->type_id() == input) {
      abstract_tensor->element()->set_type(TypeIdToType(output));
    }
  }
  return lite::RET_OK;
}

bool ValidParameterNode(const ParameterPtr &param_node) {
  MS_ASSERT(param_node != nullptr);
  if (!param_node->has_default()) {
    return true;
  }
  auto tensor_info = std::dynamic_pointer_cast<tensor::Tensor>(param_node->default_param());
  MS_ASSERT(tensor_info != nullptr);
  return tensor_info->Size() != 0;
}

STATUS ReplaceConstant(const FuncGraphPtr &func_graph, const CNodePtr &cnode) {
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
  auto tensor_info = primitive_c->GetAttr("const_data");
  if (tensor_info == nullptr) {
    MS_LOG(ERROR) << "constant cnode has no data.";
    return lite::RET_ERROR;
  }
  auto tensor_info_ptr = tensor_info->cast<tensor::TensorPtr>();
  if (tensor_info_ptr == nullptr) {
    MS_LOG(ERROR) << "valueptr is not tensor::Tensorptr.";
    return lite::RET_ERROR;
  }
  auto param_node = opt::BuildParameterNode(func_graph, tensor_info_ptr, cnode->fullname_with_scope());
  if (param_node == nullptr) {
    MS_LOG(ERROR) << "convert constant to param node failed.";
    return lite::RET_ERROR;
  }
  auto manager = func_graph->manager();
  MS_ASSERT(manager != nullptr);
  if (!manager->Replace(cnode, param_node)) {
    MS_LOG(ERROR) << "Replace param node failed.";
    return RET_ERROR;
  }
  return lite::RET_OK;
}

STATUS ReplaceTransposeWithGraphInput(const FuncGraphPtr &func_graph, const CNodePtr &cnode) {
  MS_ASSERT(func_graph != nullptr);
  MS_ASSERT(cnode != nullptr);
  if (cnode->inputs().size() != opt::kInputSizeThree) {
    MS_LOG(ERROR) << "onnx transpose input size should be 2, now is " << (cnode->inputs().size() - 1);
    return lite::RET_ERROR;
  }
  auto anf_node = cnode->input(1);
  MS_CHECK_TRUE_MSG(anf_node != nullptr, lite::RET_ERROR, "cnode's input is a nullptr.");
  auto param_node = anf_node->cast<ParameterPtr>();
  if (param_node == nullptr || param_node->has_default()) {
    MS_LOG(DEBUG) << "input is not graph input";
    return lite::RET_OK;
  }
  MS_ASSERT(param_node->abstract() != nullptr);
  if (param_node->abstract()->GetShapeTrack() == nullptr) {
    MS_LOG(ERROR) << "shape is nullptr.";
    return lite::RET_ERROR;
  }
  auto shape_ptr = param_node->abstract()->GetShapeTrack()->cast<abstract::ShapePtr>();
  if (shape_ptr == nullptr) {
    MS_LOG(ERROR) << "shape is nullptr.";
    return lite::RET_ERROR;
  }
  auto shape_vector = shape_ptr->shape();
  if (shape_vector.size() != opt::kInputSizeFour) {
    MS_LOG(DEBUG) << "only adjust 4 dims graph input.";
    return lite::RET_OK;
  }
  auto perm_anf = cnode->input(opt::kInputIndexTwo);
  MS_ASSERT(perm_anf != nullptr);
  auto perm_param = perm_anf->cast<ParameterPtr>();
  if (perm_param == nullptr || !perm_param->has_default() ||
      !utils::isa<tensor::TensorPtr>(perm_param->default_param())) {
    MS_LOG(DEBUG) << "transpose second input is not parameter node.";
    return lite::RET_OK;
  }
  auto perm_value = perm_param->default_param()->cast<tensor::TensorPtr>();
  MS_ASSERT(perm_value != nullptr);
  if (perm_value->shape().empty()) {
    MS_LOG(ERROR) << "transpose second input is invalid.";
    return lite::RET_ERROR;
  }
  std::vector<int> perm(perm_value->shape()[0]);
  if (memcpy_s(perm.data(), perm_value->shape()[0] * sizeof(int), perm_value->data_c(), perm_value->Size()) != EOK) {
    MS_LOG(ERROR) << "memcpy data failed.";
    return lite::RET_ERROR;
  }
  std::vector<int> transpose_perm;
  std::transform(perm.begin(), perm.end(), std::back_inserter(transpose_perm),
                 [](const int &val) { return val < 0 ? val + 4 : val; });
  if (transpose_perm == kNH2NCPerm) {
    auto channel = shape_vector[opt::kInputIndexThree];
    shape_vector.pop_back();
    shape_vector.insert(shape_vector.begin() + 1, channel);
    param_node->abstract()->set_shape(std::make_shared<abstract::Shape>(shape_vector));
    auto manager = func_graph->manager();
    MS_CHECK_TRUE_MSG(manager != nullptr, RET_ERROR, "funcgraph has no manager");
    if (!manager->Replace(cnode, param_node)) {
      MS_LOG(ERROR) << "Replace param node failed.";
      return RET_ERROR;
    }
  }
  return lite::RET_OK;
}

STATUS AdjustStridedSlice(const FuncGraphPtr &func_graph, const CNodePtr &cnode) {
  MS_ASSERT(func_graph != nullptr);
  auto manager = func_graph->manager();
  MS_ASSERT(cnode != nullptr);
  if (!opt::CheckInputs(cnode)) {
    MS_LOG(ERROR) << "input is invalid.";
    return lite::RET_INPUT_TENSOR_ERROR;
  }
  if (cnode->inputs().size() == opt::kInputSizeTwo) {
    if (AddAttrToInput(func_graph, cnode, opt::kInputIndexTwo, "starts") != lite::RET_OK ||
        AddAttrToInput(func_graph, cnode, opt::kInputIndexThree, "ends") != lite::RET_OK ||
        AddAttrToInput(func_graph, cnode, opt::kInputIndexFour, "axes") != lite::RET_OK ||
        AddAttrToInput(func_graph, cnode, opt::kInputIndexFive, "steps") != lite::RET_OK) {
      MS_LOG(ERROR) << "attr to input failed.";
      return lite::RET_ERROR;
    }
  } else if (cnode->inputs().size() <= opt::kInputSizeThree) {
    MS_LOG(ERROR) << "onnx slice's input size need to be >2, now is " << (cnode->inputs().size() - 1);
    return lite::RET_INPUT_TENSOR_ERROR;
  }
  int size = 1;
  for (size_t i = 2; i < cnode->inputs().size(); ++i) {
    auto param_anf = cnode->input(opt::kInputIndexTwo);
    MS_ASSERT(param_anf != nullptr);
    const auto &param_node = param_anf->cast<ParameterPtr>();
    if (param_node == nullptr || !param_node->has_default()) {
      continue;
    }
    const auto &default_data = param_node->default_param()->cast<tensor::TensorPtr>();
    if (default_data == nullptr) {
      MS_LOG(ERROR) << "this input is not a tensor::Tensor";
      return lite::RET_ERROR;
    }
    auto shape = default_data->shape();
    for (size_t j = 0; j < shape.size(); j++) {
      MS_CHECK_GE(shape.at(j), 0, RET_ERROR);
      MS_CHECK_FALSE_MSG(INT_MUL_OVERFLOW(size, static_cast<int>(shape.at(j))), RET_ERROR, "Int mul overflow.");
      size = size * static_cast<int>(shape.at(j));
    }
    break;
  }
  switch (cnode->inputs().size()) {
    case opt::kInputSizeFour: {
      std::vector<int32_t> axes;
      for (int i = 0; i < size; ++i) {
        axes.push_back(i);
      }
      auto new_param_node = opt::BuildIntVecParameterNode(func_graph, axes, cnode->fullname_with_scope() + "_axises");
      if (new_param_node == nullptr) {
        MS_LOG(ERROR) << "new a parameter node failed.";
      }
      manager->AddEdge(cnode, new_param_node);
      // fall through
    }
    case opt::kInputSizeFive: {
      std::vector<int32_t> steps;
      for (int i = 0; i < size; ++i) {
        steps.push_back(1);
      }
      auto new_param_node = opt::BuildIntVecParameterNode(func_graph, steps, cnode->fullname_with_scope() + "_steps");
      if (new_param_node == nullptr) {
        MS_LOG(ERROR) << "new a parameter node failed.";
        return lite::RET_ERROR;
      }
      manager->AddEdge(cnode, new_param_node);
      break;
    }
    default:
      MS_LOG(DEBUG) << "no need to adjust.";
      return lite::RET_NO_CHANGE;
  }
  return lite::RET_OK;
}

STATUS AdjustResize(bool *need_update_manager, const CNodePtr &cnode) {
  MS_ASSERT(cnode != nullptr);
  MS_CHECK_TRUE_RET(!cnode->inputs().empty(), lite::RET_ERROR);
  auto node = cnode->input(0);
  MS_ASSERT(node != nullptr);
  auto resize_prim = GetValueNode<std::shared_ptr<ops::PrimitiveC>>(node);
  if (resize_prim == nullptr) {
    MS_LOG(ERROR) << "cnode is invalid.";
    return lite::RET_ERROR;
  }
  if (resize_prim->GetAttr(ops::kCoordinateTransformMode) == nullptr) {
    return lite::RET_OK;
  }
  if (cnode->inputs().size() == opt::kInputSizeFour) {
    auto new_input = cnode->inputs();
    new_input.erase(new_input.begin() + opt::kInputIndexTwo);
    cnode->set_inputs(new_input);
    *need_update_manager = true;
  } else if (cnode->inputs().size() > opt::kInputSizeFour) {
    std::vector<AnfNodePtr> new_resize_inputs;
    new_resize_inputs.push_back(cnode->inputs()[0]);
    new_resize_inputs.push_back(cnode->inputs()[1]);

    // remove roi and checkout the scale or size as the third input.
    int shape_index = opt::kInputIndexFour;
    auto scale_node = cnode->inputs()[opt::kInputIndexThree];
    auto size_node = cnode->inputs()[opt::kInputIndexFour];
    MS_ASSERT(scale_node != nullptr);
    MS_ASSERT(size_node != nullptr);
    if (scale_node->isa<CNode>() && size_node->isa<CNode>()) {
      MS_LOG(ERROR) << "One of scale and size should be specified.";
      return lite::RET_ERROR;
    } else if ((scale_node->isa<CNode>() && size_node->isa<Parameter>()) ||
               (scale_node->isa<Parameter>() && size_node->isa<CNode>())) {
      auto param_node =
        scale_node->isa<Parameter>() ? scale_node->cast<ParameterPtr>() : size_node->cast<ParameterPtr>();
      MS_ASSERT(param_node != nullptr);
      if (ValidParameterNode(param_node)) {
        MS_LOG(ERROR) << "One of scale and size should be specified.";
        return lite::RET_ERROR;
      }
      shape_index = scale_node->isa<CNode>() ? opt::kInputIndexThree : opt::kInputIndexFour;
    } else if (scale_node->isa<Parameter>() && size_node->isa<Parameter>()) {
      auto scale_param = scale_node->cast<ParameterPtr>();
      auto size_param = size_node->cast<ParameterPtr>();
      MS_ASSERT(scale_param != nullptr);
      MS_ASSERT(size_param != nullptr);
      bool is_scale_valid = ValidParameterNode(scale_param);
      bool is_size_valid = ValidParameterNode(size_param);
      if (!(is_scale_valid || is_size_valid)) {
        MS_LOG(ERROR) << "One of scale and size should be specified.";
        return lite::RET_ERROR;
      }
      shape_index = is_scale_valid ? opt::kInputIndexThree : opt::kInputIndexFour;
    }
    new_resize_inputs.push_back(cnode->inputs()[shape_index]);
    cnode->set_inputs(new_resize_inputs);
    *need_update_manager = true;
  }
  return lite::RET_OK;
}

STATUS AdjustUnsqueeze(bool *need_update_manager, const CNodePtr &cnode) {
  MS_ASSERT(cnode != nullptr);
  MS_CHECK_TRUE_RET(!cnode->inputs().empty(), lite::RET_ERROR);
  if (cnode->inputs().size() == opt::kInputSizeThree) {
    auto new_input = cnode->inputs();
    new_input.erase(new_input.begin() + opt::kInputIndexTwo);
    cnode->set_inputs(new_input);
    *need_update_manager = true;
  }
  return lite::RET_OK;
}

STATUS AdjustRandomNormal(const FuncGraphPtr &func_graph, const CNodePtr &cnode) {
  MS_ASSERT(func_graph != nullptr && cnode != nullptr);
  if (cnode->size() != 1) {
    return RET_OK;
  }
  auto random_normal_node = ops::GetOperator<ops::RandomNormal>(cnode->input(0));
  MS_CHECK_TRUE_RET(random_normal_node != nullptr, RET_ERROR);
  auto prim = random_normal_node->GetPrim();
  MS_CHECK_TRUE_RET(prim != nullptr, RET_ERROR);
  MS_CHECK_TRUE_RET(prim->GetAttr(ops::kDataType) != nullptr, RET_ERROR);
  TypeId data_type = static_cast<TypeId>(GetValue<int>(prim->GetAttr(ops::kDataType)));
  MS_CHECK_TRUE_RET(prim->GetAttr(ops::kShape) != nullptr, RET_ERROR);
  std::vector<int64_t> shape = GetValue<std::vector<int64_t>>(prim->GetAttr(ops::kShape));
  size_t data_size = abstract::TypeIdSize(data_type);
  for (auto dim : shape) {
    MS_CHECK_INT_MUL_NOT_OVERFLOW(static_cast<int>(data_size), static_cast<int>(dim), RET_ERROR);
    data_size *= static_cast<size_t>(dim);
  }
  MS_CHECK_TRUE_RET(data_size != 0, RET_ERROR);
  auto data = malloc(data_size);
  if (data == nullptr) {
    MS_LOG(ERROR) << "malloc data failed.";
    return RET_ERROR;
  }
  if (memset_s(data, data_size, 0, data_size) != EOK) {
    MS_LOG(ERROR) << "malloc data failed.";
    return RET_ERROR;
  }
  auto tensor_info = CreateTensorInfo(data, data_size, shape, data_type);
  free(data);
  MS_CHECK_TRUE_RET(tensor_info != nullptr, RET_ERROR);
  auto parameter = opt::BuildParameterNode(func_graph, tensor_info, cnode->fullname_with_scope());
  if (parameter == nullptr) {
    MS_LOG(ERROR) << "BuildParameterNode failed.";
    return RET_ERROR;
  }
  cnode->set_input(1, parameter);
  return RET_OK;
}

STATUS AdjustGatherD(const FuncGraphPtr &func_graph, const CNodePtr &cnode) {
  MS_ASSERT(func_graph != nullptr && cnode != nullptr);
  auto gather_d_node = ops::GetOperator<ops::GatherD>(cnode->input(0));
  MS_CHECK_TRUE_RET(gather_d_node != nullptr, RET_ERROR);
  auto prim = gather_d_node->GetPrim();
  MS_CHECK_TRUE_RET(prim != nullptr, RET_ERROR);
  MS_CHECK_TRUE_RET(prim->GetAttr(ops::kDims) != nullptr, RET_ERROR);
  int32_t dim_val = GetValue<int32_t>(prim->GetAttr(ops::kDims));
  auto dim_parameter_ptr = mindspore::opt::BuildIntValueParameterNode(func_graph, dim_val, "dim");
  MS_CHECK_TRUE_RET(dim_parameter_ptr != nullptr, RET_ERROR);
  auto attr_index = cnode->input(THIRD_INPUT);
  MS_ASSERT(attr_index != nullptr);
  std::vector<AnfNodePtr> new_inputs;
  new_inputs.push_back(cnode->inputs()[FIRST_INPUT]);
  new_inputs.push_back(cnode->inputs()[SECOND_INPUT]);
  new_inputs.push_back(static_cast<AnfNodePtr>(dim_parameter_ptr));
  new_inputs.push_back(attr_index);
  cnode->set_inputs(new_inputs);
  return RET_OK;
}
}  // namespace

bool OnnxInputAdjust::Adjust(const FuncGraphPtr &func_graph, const converter::ConverterParameters &flag) {
  MS_ASSERT(func_graph != nullptr);
  auto manager = Manage(func_graph, true);
  if (manager == nullptr) {
    MS_LOG(ERROR) << "manager is nullptr.";
    return lite::RET_NULL_PTR;
  }
  auto node_list = TopoSort(func_graph->get_return());
  int status = RET_OK;
  bool need_update_manager = false;
  for (auto &node : node_list) {
    if (utils::isa<ParameterPtr>(node)) {
      auto param_node = node->cast<ParameterPtr>();
      status = ReplaceTypeParameterNode(func_graph, param_node, kNumberTypeFloat64, kNumberTypeFloat32);
      if (status != lite::RET_OK) {
        MS_LOG(ERROR) << "replace fp64 param node failed.";
        return status;
      }
      status = ReplaceTypeParameterNode(func_graph, param_node, kNumberTypeInt64, kNumberTypeInt32);
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
    if (opt::CheckPrimitiveType(node, prim::kPrimConstant)) {
      status = ReplaceConstant(func_graph, cnode);
    } else if (opt::CheckPrimitiveType(node, prim::kPrimTranspose) && flag.save_type != kMindIR) {
      status = ReplaceTransposeWithGraphInput(func_graph, cnode);
    } else if (opt::CheckPrimitiveType(node, prim::kPrimStridedSlice)) {
      status = AdjustStridedSlice(func_graph, cnode);
    } else if (opt::CheckPrimitiveType(node, prim::kPrimResize)) {
      status = AdjustResize(&need_update_manager, cnode);
    } else if (opt::CheckPrimitiveType(node, prim::kPrimRandomNormal)) {
      status = AdjustRandomNormal(func_graph, cnode);
    } else if (opt::CheckPrimitiveType(node, prim::kPrimGatherD)) {
      status = AdjustGatherD(func_graph, cnode);
    } else if (opt::CheckPrimitiveType(node, prim::kPrimUnsqueeze)) {
      status = AdjustUnsqueeze(&need_update_manager, cnode);
    } else {
      continue;
    }
    if (status != lite::RET_OK && status != lite::RET_NO_CHANGE) {
      MS_LOG(ERROR) << "adjust input pass is failed.";
      return false;
    }
  }
  if (need_update_manager) {
    mindspore::opt::UpdateManager(func_graph);
  }
  return true;
}
}  // namespace mindspore::lite
