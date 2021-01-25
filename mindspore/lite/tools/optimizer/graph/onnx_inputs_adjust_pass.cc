/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
#include <functional>
#include <memory>
#include <string>
#include <vector>
#include "mindspore/lite/include/errorcode.h"
#include "src/ops/primitive_c.h"

namespace mindspore::opt {
bool OnnxInputAdjustOpPass::CheckInputs(const CNodePtr &cnode) {
  if (cnode == nullptr) {
    MS_LOG(ERROR) << "cnode is nullptr.";
    return false;
  }
  if (std::any_of(cnode->inputs().begin(), cnode->inputs().end(),
                  [](const AnfNodePtr &anf_node) { return anf_node == nullptr; })) {
    MS_LOG(ERROR) << "input is nullptr.";
    return false;
  }
  return true;
}

ParameterPtr OnnxInputAdjustOpPass::BuildParameterNode(const FuncGraphPtr &func_graph, const std::vector<int> &data,
                                                       const std::string &node_name) {
  MS_ASSERT(func_graph != nullptr);
  MS_ASSERT(data.size() != 0);
  auto param_node = func_graph->add_parameter();
  auto type_ptr = TypeIdToType(kNumberTypeInt32);
  std::vector<int64_t> shape_vector{static_cast<int64_t>(data.size())};
  auto abstract_tensor = std::make_shared<abstract::AbstractTensor>(type_ptr, shape_vector);
  param_node->set_abstract(abstract_tensor);
  param_node->set_name(node_name);
  ParamValueLitePtr param_value = std::make_shared<ParamValueLite>();
  MS_ASSERT(param_value != nullptr);
  std::vector<int> shape{static_cast<int>(data.size())};
  param_value->set_tensor_shape(shape);
  param_value->set_tensor_type(kNumberTypeInt32);
  param_value->set_format(schema::Format::Format_NCHW);
  char *default_data = new char[data.size() * sizeof(int)];
  if (memcpy_s(default_data, data.size() * sizeof(int), data.data(), data.size() * sizeof(int)) != EOK) {
    MS_LOG(ERROR) << "memcpy data failed.";
    delete[] default_data;
    return nullptr;
  }
  param_value->SetTensorData(default_data, data.size() * sizeof(int));
  param_node->set_default_param(param_value);
  return param_node;
}

ParameterPtr OnnxInputAdjustOpPass::BuildParameterNode(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                                       const ParamValueLitePtr &param_value) {
  MS_ASSERT(func_graph != nullptr);
  MS_ASSERT(cnode != nullptr);
  MS_ASSERT(param_value != nullptr);
  auto param_node = func_graph->add_parameter();
  auto shape = param_value->tensor_shape();
  std::vector<int64_t> shape_vector;
  std::transform(shape.begin(), shape.end(), std::back_inserter(shape_vector),
                 [](const int &val) { return static_cast<int64_t>(val); });
  auto data_type = param_value->tensor_type() == kNumberTypeInt64 ? kNumberTypeInt32 : param_value->tensor_type();
  auto abstract_tensor = std::make_shared<abstract::AbstractTensor>(TypeIdToType(data_type), shape_vector);
  param_node->set_abstract(abstract_tensor);
  if (utils::isa<CNodePtr>(node)) {
    param_node->set_name(node->cast<CNodePtr>()->fullname_with_scope());
  } else if (utils::isa<ParameterPtr>(node)) {
    param_node->set_name(node->cast<ParameterPtr>()->name());
  }
  ParamValueLitePtr param_value_new = std::make_shared<ParamValueLite>();
  param_value_new->set_format(param_value->format());
  param_value_new->set_tensor_shape(shape);
  size_t data_count = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
  if (param_value->tensor_size() == 0) {
    if (param_value->tensor_type() == kNumberTypeInt64) {
      param_value_new->set_tensor_type(kNumberTypeInt32);
    }
    param_node->set_default_param(param_value_new);
    return param_node;
  }
  if (param_value->tensor_type() == kNumberTypeInt64) {
    param_value_new->set_tensor_type(kNumberTypeInt32);
    auto *tensor_data = new (std::nothrow) int[data_count];
    if (tensor_data == nullptr) {
      MS_LOG(ERROR) << "new data failed";
      return nullptr;
    }
    auto *origin_data = reinterpret_cast<int64_t *>(param_value->tensor_addr());
    for (size_t i = 0; i < data_count; ++i) {
      if (origin_data[i] > static_cast<int64_t>(INT32_MAX) || origin_data[i] < static_cast<int64_t>(INT32_MIN)) {
        MS_LOG(WARNING) << "int64 data " << origin_data[i] << "too big to fit into int32";
        tensor_data[i] = origin_data[i] > 0 ? INT32_MAX : INT32_MIN;
      } else {
        tensor_data[i] = static_cast<int>(origin_data[i]);
      }
    }
    param_value_new->SetTensorData(tensor_data, data_count * sizeof(int32_t));
  } else {
    param_value_new->set_tensor_type(param_value->tensor_type());
    char *tensor_data = new (std::nothrow) char[param_value->tensor_size()];
    if (tensor_data == nullptr) {
      MS_LOG(ERROR) << "new data failed";
      return nullptr;
    }
    if (memcpy_s(tensor_data, param_value->tensor_size(), param_value->tensor_addr(), param_value->tensor_size()) !=
        RET_OK) {
      MS_LOG(ERROR) << "memcpy data failed.";
      delete[] tensor_data;
      return nullptr;
    }
    param_value_new->SetTensorData(tensor_data, param_value->tensor_size());
  }
  param_node->set_default_param(param_value_new);
  return param_node;
}

STATUS OnnxInputAdjustOpPass::StridedSliceAttrToInput(const FuncGraphPtr &func_graph, const CNodePtr &cnode,
                                                      const std::string &attr_name) {
  MS_ASSERT(func_graph != nullptr);
  MS_ASSERT(cnode != nullptr);
  auto inputs = cnode->inputs();
  auto primitive_c = GetValueNode<std::shared_ptr<lite::PrimitiveC>>(cnode->input(0));
  auto value_ptr = primitive_c->GetAttr(attr_name);
  MS_ASSERT(value_ptr != nullptr);
  std::vector<int> value_data = GetValue<std::vector<int>>(value_ptr);
  auto param_node = BuildParameterNode(func_graph, value_data, cnode->fullname_with_scope() + "_" + attr_name);
  inputs.push_back(param_node);
  cnode->set_inputs(inputs);
  primitive_c->EraseAttr(attr_name);
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

STATUS OnnxInputAdjustOpPass::AdjustPower(const CNodePtr &cnode) {
  MS_ASSERT(cnode != nullptr);
  if (!CheckInputs(cnode)) {
    MS_LOG(ERROR) << "input is invalid.";
    return lite::RET_INPUT_TENSOR_ERROR;
  }
  if (cnode->inputs().size() != 3) {
    MS_LOG(ERROR) << "onnx power inputs is 2, but now is " << cnode->inputs().size() - 1;
    return lite::RET_ERROR;
  }
  auto pow_param = cnode->input(2)->cast<ParameterPtr>();
  if (pow_param == nullptr || !pow_param->has_default()) {
    MS_LOG(ERROR) << "pow is from other node, which hasn't been supported.";
    return lite::RET_NOT_SUPPORT;
  }
  auto pow_default = pow_param->default_param()->cast<ParamValueLitePtr>();
  if (pow_default == nullptr) {
    MS_LOG(ERROR) << "pow is not a paramValueLite.";
    return lite::RET_NULL_PTR;
  }
  if (std::accumulate(pow_default->tensor_shape().begin(), pow_default->tensor_shape().end(), 1,
                      std::multiplies<int>()) != 1) {
    MS_LOG(ERROR) << "the pow element num is bigger than 1, which don't support now.";
    return lite::RET_NOT_SUPPORT;
  }
  if (pow_default->tensor_addr() == nullptr) {
    MS_LOG(ERROR) << "power's attr pow can't be obtained.";
    return lite::RET_INVALID_OP_ATTR;
  }
  auto primitive_c = GetValueNode<std::shared_ptr<lite::PrimitiveC>>(cnode->input(0));
  if (primitive_c == nullptr || primitive_c->primitiveT() == nullptr ||
      primitive_c->primitiveT()->value.value == nullptr) {
    MS_LOG(ERROR) << "get primitive_c failed.";
    return lite::RET_NULL_PTR;
  }
  reinterpret_cast<schema::PowerT *>(primitive_c->primitiveT()->value.value)->power =
    *reinterpret_cast<float *>(pow_default->tensor_addr());
  auto inputs = cnode->inputs();
  inputs.pop_back();
  cnode->set_inputs(inputs);
  return lite::RET_OK;
}

STATUS OnnxInputAdjustOpPass::AdjustStridedSlice(const FuncGraphPtr &func_graph, const CNodePtr &cnode) {
  MS_ASSERT(cnode != nullptr);
  if (!CheckInputs(cnode)) {
    MS_LOG(ERROR) << "input is invalid.";
    return lite::RET_INPUT_TENSOR_ERROR;
  }
  if (cnode->inputs().size() == 2) {
    if (StridedSliceAttrToInput(func_graph, cnode, "starts") != lite::RET_OK ||
        StridedSliceAttrToInput(func_graph, cnode, "ends") != lite::RET_OK ||
        StridedSliceAttrToInput(func_graph, cnode, "axes") != lite::RET_OK ||
        StridedSliceAttrToInput(func_graph, cnode, "steps") != lite::RET_OK) {
      MS_LOG(ERROR) << "attr to input failed.";
      return lite::RET_ERROR;
    }
  } else if (cnode->inputs().size() < 4) {
    MS_LOG(ERROR) << "onnx slice's input size need to be larger than 2, now is " << cnode->inputs().size() - 1;
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
      std::vector<int> axes;
      for (int i = 0; i < size; ++i) {
        axes.push_back(i);
      }
      auto new_param_node = BuildParameterNode(func_graph, axes, cnode->fullname_with_scope() + "_axes");
      if (new_param_node == nullptr) {
        MS_LOG(ERROR) << "new a parameter node failed.";
      }
      inputs.push_back(new_param_node);
    }
    case 5: {
      std::vector<int> steps;
      for (int i = 0; i < size; ++i) {
        steps.push_back(1);
      }
      auto new_param_node = BuildParameterNode(func_graph, steps, cnode->fullname_with_scope() + "_steps");
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

STATUS OnnxInputAdjustOpPass::AdjustResize(const CNodePtr &cnode) {
  MS_ASSERT(cnode != nullptr);
  auto node = cnode->input(0);
  MS_ASSERT(value_node != nullptr);
  auto value_node = node->cast<ValueNodePtr>();
  if (value_node == nullptr) {
    MS_LOG(ERROR) << "cnode input0 is not a valuenode.";
    return lite::RET_ERROR;
  }
  MS_ASSERT(value_node->value() != nullptr);
  auto primitive_c = value_node->value()->cast<PrimitiveCPtr>();
  if (primitive_c == nullptr) {
    MS_LOG(ERROR) << "cnode has no primitive_c.";
    return lite::RET_ERROR;
  }
  auto primitive = primitive_c->primitiveT();
  if (primitive == nullptr) {
    MS_LOG(ERROR) << "cnode has no schema::primitive.";
    return lite::RET_ERROR;
  }
  if (primitive->value.type != schema::PrimitiveType_Resize) {
    MS_LOG(DEBUG) << "cnode is not cast node.";
    return RET_OK;
  }
  auto value = primitive->value.value;
  if (value == nullptr) {
    MS_LOG(ERROR) << "value is nullptr.";
    return lite::RET_ERROR;
  }
  auto attr = reinterpret_cast<schema::ResizeT *>(value);
  if (cnode->inputs().size() > 3 &&
      attr->coordinateTransformMode != schema::CoordinateTransformMode_TF_CROP_AND_RESIZE) {
    auto new_resize_inputs = cnode->inputs();
    new_resize_inputs.erase(new_resize_inputs.begin() + 2);
    cnode->set_inputs(new_resize_inputs);
  }
  if (cnode->inputs().size() > 3 && attr->coordinateTransformMode == schema::CoordinateTransformMode_HALF_PIXEL) {
    std::vector<AnfNodePtr> new_resize_inputs;
    new_resize_inputs.push_back(cnode->inputs()[0]);
    new_resize_inputs.push_back(cnode->inputs()[1]);
    new_resize_inputs.push_back(cnode->inputs()[4]);
    cnode->set_inputs(new_resize_inputs);
  }
  return lite::RET_OK;
}

STATUS OnnxInputAdjustOpPass::AdjustConvOrDeConv(const CNodePtr &cnode) {
  MS_ASSERT(cnode != nullptr);
  if (!CheckInputs(cnode)) {
    MS_LOG(ERROR) << "input is invalid.";
    return lite::RET_INPUT_TENSOR_ERROR;
  }
  auto type = opt::GetCNodeType(cnode);
  if (type != schema::PrimitiveType_Conv2D && type != schema::PrimitiveType_DeConv2D) {
    MS_LOG(DEBUG) << "node is not conv2d and deconv2d.";
    return lite::RET_NO_CHANGE;
  }
  if (cnode->inputs().size() < 3) {
    MS_LOG(ERROR) << "conv2d or deconv2d's input size is error, which is " << cnode->inputs().size() - 1;
    return lite::RET_ERROR;
  }
  auto weight_param_node = cnode->input(2)->cast<ParameterPtr>();
  if (weight_param_node == nullptr || !weight_param_node->has_default()) {
    MS_LOG(INFO) << "weight tensor is not const tensor, which hasn't been supported.";
    return lite::RET_NOT_SUPPORT;
  }
  auto weight_param_value = weight_param_node->default_param()->cast<ParamValueLitePtr>();
  if (weight_param_value == nullptr) {
    MS_LOG(ERROR) << "weight is not a paramValueLite.";
    return lite::RET_ERROR;
  }
  auto primitive_c = GetValueNode<std::shared_ptr<lite::PrimitiveC>>(cnode->input(0));
  if (primitive_c == nullptr || primitive_c->primitiveT() == nullptr ||
      primitive_c->primitiveT()->value.value == nullptr) {
    MS_LOG(ERROR) << "get primitive_c failed.";
    return lite::RET_NULL_PTR;
  }
  if (type == schema::PrimitiveType_Conv2D) {
    weight_param_value->set_format(reinterpret_cast<schema::Conv2DT *>(primitive_c->primitiveT()->value.value)->format);
  } else {
    weight_param_value->set_format(
      reinterpret_cast<schema::DeConv2DT *>(primitive_c->primitiveT()->value.value)->format);
  }
  return lite::RET_OK;
}

STATUS OnnxInputAdjustOpPass::AdjustTile(const CNodePtr &cnode) {
  MS_ASSERT(cnode != nullptr);
  if (!CheckInputs(cnode)) {
    MS_LOG(ERROR) << "input is invalid.";
    return lite::RET_INPUT_TENSOR_ERROR;
  }
  if (cnode->inputs().size() != 3) {
    MS_LOG(ERROR) << "x tile input size should be 2, now is " << cnode->inputs().size() - 1;
    return lite::RET_INPUT_TENSOR_ERROR;
  }
  auto multiples_node = cnode->input(2)->cast<ParameterPtr>();
  if (multiples_node == nullptr || !multiples_node->has_default()) {
    MS_LOG(INFO) << "multiples tensor is not const tensor, which hasn't been supported.";
    return lite::RET_NOT_SUPPORT;
  }
  auto multiples_param_value = multiples_node->cast<ParamValueLitePtr>();
  if (multiples_param_value == nullptr) {
    MS_LOG(ERROR) << "weight is not a paramValueLite.";
    return lite::RET_ERROR;
  }
  size_t dims_size = multiples_param_value->tensor_size() / sizeof(int);
  if (dims_size == 0) {
    MS_LOG(INFO) << "multiples tensor is not const tensor, which hasn't been supported.";
    return lite::RET_NOT_SUPPORT;
  }
  std::vector<int> multiples(dims_size, 0);
  if (memcpy_s(multiples.data(), dims_size * sizeof(int), multiples_param_value->tensor_addr(),
               dims_size * sizeof(int)) != EOK) {
    MS_LOG(ERROR) << "memcpy_s failed.";
    return lite::RET_ERROR;
  }
  std::vector<int> dims;
  for (size_t i = 0; i < dims_size; ++i) {
    dims.push_back(i);
  }
  auto primitive_c = GetValueNode<std::shared_ptr<lite::PrimitiveC>>(cnode->input(0));
  if (primitive_c == nullptr || primitive_c->primitiveT() == nullptr ||
      primitive_c->primitiveT()->value.value == nullptr) {
    MS_LOG(ERROR) << "get primitive_c failed.";
    return lite::RET_NULL_PTR;
  }
  reinterpret_cast<schema::TileT *>(primitive_c->primitiveT()->value.value)->multiples = multiples;
  reinterpret_cast<schema::TileT *>(primitive_c->primitiveT()->value.value)->dims = dims;
  return lite::RET_OK;
}

STATUS OnnxInputAdjustOpPass::AdjustCast(const CNodePtr &cnode) {
  MS_ASSERT(cnode != nullptr);
  auto node = cnode->input(0);
  MS_ASSERT(value_node != nullptr);
  auto value_node = node->cast<ValueNodePtr>();
  if (value_node == nullptr) {
    MS_LOG(ERROR) << "cnode input0 is not a valuenode.";
    return lite::RET_ERROR;
  }
  MS_ASSERT(value_node->value() != nullptr);
  auto primitive_c = value_node->value()->cast<PrimitiveCPtr>();
  if (primitive_c == nullptr) {
    MS_LOG(ERROR) << "cnode has no primitive_c.";
    return lite::RET_ERROR;
  }
  auto primitive = primitive_c->primitiveT();
  if (primitive == nullptr) {
    MS_LOG(ERROR) << "cnode has no schema::primitive.";
    return lite::RET_ERROR;
  }
  if (primitive->value.type != schema::PrimitiveType_Cast) {
    MS_LOG(DEBUG) << "cnode is not cast node.";
    return RET_OK;
  }
  auto value = primitive->value.value;
  if (value == nullptr) {
    MS_LOG(ERROR) << "value is nullptr.";
    return lite::RET_ERROR;
  }
  auto attr = reinterpret_cast<schema::CastT *>(value);
  if (attr->dstT == kNumberTypeInt64) {
    attr->dstT = kNumberTypeInt32;
  }
  return lite::RET_OK;
}

STATUS OnnxInputAdjustOpPass::ReplaceConstant(const FuncGraphPtr &func_graph, const CNodePtr &cnode) {
  MS_ASSERT(func_graph != nullptr);
  MS_ASSERT(cnode != nullptr);
  if (cnode->inputs().size() < 1 || cnode->input(0) == nullptr) {
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
  if (cnode->inputs().size() != 2) {
    MS_LOG(ERROR) << "onnx transpose input size is 1, now is " << cnode->inputs().size() - 1;
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
  auto prim_anf = cnode->input(0);
  if (prim_anf == nullptr || !utils::isa<ValueNodePtr>(prim_anf)) {
    MS_LOG(ERROR) << "cnode input0 is invalid.";
    return lite::RET_ERROR;
  }
  auto value_node = prim_anf->cast<ValueNodePtr>();
  MS_ASSERT(value_node->value() != nullptr);
  auto prim = value_node->value()->cast<PrimitiveCPtr>();
  MS_ASSERT(prim != nullptr && prim->primitiveT() != nullptr && prim->primitiveT()->value.value != nullptr);
  auto attr = reinterpret_cast<schema::TransposeT *>(prim->primitiveT()->value.value);
  auto perm = attr->perm;
  std::vector<int> transpose_attr;
  std::transform(perm.begin(), perm.end(), std::back_inserter(transpose_attr),
                 [](const int &val) { return val < 0 ? val + 4 : val; });
  if (transpose_attr[0] == 0 && transpose_attr[1] == 3 && transpose_attr[2] == 1) {
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
    auto type = opt::GetCNodeType(node);
    if (type == schema::PrimitiveType_Power) {
      status = AdjustPower(cnode);
    } else if (type == schema::PrimitiveType_StridedSlice) {
      status = AdjustStridedSlice(func_graph, cnode);
    } else if (type == schema::PrimitiveType_Conv2D || type == schema::PrimitiveType_DeConv2D) {
      status = AdjustConvOrDeConv(cnode);
    } else if (type == schema::PrimitiveType_Tile) {
      status = AdjustConvOrDeConv(cnode);
    } else if (type == schema::PrimitiveType_Constant) {
      status = ReplaceConstant(func_graph, cnode);
    } else if (type == schema::PrimitiveType_Cast) {
      status = AdjustCast(cnode);
    } else if (type == schema::PrimitiveType_Transpose) {
      status = ReplaceTransposeWithGraphInput(func_graph, cnode);
    } else if (type == schema::PrimitiveType_Resize) {
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
