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

#include "c_api/include/node.h"
#include "c_api/src/helper.h"
#include "c_api/src/common.h"
#include "c_api/src/utils.h"
#include "base/base.h"
#include "ops/core_ops.h"
#include "ir/param_info.h"
#include "backend/common/optimizer/helper.h"

STATUS SetAttrs(ResMgrHandle res_mgr, const PrimitivePtr &prim, char **attr_names, AttrHandle attrs[],
                size_t attr_num) {
  AttrMap attr_map{};
  for (size_t i = 0; i < attr_num; ++i) {
    if (attr_names[i] == nullptr) {
      MS_LOG(ERROR) << "Input array [attr_names] has nullptr element, index: " << i;
      return RET_NULL_PTR;
    }
    auto value = GetSrcPtr<ValuePtr>(res_mgr, attrs[i]);
    if (value == nullptr) {
      MS_LOG(ERROR) << "Get source pointer failed.";
      return RET_NULL_PTR;
    }
    std::string name(attr_names[i]);
    if (name == "data_format") {
      attr_map["format"] = value;
    } else if (name == "group") {
      attr_map["groups"] = value;
    }
    attr_map[name] = value;
  }
  (void)prim->SetAttrs(attr_map);
  return RET_OK;
}

NodeHandle MSNewOp(ResMgrHandle res_mgr, GraphHandle graph, const char *op_type, const Handle inputs[],
                   size_t input_num, char **attr_names, AttrHandle attrs[], size_t attr_num) {
  if (res_mgr == nullptr || graph == nullptr || op_type == nullptr || inputs == nullptr) {
    MS_LOG(ERROR) << "Input Handle [res_mgr] or [graph] or [op_type] or [inputs] is nullptr.";
    return nullptr;
  }
  // convert raw input pointer to source shared pointer
  auto res_fg = GetSrcPtr<FuncGraphPtr>(res_mgr, graph);
  if (res_fg == nullptr) {
    MS_LOG(ERROR) << "Get source pointer failed.";
    return nullptr;
  }
  auto res_mgr_ptr = reinterpret_cast<ResourceManager *>(res_mgr);
  std::vector<AnfNodePtr> cnode_inputs{};
  mindspore::AbstractBasePtrList abs_list{};
  auto prim = std::make_shared<PrimitiveImpl>(op_type);
  if (attr_names != nullptr && attrs != nullptr) {
    auto ret = SetAttrs(res_mgr, prim, attr_names, attrs, attr_num);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "Op set attributes failed.";
      return nullptr;
    }
  }
  auto prim_node = mindspore::NewValueNode(prim);
  cnode_inputs.push_back(prim_node);
  CNodePtr cnode = nullptr;
  try {
    for (size_t i = 0; i < input_num; ++i) {
      auto input = GetSrcPtr<AnfNodePtr>(res_mgr, inputs[i]);
      MS_EXCEPTION_IF_NULL(input);
      ConvertConstScalarInputToTensor(input);
      cnode_inputs.push_back(input);
      abs_list.push_back(input->abstract());
    }
    cnode = res_fg->NewCNode(cnode_inputs);
    if (res_mgr_ptr->GetInfer()) {
      auto out_abs = mindspore::opt::CppInferShapeAndType(prim, abs_list);
      cnode->set_abstract(out_abs);
    }
  } catch (const std::exception &e) {
    MS_LOG(ERROR) << "FuncGraph create CNode failed. Error info: " << e.what();
    return nullptr;
  }
  MS_LOG(INFO) << "Add Operator" << op_type;
  return GetRawPtr(res_mgr, cnode);
}

NodeHandle MSPackNodesTuple(ResMgrHandle res_mgr, GraphHandle graph, const Handle nodes[], size_t node_num) {
  if (res_mgr == nullptr || graph == nullptr || nodes == nullptr) {
    MS_LOG(ERROR) << "Input GraphHandle [res_mgr] or [graph] or [nodes] is nullptr.";
    return nullptr;
  }
  CNodePtr make_tuple_cnode = nullptr;
  try {
    auto res_fg = GetSrcPtr<FuncGraphPtr>(res_mgr, graph);
    MS_EXCEPTION_IF_NULL(res_fg);
    std::vector<AnfNodePtr> in_nodes{NewValueNode(mindspore::prim::kPrimMakeTuple)};
    mindspore::AbstractBasePtrList abs_list{};
    for (size_t i = 0; i < node_num; ++i) {
      auto in_node = GetSrcPtr<AnfNodePtr>(res_mgr, nodes[i]);
      MS_EXCEPTION_IF_NULL(in_node);
      in_nodes.push_back(in_node);
      ConvertConstScalarInputToTensor(in_node);
      abs_list.push_back(in_node->abstract());
    }
    make_tuple_cnode = res_fg->NewCNode(in_nodes);
    make_tuple_cnode->set_abstract(std::make_shared<AbstractTupleImpl>(abs_list));
  } catch (const std::exception &e) {
    MS_LOG(ERROR) << "FuncGraph set output failed. Error info: " << e.what();
    return nullptr;
  }
  return GetRawPtr(res_mgr, make_tuple_cnode);
}

NodeHandle MSOpGetSpecOutput(ResMgrHandle res_mgr, GraphHandle graph, const NodeHandle op, size_t i) {
  if (res_mgr == nullptr || graph == nullptr || op == nullptr) {
    MS_LOG(ERROR) << "Input Handle [res_mgr] or [op] is nullptr.";
    return nullptr;
  }
  CNodePtr ret_node = nullptr;
  try {
    auto res_fg = GetSrcPtr<FuncGraphPtr>(res_mgr, graph);
    MS_EXCEPTION_IF_NULL(res_fg);
    auto cnode = GetSrcPtr<CNodePtr>(res_mgr, op);
    MS_EXCEPTION_IF_NULL(cnode);
    auto abs = cnode->abstract();
    if (abs == nullptr) {
      MS_LOG(ERROR) << "Input op's abstract is nullptr!";
      return nullptr;
    }
    if (abs->isa<mindspore::abstract::AbstractTuple>()) {
      auto branch_num = abs->cast<mindspore::abstract::AbstractTuplePtr>()->size();
      if (i >= branch_num) {
        MS_LOG(ERROR) << "Invalid output branch index, it should be less than " << branch_num << ", but got: " << i;
        return nullptr;
      }
      auto idx = mindspore::NewValueNode(mindspore::SizeToLong(i));
      auto abs_scalar = std::make_shared<mindspore::abstract::AbstractScalar>(mindspore::SizeToInt(i));
      idx->set_abstract(abs_scalar);
      ret_node = res_fg->NewCNode({NewValueNode(mindspore::prim::kPrimTupleGetItem), cnode, idx});
      ret_node->set_abstract(abs->cast<mindspore::abstract::AbstractTuplePtr>()->elements()[i]);
    } else {
      if (i >= 1) {
        MS_LOG(ERROR) << "Invalid output index. The op has only one output, so the output index should be 0, or you can"
                         " directly use this op as the output without calling this function, but got: "
                      << i;
        return nullptr;
      }
      MS_LOG(WARNING) << "The op has only one output, you can directly use this op as the output without calling this "
                         "function. Now the op itself is returned.";
      ret_node = cnode;
    }
  } catch (const std::exception &e) {
    MS_LOG(ERROR) << "FuncGraph get output failed. Error info: " << e.what();
    return nullptr;
  }
  return GetRawPtr(res_mgr, ret_node);
}

NodeHandle MSOpGetInput(ResMgrHandle res_mgr, const NodeHandle op, size_t i) {
  if (res_mgr == nullptr || op == nullptr) {
    MS_LOG(ERROR) << "Input Handle [res_mgr] or [op] is nullptr.";
    return nullptr;
  }
  mindspore::AnfNodePtr anf_node = nullptr;
  try {
    auto src_cnode = GetSrcPtr<CNodePtr>(res_mgr, op);
    MS_EXCEPTION_IF_NULL(src_cnode);
    if (i >= src_cnode->size() - 1) {
      MS_LOG(ERROR) << "Invalid input index, it should be less than " << src_cnode->size() - 1 << ", but got: " << i;
      return nullptr;
    }
    anf_node = src_cnode->input(i + 1);
  } catch (const std::exception &e) {
    MS_LOG(ERROR) << "Get input from CNode failed. Error info: " << e.what();
    return nullptr;
  }
  return GetRawPtr(res_mgr, anf_node);
}

size_t MSOpGetInputsNum(ResMgrHandle res_mgr, const NodeHandle op, STATUS *error) {
  if (error == nullptr) {
    MS_LOG(ERROR) << "Input status flag [error] is nullptr.";
    return 0;
  }
  if (res_mgr == nullptr || op == nullptr) {
    MS_LOG(ERROR) << "Input Handle [res_mgr] or [op] is nullptr.";
    *error = RET_NULL_PTR;
    return 0;
  }
  size_t input_num;
  try {
    auto src_cnode = GetSrcPtr<CNodePtr>(res_mgr, op);
    MS_EXCEPTION_IF_NULL(src_cnode);
    input_num = src_cnode->inputs().size() - 1;
  } catch (const std::exception &e) {
    MS_LOG(ERROR) << "FuncGraph get input number failed. Error info: " << e.what();
    *error = RET_ERROR;
    return 0;
  }
  *error = RET_OK;
  return input_num;
}

STATUS MSOpGetInputs(ResMgrHandle res_mgr, const NodeHandle op, NodeHandle inputs[], size_t input_num) {
  if (res_mgr == nullptr || op == nullptr || inputs == nullptr) {
    MS_LOG(ERROR) << "Input Handle [res_mgr] or [op] or [inputs] is nullptr.";
    return RET_NULL_PTR;
  }
  try {
    auto src_cnode = GetSrcPtr<CNodePtr>(res_mgr, op);
    MS_EXCEPTION_IF_NULL(src_cnode);
    auto in_num = src_cnode->size() - 1;
    if (in_num != input_num) {
      MS_LOG(ERROR) << "Invalid input number, it should be: " << in_num << ", but got: " << input_num;
      return RET_ERROR;
    }
    auto cnode_inputs = src_cnode->inputs();
    for (size_t i = 0; i < input_num; i++) {
      inputs[i] = GetRawPtr(res_mgr, cnode_inputs[i + 1]);
    }
  } catch (const std::exception &e) {
    MS_LOG(ERROR) << "Get inputs from CNode failed. Error info: " << e.what();
    return RET_ERROR;
  }
  return RET_OK;
}

NodeHandle MSNewSubGraphNode(ResMgrHandle res_mgr, GraphHandle graph, GraphHandle sub_graph, const Handle inputs[],
                             size_t input_num) {
  if (res_mgr == nullptr || graph == nullptr || sub_graph == nullptr || inputs == nullptr) {
    MS_LOG(ERROR) << "Input Handle [res_mgr] or [graph] or [sub_graph] or [inputs] is nullptr.";
    return nullptr;
  }
  CNodePtr cnode = nullptr;
  try {
    auto res_fg = GetSrcPtr<FuncGraphPtr>(res_mgr, graph);
    MS_EXCEPTION_IF_NULL(res_fg);
    auto res_sub_fg = GetSrcPtr<FuncGraphPtr>(res_mgr, sub_graph);
    MS_EXCEPTION_IF_NULL(res_sub_fg);
    auto sub_fg_node = mindspore::NewValueNode(res_sub_fg);
    std::vector<AnfNodePtr> cnode_inputs{};
    cnode_inputs.push_back(sub_fg_node);
    for (size_t i = 0; i < input_num; ++i) {
      auto cnode_input = GetSrcPtr<AnfNodePtr>(res_mgr, inputs[i]);
      MS_EXCEPTION_IF_NULL(cnode_input);
      cnode_inputs.push_back(cnode_input);
    }
    cnode = res_fg->NewCNode(cnode_inputs);
  } catch (const std::exception &e) {
    MS_LOG(ERROR) << "FuncGraph create SubGraph node failed. Error info: " << e.what();
    return nullptr;
  }
  MS_LOG(INFO) << "Add subgraph node";
  return GetRawPtr(res_mgr, cnode);
}

NodeHandle MSNewPlaceholder(ResMgrHandle res_mgr, GraphHandle graph, TypeId type, const int64_t shape[],
                            size_t shape_size) {
  if (res_mgr == nullptr || graph == nullptr) {
    MS_LOG(ERROR) << "Input Handle [res_mgr] or [graph] is nullptr.";
    return nullptr;
  }
  ParameterPtr param = nullptr;
  try {
    auto res_fg = GetSrcPtr<FuncGraphPtr>(res_mgr, graph);
    MS_EXCEPTION_IF_NULL(res_fg);
    param = res_fg->add_parameter();
    auto type_ptr = mindspore::TypeIdToType(mindspore::TypeId(type));
    AbstractBasePtr abs = GetAbstract(type_ptr, shape, shape_size, true);
    param->set_abstract(abs);
  } catch (const std::exception &e) {
    MS_LOG(ERROR) << "FuncGraph add parameter failed. Error info: " << e.what();
    return nullptr;
  }
  return GetRawPtr(res_mgr, param);
}

NodeHandle MSNewTensorVariable(ResMgrHandle res_mgr, GraphHandle graph, void *data, TypeId type, const int64_t shape[],
                               size_t shape_size, size_t data_len) {
  if (res_mgr == nullptr || graph == nullptr || data == nullptr || shape == nullptr) {
    MS_LOG(ERROR) << "Input Handle [res_mgr] or [graph] or [data] or [shape] is nullptr.";
    return nullptr;
  }
  ParameterPtr param = nullptr;
  ShapeVector shape_vec(shape, shape + shape_size);
  try {
    auto res_fg = GetSrcPtr<FuncGraphPtr>(res_mgr, graph);
    MS_EXCEPTION_IF_NULL(res_fg);
    param = res_fg->add_parameter();
    auto tensor = std::make_shared<TensorImpl>(mindspore::TypeId(type), shape_vec, data, data_len);
    tensor->set_param_info(std::make_shared<mindspore::ParamInfo>());
    param->set_abstract(tensor->ToAbstract());
    param->set_default_param(tensor);
  } catch (const std::exception &e) {
    MS_LOG(ERROR) << "New Tensor Variable failed. Error info: " << e.what();
    return nullptr;
  }
  return GetRawPtr(res_mgr, param);
}

NodeHandle MSNewTensorVariableFromTensor(ResMgrHandle res_mgr, GraphHandle graph, TensorHandle tensor) {
  if (res_mgr == nullptr || graph == nullptr || tensor == nullptr) {
    MS_LOG(ERROR) << "Input Handle [res_mgr] or [graph] or [tensor] is nullptr.";
    return nullptr;
  }
  ParameterPtr param = nullptr;
  try {
    auto res_fg = GetSrcPtr<FuncGraphPtr>(res_mgr, graph);
    MS_EXCEPTION_IF_NULL(res_fg);
    auto tensor_impl = GetSrcPtr<TensorPtr>(res_mgr, tensor);
    MS_EXCEPTION_IF_NULL(tensor_impl);
    param = res_fg->add_parameter();
    param->set_abstract(tensor_impl->ToAbstract());
    param->set_default_param(tensor_impl);
  } catch (const std::exception &e) {
    MS_LOG(ERROR) << "New Tensor Variable failed. Error info: " << e.what();
    return nullptr;
  }
  return GetRawPtr(res_mgr, param);
}

size_t MSTensorVariableGetDataSize(ResMgrHandle res_mgr, NodeHandle node, STATUS *error) {
  if (error == nullptr) {
    MS_LOG(ERROR) << "Input status flag [error] is nullptr.";
    return 0;
  }
  if (res_mgr == nullptr || node == nullptr) {
    MS_LOG(ERROR) << "Input Handle [res_mgr] or [node] is nullptr.";
    *error = RET_NULL_PTR;
    return 0;
  }
  try {
    auto node_impl = GetSrcPtr<ParameterPtr>(res_mgr, node);
    MS_EXCEPTION_IF_NULL(node_impl);
    auto val = node_impl->default_param();
    MS_EXCEPTION_IF_NULL(val);
    auto tensor = val->cast<TensorPtr>();
    MS_EXCEPTION_IF_NULL(tensor);
    size_t data_size = tensor->Size();
    *error = RET_OK;
    return data_size;
  } catch (const std::exception &e) {
    MS_LOG(ERROR) << "Tensor Variable get data failed. Error info: " << e.what();
    *error = RET_ERROR;
    return 0;
  }
}

void *MSTensorVariableGetData(ResMgrHandle res_mgr, NodeHandle node) {
  if (res_mgr == nullptr || node == nullptr) {
    MS_LOG(ERROR) << "Input Handle [res_mgr] or [node] is nullptr.";
    return nullptr;
  }
  try {
    auto node_impl = GetSrcPtr<ParameterPtr>(res_mgr, node);
    MS_EXCEPTION_IF_NULL(node_impl);
    auto val = node_impl->default_param();
    MS_EXCEPTION_IF_NULL(val);
    auto tensor = val->cast<TensorPtr>();
    MS_EXCEPTION_IF_NULL(tensor);
    void *data = tensor->data_c();
    return data;
  } catch (const std::exception &e) {
    MS_LOG(ERROR) << "Tensor Variable get data failed. Error info: " << e.what();
    return nullptr;
  }
}

NodeHandle MSNewTensorConstant(ResMgrHandle res_mgr, void *data, TypeId type, const int64_t shape[], size_t shape_size,
                               size_t data_len) {
  if (res_mgr == nullptr || data == nullptr || shape == nullptr) {
    MS_LOG(ERROR) << "Input Handle [res_mgr] or [data] or [shape] is nullptr.";
    return nullptr;
  }
  ShapeVector shape_vec(shape, shape + shape_size);
  ValueNodePtr value_node = nullptr;
  try {
    auto tensor = std::make_shared<TensorImpl>(mindspore::TypeId(type), shape_vec, data, data_len);
    tensor->set_param_info(std::make_shared<mindspore::ParamInfo>());
    value_node = mindspore::NewValueNode(tensor);
    value_node->set_abstract(tensor->ToAbstract());
  } catch (const std::exception &e) {
    MS_LOG(ERROR) << "New Tensor Variable failed. Error info: " << e.what();
    return nullptr;
  }
  return GetRawPtr(res_mgr, value_node);
}

NodeHandle MSNewTensorConstantFromTensor(ResMgrHandle res_mgr, TensorHandle tensor) {
  if (res_mgr == nullptr || tensor == nullptr) {
    MS_LOG(ERROR) << "Input Handle [res_mgr] or [tensor] is nullptr.";
    return nullptr;
  }
  ValueNodePtr value_node = nullptr;
  try {
    auto tensor_impl = GetSrcPtr<TensorPtr>(res_mgr, tensor);
    MS_EXCEPTION_IF_NULL(tensor_impl);
    value_node = mindspore::NewValueNode(tensor_impl);
    value_node->set_abstract(tensor_impl->ToAbstract());
  } catch (const std::exception &e) {
    MS_LOG(ERROR) << "New Tensor Variable failed. Error info: " << e.what();
    return nullptr;
  }
  return GetRawPtr(res_mgr, value_node);
}

size_t MSTensorConstantGetDataSize(ResMgrHandle res_mgr, NodeHandle node, STATUS *error) {
  if (error == nullptr) {
    MS_LOG(ERROR) << "Input status flag [error] is nullptr.";
    return 0;
  }
  if (res_mgr == nullptr || node == nullptr) {
    MS_LOG(ERROR) << "Input Handle [res_mgr] or [node] is nullptr.";
    *error = RET_NULL_PTR;
    return 0;
  }
  try {
    auto node_impl = GetSrcPtr<ValueNodePtr>(res_mgr, node);
    MS_EXCEPTION_IF_NULL(node_impl);
    auto val = node_impl->value();
    MS_EXCEPTION_IF_NULL(val);
    auto tensor = val->cast<TensorPtr>();
    MS_EXCEPTION_IF_NULL(tensor);
    size_t data_size = tensor->Size();
    *error = RET_OK;
    return data_size;
  } catch (const std::exception &e) {
    MS_LOG(ERROR) << "Tensor Constant get data failed. Error info: " << e.what();
    *error = RET_ERROR;
    return 0;
  }
}

void *MSTensorConstantGetData(ResMgrHandle res_mgr, NodeHandle node) {
  if (res_mgr == nullptr || node == nullptr) {
    MS_LOG(ERROR) << "Input Handle [res_mgr] or [node] is nullptr.";
    return nullptr;
  }
  try {
    auto node_impl = GetSrcPtr<ValueNodePtr>(res_mgr, node);
    MS_EXCEPTION_IF_NULL(node_impl);
    auto val = node_impl->value();
    MS_EXCEPTION_IF_NULL(val);
    auto tensor = val->cast<TensorPtr>();
    MS_EXCEPTION_IF_NULL(tensor);
    void *data = tensor->data_c();
    return data;
  } catch (const std::exception &e) {
    MS_LOG(ERROR) << "Tensor Constant get data failed. Error info: " << e.what();
    return nullptr;
  }
}

NodeHandle MSNewScalarConstantFloat32(ResMgrHandle res_mgr, float value) {
  MS_LOG(INFO) << "New Float32 Scalar Value!s";
  if (res_mgr == nullptr) {
    MS_LOG(ERROR) << "Input Handle [res_mgr] is nullptr.";
    return nullptr;
  }
  auto value_node = mindspore::NewValueNode(value);
  value_node->set_abstract(std::make_shared<AbstractScalarImpl>(value));
  return GetRawPtr(res_mgr, value_node);
}

NodeHandle MSNewScalarConstantBool(ResMgrHandle res_mgr, bool value) {
  MS_LOG(INFO) << "New Bool Scalar Value!";
  if (res_mgr == nullptr) {
    MS_LOG(ERROR) << "Input Handle [res_mgr] is nullptr.";
    return nullptr;
  }
  auto value_node = mindspore::NewValueNode(value);
  value_node->set_abstract(std::make_shared<AbstractScalarImpl>(value));
  return GetRawPtr(res_mgr, value_node);
}

NodeHandle MSNewScalarConstantInt32(ResMgrHandle res_mgr, int value) {
  MS_LOG(INFO) << "New Int32 Scalar Value!";
  if (res_mgr == nullptr) {
    MS_LOG(ERROR) << "Input Handle [res_mgr] is nullptr.";
    return nullptr;
  }
  auto value_node = mindspore::NewValueNode(value);
  value_node->set_abstract(std::make_shared<AbstractScalarImpl>(value));
  return GetRawPtr(res_mgr, value_node);
}

NodeHandle MSNewScalarConstantInt64(ResMgrHandle res_mgr, int64_t value) {
  MS_LOG(INFO) << "New Int64 Scalar Value!";
  if (res_mgr == nullptr) {
    MS_LOG(ERROR) << "Input Handle [res_mgr] is nullptr.";
    return nullptr;
  }
  auto value_node = mindspore::NewValueNode(value);
  value_node->set_abstract(std::make_shared<AbstractScalarImpl>(value));
  return GetRawPtr(res_mgr, value_node);
}

NodeHandle MSNewStringConstant(ResMgrHandle res_mgr, const char *str) {
  MS_LOG(INFO) << "New String Scalar Value!";
  if (res_mgr == nullptr || str == nullptr) {
    MS_LOG(ERROR) << "Input Handle [res_mgr] or [str] is nullptr.";
    return nullptr;
  }
  string str_val(str);
  auto value_node = mindspore::NewValueNode(str_val);
  value_node->set_abstract(std::make_shared<AbstractScalarImpl>(str_val));
  return GetRawPtr(res_mgr, value_node);
}

NodeHandle MSNewTupleConstantInt64(ResMgrHandle res_mgr, const int64_t vec[], size_t size) {
  MS_LOG(INFO) << "New Vector Value!";
  if (res_mgr == nullptr || vec == nullptr) {
    MS_LOG(ERROR) << "Input Handle [res_mgr] or [vec] is nullptr.";
    return nullptr;
  }
  auto value_node = mindspore::NewValueNode(std::vector<int64_t>(vec, vec + size));
  mindspore::AbstractBasePtrList abs_list = {};
  for (size_t i = 0; i < size; i++) {
    AbstractBasePtr base = std::make_shared<AbstractScalarImpl>(vec[i]);
    abs_list.push_back(base);
  }
  auto abstract = std::make_shared<AbstractTupleImpl>(abs_list);
  value_node->set_abstract(abstract);
  return GetRawPtr(res_mgr, value_node);
}

NodeHandle MSNewTypeConstant(ResMgrHandle res_mgr, TypeId type) {
  MS_LOG(INFO) << "New Type Value: " << type;
  if (res_mgr == nullptr) {
    MS_LOG(ERROR) << "Input Handle [res_mgr] is nullptr.";
    return nullptr;
  }
  auto type_ptr = mindspore::TypeIdToType(mindspore::TypeId(type));
  auto value_node = mindspore::NewValueNode(type_ptr);
  auto abstract = std::make_shared<AbstractTypeImpl>(type_ptr);
  value_node->set_abstract(abstract);
  return GetRawPtr(res_mgr, value_node);
}

int MSScalarConstantGetValueInt32(ResMgrHandle res_mgr, const NodeHandle node, STATUS *error) {
  MS_LOG(INFO) << "Get Int32 Scalar Value!";
  if (error == nullptr) {
    MS_LOG(ERROR) << "Input status flag [error] is nullptr.";
    return 0;
  }
  if (res_mgr == nullptr || node == nullptr) {
    MS_LOG(ERROR) << "Input Handle [res_mgr] or [node] is nullptr.";
    *error = RET_NULL_PTR;
    return 0;
  }
  int ret_val = 0;
  *error = RET_OK;
  try {
    auto node_impl = GetSrcPtr<ValueNodePtr>(res_mgr, node);
    MS_EXCEPTION_IF_NULL(node_impl);
    auto val = node_impl->value();
    MS_EXCEPTION_IF_NULL(val);
    if (val->isa<TensorImpl>()) {
      auto val_tensor = val->cast<TensorPtr>();
      auto data = val_tensor->data_c();
      MS_EXCEPTION_IF_NULL(data);
      ret_val = static_cast<int *>(data)[0];
    } else if (val->isa<Int32ImmImpl>()) {
      auto val_imm = val->cast<Int32ImmPtr>();
      ret_val = val_imm->value();
    } else {
      MS_LOG(ERROR) << "Input node has invalid value type: " << val->type_name();
      *error = RET_ERROR;
    }
  } catch (const std::exception &e) {
    MS_LOG(ERROR) << "Get Int32 Scalar value failed. Error info: " << e.what();
    *error = RET_ERROR;
  }
  return ret_val;
}

float MSScalarConstantGetValueFloat32(ResMgrHandle res_mgr, const NodeHandle node, STATUS *error) {
  MS_LOG(INFO) << "Get Float32 Scalar Value!";
  if (error == nullptr) {
    MS_LOG(ERROR) << "Input status flag [error] is nullptr.";
    return 0;
  }
  if (res_mgr == nullptr || node == nullptr) {
    MS_LOG(ERROR) << "Input Handle [res_mgr] or [node] is nullptr.";
    *error = RET_NULL_PTR;
    return 0;
  }
  float ret_val = 0;
  *error = RET_OK;
  try {
    auto node_impl = GetSrcPtr<ValueNodePtr>(res_mgr, node);
    MS_EXCEPTION_IF_NULL(node_impl);
    auto val = node_impl->value();
    MS_EXCEPTION_IF_NULL(val);
    if (val->isa<TensorImpl>()) {
      auto val_tensor = val->cast<TensorPtr>();
      auto data = val_tensor->data_c();
      MS_EXCEPTION_IF_NULL(data);
      ret_val = static_cast<float *>(data)[0];
    } else if (val->isa<Float32ImmImpl>()) {
      auto val_imm = val->cast<Float32ImmPtr>();
      ret_val = val_imm->value();
    } else {
      MS_LOG(ERROR) << "Input node has invalid value type: " << val->type_name();
      *error = RET_ERROR;
    }
  } catch (const std::exception &e) {
    MS_LOG(ERROR) << "Get Float32 Scalar value failed. Error info: " << e.what();
    *error = RET_ERROR;
  }
  return ret_val;
}

bool MSScalarConstantGetValueBool(ResMgrHandle res_mgr, const NodeHandle node, STATUS *error) {
  MS_LOG(INFO) << "Get Bool Scalar Value!";
  if (error == nullptr) {
    MS_LOG(ERROR) << "Input status flag [error] is nullptr.";
    return false;
  }
  if (res_mgr == nullptr || node == nullptr) {
    MS_LOG(ERROR) << "Input Handle [res_mgr] or [node] is nullptr.";
    *error = RET_NULL_PTR;
    return false;
  }
  int ret_val = false;
  *error = RET_OK;
  try {
    auto node_impl = GetSrcPtr<ValueNodePtr>(res_mgr, node);
    MS_EXCEPTION_IF_NULL(node_impl);
    auto val = node_impl->value();
    MS_EXCEPTION_IF_NULL(val);
    if (val->isa<TensorImpl>()) {
      auto val_tensor = val->cast<TensorPtr>();
      auto data = val_tensor->data_c();
      MS_EXCEPTION_IF_NULL(data);
      ret_val = static_cast<bool *>(data)[0];
    } else if (val->isa<BoolImmImpl>()) {
      auto val_imm = val->cast<BoolImmPtr>();
      ret_val = val_imm->value();
    } else {
      MS_LOG(ERROR) << "Input node has invalid value type: " << val->type_name();
      *error = RET_ERROR;
    }
  } catch (const std::exception &e) {
    MS_LOG(ERROR) << "Get Bool Scalar value failed. Error info: " << e.what();
    *error = RET_ERROR;
  }
  return ret_val;
}

int64_t MSScalarConstantGetValueInt64(ResMgrHandle res_mgr, const NodeHandle node, STATUS *error) {
  MS_LOG(INFO) << "Get Int64 Scalar Value!";
  if (error == nullptr) {
    MS_LOG(ERROR) << "Input status flag [error] is nullptr.";
    return 0;
  }
  if (res_mgr == nullptr || node == nullptr) {
    MS_LOG(ERROR) << "Input Handle [res_mgr] or [node] is nullptr.";
    *error = RET_NULL_PTR;
    return 0;
  }
  int64_t ret_val = 0;
  *error = RET_OK;
  try {
    auto node_impl = GetSrcPtr<ValueNodePtr>(res_mgr, node);
    MS_EXCEPTION_IF_NULL(node_impl);
    auto val = node_impl->value();
    MS_EXCEPTION_IF_NULL(val);
    if (val->isa<TensorImpl>()) {
      auto val_tensor = val->cast<TensorPtr>();
      auto data = val_tensor->data_c();
      MS_EXCEPTION_IF_NULL(data);
      ret_val = static_cast<int64_t *>(data)[0];
    } else if (val->isa<Int64ImmImpl>()) {
      auto val_imm = val->cast<Int64ImmPtr>();
      ret_val = val_imm->value();
    } else {
      MS_LOG(ERROR) << "Input node has invalid value type: " << val->type_name();
      *error = RET_ERROR;
    }
  } catch (const std::exception &e) {
    MS_LOG(ERROR) << "Get Int64 Scalar value failed. Error info: " << e.what();
    *error = RET_ERROR;
  }
  return ret_val;
}

STATUS MSStringConstantGetValue(ResMgrHandle res_mgr, const NodeHandle node, char str_buf[], size_t str_len) {
  MS_LOG(INFO) << "Get String Constant Value!";
  if (res_mgr == nullptr || node == nullptr || str_buf == nullptr) {
    MS_LOG(ERROR) << "Input Handle [res_mgr] or [node] or [str_buf] is nullptr.";
    return RET_NULL_PTR;
  }
  try {
    auto node_impl = GetSrcPtr<ValueNodePtr>(res_mgr, node);
    MS_EXCEPTION_IF_NULL(node_impl);
    auto val = node_impl->value();
    MS_EXCEPTION_IF_NULL(val);
    auto val_str = val->cast<StringImmPtr>();
    std::string ret_val = val_str->value();
    size_t valid_size = ret_val.size() < str_len - 1 ? ret_val.size() : str_len - 1;
    for (size_t i = 0; i < valid_size; i++) {
      str_buf[i] = ret_val.c_str()[i];
    }
    str_buf[valid_size] = '\0';
    return RET_OK;
  } catch (const std::exception &e) {
    MS_LOG(ERROR) << "Get String Constant value failed. Error info: " << e.what();
    return RET_ERROR;
  }
}

size_t MSTupleConstantGetSize(ResMgrHandle res_mgr, const NodeHandle node, STATUS *error) {
  MS_LOG(INFO) << "Get Tuple Constant size!";
  if (error == nullptr) {
    MS_LOG(ERROR) << "Input status flag [error] is nullptr.";
    return 0;
  }
  if (res_mgr == nullptr || node == nullptr) {
    MS_LOG(ERROR) << "Input Handle [res_mgr] or [node] is nullptr.";
    *error = RET_NULL_PTR;
    return 0;
  }
  try {
    auto node_impl = GetSrcPtr<ValueNodePtr>(res_mgr, node);
    MS_EXCEPTION_IF_NULL(node_impl);
    auto val = node_impl->value();
    MS_EXCEPTION_IF_NULL(val);
    auto val_tuple = val->cast<ValueTuplePtr>();
    auto tuple_size = val_tuple->size();
    *error = RET_OK;
    return tuple_size;
  } catch (const std::exception &e) {
    MS_LOG(ERROR) << "Get Tuple Constant size failed. Error info: " << e.what();
    *error = RET_ERROR;
    return 0;
  }
}

STATUS MSTupleConstantGetValueInt64(ResMgrHandle res_mgr, const NodeHandle node, int64_t vec[], size_t size) {
  MS_LOG(INFO) << "Get Tuple Constant Value!";
  if (res_mgr == nullptr || node == nullptr || vec == nullptr) {
    MS_LOG(ERROR) << "Input Handle [res_mgr] or [node] or [vec] is nullptr.";
    return RET_NULL_PTR;
  }
  try {
    auto node_impl = GetSrcPtr<ValueNodePtr>(res_mgr, node);
    MS_EXCEPTION_IF_NULL(node_impl);
    auto val = node_impl->value();
    MS_EXCEPTION_IF_NULL(val);
    auto val_tuple = val->cast<ValueTuplePtr>();
    auto val_list = val_tuple->value();
    if (val_list.size() != size) {
      MS_LOG(ERROR) << "Invalid input vector length, it should be: " << val_list.size() << ", but got: " << size;
      return RET_ERROR;
    }
    for (size_t i = 0; i < size; i++) {
      auto val_imm = val_list[i]->cast<Int64ImmPtr>();
      vec[i] = val_imm->value();
    }
    return RET_OK;
  } catch (const std::exception &e) {
    MS_LOG(ERROR) << "Get String Constant value failed. Error info: " << e.what();
    return RET_ERROR;
  }
}

TypeId MSTypeConstantGetValue(ResMgrHandle res_mgr, const NodeHandle node, STATUS *error) {
  MS_LOG(INFO) << "Get Type Constant Value!";
  if (error == nullptr) {
    MS_LOG(ERROR) << "Input status flag [error] is nullptr.";
    return (enum TypeId)0;
  }
  if (res_mgr == nullptr || node == nullptr) {
    MS_LOG(ERROR) << "Input Handle [res_mgr] or [node] is nullptr.";
    *error = RET_NULL_PTR;
    return (enum TypeId)0;
  }
  try {
    auto node_impl = GetSrcPtr<ValueNodePtr>(res_mgr, node);
    MS_EXCEPTION_IF_NULL(node_impl);
    auto val = node_impl->value();
    MS_EXCEPTION_IF_NULL(val);
    auto val_type = val->cast<TypePtr>();
    auto ret_val = static_cast<TypeId>(val_type->type_id());
    *error = RET_OK;
    return ret_val;
  } catch (const std::exception &e) {
    MS_LOG(ERROR) << "Get Type Constant value failed. Error info: " << e.what();
    *error = RET_ERROR;
    return (enum TypeId)0;
  }
}

STATUS MSOpSetName(ResMgrHandle res_mgr, const NodeHandle node, const char *name) {
  if (res_mgr == nullptr || node == nullptr || name == nullptr) {
    MS_LOG(ERROR) << "Input Handle [res_mgr] or [node] or [name] is nullptr.";
    return RET_NULL_PTR;
  }
  auto node_impl = GetSrcPtr<CNodePtr>(res_mgr, node);
  if (node_impl == nullptr) {
    MS_LOG(ERROR) << "Get source pointer failed. Please check whether the input node is an operator node.";
    return RET_ERROR;
  }
  node_impl->set_fullname_with_scope(name);
  return RET_OK;
}

STATUS MSNodeGetName(ResMgrHandle res_mgr, const NodeHandle node, char str_buf[], size_t str_len) {
  if (res_mgr == nullptr || node == nullptr || str_buf == nullptr) {
    MS_LOG(ERROR) << "Input Handle [res_mgr] or [node] or [str_buf] is nullptr.";
    return RET_NULL_PTR;
  }
  auto node_impl = GetSrcPtr<AnfNodePtr>(res_mgr, node);
  if (node_impl == nullptr) {
    MS_LOG(ERROR) << "Get source pointer failed.";
    return RET_ERROR;
  }
  auto name = node_impl->fullname_with_scope();
  size_t valid_size = name.size() < str_len - 1 ? name.size() : str_len - 1;
  for (size_t i = 0; i < valid_size; i++) {
    str_buf[i] = name.c_str()[i];
  }
  str_buf[valid_size] = '\0';
  return RET_OK;
}
