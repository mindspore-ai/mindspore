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

#include "tools/optimizer/graph/add_tensor_array.h"
#include <vector>
#include <memory>
#include "tools/optimizer/common/gllo_utils.h"
#include "mindspore/core/base/core_ops.h"
#include "mindspore/core/abstract/dshape.h"
#include "ops/tensor_array.h"
#include "ops/tensor_array_read.h"
#include "ops/tensor_array_write.h"
#include "tools/converter/ops/ops_def.h"
#include "nnacl/op_base.h"

namespace mindspore::opt {
constexpr auto kDefaultIndex = 0;
constexpr auto kInputNodeIndex = 1;
constexpr auto kDefaultNumTensors = 1;
constexpr auto kFlowInPlaceHolder = 1;

static bool IsSupportedNode(const BaseRef &n) {
  static const std::vector<PrimitivePtr> support_list = {
    prim::kPrimAffine,
  };
  if (utils::isa<AnfNodePtr>(n)) {
    auto anf_node = utils::cast<AnfNodePtr>(n);
    MS_ASSERT(anf_node != nullptr);
    return std::any_of(support_list.begin(), support_list.end(),
                       [&anf_node](const auto &primitive) { return CheckPrimitiveType(anf_node, primitive); });
  }
  return false;
}

static int SetGraphOutput(const FuncGraphPtr &func_graph, const AnfNodePtr &tensor_array_write_node) {
  MS_ASSERT(func_graph != nullptr);
  MS_ASSERT(tensor_array_write_node != nullptr);
  // set tensor_array_write_node as graph output to keep it
  auto return_node = func_graph->get_return();
  if (!CheckPrimitiveType(return_node, prim::kPrimReturn)) {
    MS_LOG(ERROR) << "graph return node is not return";
    return lite::RET_ERROR;
  }
  auto return_cnode = return_node->cast<CNodePtr>();
  if (return_cnode == nullptr) {
    MS_LOG(ERROR) << "graph return node is not cnode";
    return lite::RET_NULL_PTR;
  }
  auto output_node = return_node->input(kInputNodeIndex);
  if (output_node == nullptr) {
    MS_LOG(ERROR) << "graph output node is null";
    return lite::RET_NULL_PTR;
  }
  auto output_cnode = output_node->cast<CNodePtr>();
  if (output_cnode == nullptr) {
    MS_LOG(ERROR) << "graph output node is not cnode";
    return lite::RET_NULL_PTR;
  }

  // for multiple output graph, add output directly
  if (CheckPrimitiveType(output_node, prim::kPrimMakeTuple)) {
    output_cnode->add_input(tensor_array_write_node);
    return lite::RET_OK;
  }

  // for single output graph, create tuple for graph output
  // make_tuple node
  auto make_tuple_prim_ptr = std::make_shared<lite::MakeTuple>();
  if (make_tuple_prim_ptr == nullptr) {
    MS_LOG(ERROR) << "make_tuple_prim_ptr is nullptr";
    return lite::RET_NULL_PTR;
  }
  auto make_tuple_vnode = NewValueNode(make_tuple_prim_ptr);
  MS_CHECK_TRUE_RET(make_tuple_vnode != nullptr, lite::RET_NULL_PTR);
  auto make_tuple_cnode = func_graph->NewCNode({make_tuple_vnode, output_node, tensor_array_write_node});
  if (make_tuple_cnode == nullptr) {
    MS_LOG(ERROR) << "NewCNode failed";
    return lite::RET_NULL_PTR;
  }
  make_tuple_cnode->set_fullname_with_scope("return tuple");

  // return node
  auto return_prim_ptr = std::make_shared<lite::Return>();
  if (return_prim_ptr == nullptr) {
    MS_LOG(ERROR) << "return_prim_ptr is nullptr";
    return lite::RET_NULL_PTR;
  }
  auto return_value_node = NewValueNode(return_prim_ptr);
  MS_CHECK_TRUE_RET(return_value_node != nullptr, lite::RET_NULL_PTR);
  auto new_return_node = func_graph->NewCNode({return_value_node, make_tuple_cnode});
  MS_CHECK_TRUE_RET(new_return_node != nullptr, lite::RET_NULL_PTR);
  new_return_node->set_fullname_with_scope(return_cnode->fullname_with_scope());
  MS_ASSERT(new_return_node != nullptr);
  func_graph->set_return(new_return_node);
  MS_ASSERT(new_return_node != nullptr);

  return lite::RET_OK;
}

const BaseRef AddTensorArray::DefinePattern() const {
  auto support_detect = std::make_shared<CondVar>(IsSupportedNode);
  MS_ASSERT(support_detect != nullptr);
  auto inputs_var = std::make_shared<SeqVar>();
  MS_ASSERT(inputs_var != nullptr);
  return VectorRef({support_detect, inputs_var});
}

const AnfNodePtr AddTensorArray::Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                         const EquivPtr &equiv) const {
  if (func_graph == nullptr || node == nullptr || equiv == nullptr) {
    lite::ReturnCode::GetSingleReturnCode()->UpdateReturnCode(lite::RET_NULL_PTR);
    return nullptr;
  }
  if (!IsSupportedNode(node)) {
    MS_LOG(ERROR) << "the layer processed by affine fusion is not supported.";
    return nullptr;
  }
  auto cnode = node->cast<CNodePtr>();
  if (cnode == nullptr) {
    return nullptr;
  }
  MS_LOG(INFO) << "supported node detected:  " << cnode->fullname_with_scope();

  auto abstract = cnode->abstract();
  if (abstract == nullptr) {
    MS_LOG(ERROR) << "The abstract tensor is null.";
    lite::ReturnCode::GetSingleReturnCode()->UpdateReturnCode(lite::RET_NULL_PTR);
    return nullptr;
  }

  if (!utils::isa<abstract::AbstractTensorPtr>(abstract)) {
    MS_LOG(DEBUG) << "Abstract of parameter should be abstract tensor";
    lite::ReturnCode::GetSingleReturnCode()->UpdateReturnCode(lite::RET_PARAM_INVALID);
    return nullptr;
  }

  auto abstract_tensor = utils::cast<abstract::AbstractTensorPtr>(abstract);
  MS_ASSERT(abstract_tensor != nullptr);
  if (!utils::isa<tensor::TensorPtr>(abstract_tensor->GetValueTrack())) {  // input node not complete infershape
    MS_LOG(DEBUG) << "Value of abstract is not tensor::Tensor, indicate that infershape has failed";
    return nullptr;
  }
  auto tensor_info = utils::cast<tensor::TensorPtr>(abstract_tensor->GetValueTrack());
  MS_ASSERT(tensor_info != nullptr);
  if (tensor_info->data_type() == kObjectTypeTensorType) {
    MS_LOG(ERROR) << "tensor::Tensor of abstract is nullptr";
    lite::ReturnCode::GetSingleReturnCode()->UpdateReturnCode(lite::RET_NOT_SUPPORT);
    return nullptr;
  }

  // tensor_array
  auto tensor_array = std::make_shared<ops::TensorArray>();
  MS_CHECK_TRUE_RET(tensor_array != nullptr, nullptr);
  std::vector<int> element_shape;
  std::for_each(tensor_info->shape().begin(), tensor_info->shape().end(),
                [&element_shape](int64_t v) { element_shape.push_back(static_cast<int>(v)); });
  tensor_array->set_element_shape(element_shape);
  tensor_array->set_data_type(tensor_info->data_type());
  auto tensor_array_vnode = NewValueNode(tensor_array);
  MS_CHECK_TRUE_RET(tensor_array_vnode != nullptr, nullptr);
  auto num_tensors_vnode = NewValueNode(kDefaultNumTensors);
  MS_CHECK_TRUE_RET(num_tensors_vnode != nullptr, nullptr);
  auto tensor_array_node = func_graph->NewCNode({tensor_array_vnode, num_tensors_vnode});
  MS_ASSERT(tensor_array_node != nullptr);
  tensor_array_node->set_abstract(abstract->Clone());
  tensor_array_node->set_fullname_with_scope(cnode->fullname_with_scope() + "_tensor_array");

  // {"handle", "index", "flow_in"} -> {"tensor"}
  auto tensor_array_read = std::make_shared<ops::TensorArrayRead>();
  MS_CHECK_TRUE_RET(tensor_array_read != nullptr, nullptr);
  auto tensor_array_read_vnode = NewValueNode(tensor_array_read);
  MS_CHECK_TRUE_RET(tensor_array_read_vnode != nullptr, nullptr);
  auto read_index_vnode = NewValueNode(kDefaultIndex);
  MS_CHECK_TRUE_RET(read_index_vnode != nullptr, nullptr);
  auto read_flow_in_vnode = NewValueNode(kFlowInPlaceHolder);
  MS_CHECK_TRUE_RET(read_flow_in_vnode != nullptr, nullptr);
  auto tensor_array_read_node =
    func_graph->NewCNode({tensor_array_read_vnode, tensor_array_node, read_index_vnode, read_flow_in_vnode});
  MS_ASSERT(tensor_array_read_node != nullptr);
  tensor_array_read_node->set_abstract(abstract->Clone());
  tensor_array_read_node->set_fullname_with_scope(cnode->fullname_with_scope() + "_tensor_array_read");
  cnode->add_input(tensor_array_read_node);

  // {"handle", "index", "value", "flow_in"} -> {"flow_out"}
  auto tensor_array_write = std::make_shared<ops::TensorArrayWrite>();
  MS_CHECK_TRUE_RET(tensor_array_write != nullptr, nullptr);
  auto tensor_array_write_vnode = NewValueNode(tensor_array_write);
  MS_CHECK_TRUE_RET(tensor_array_write_vnode != nullptr, nullptr);
  auto write_index_vnode = NewValueNode(kDefaultIndex);
  MS_CHECK_TRUE_RET(write_index_vnode != nullptr, nullptr);
  auto write_flow_in_vnode = NewValueNode(kFlowInPlaceHolder);
  MS_CHECK_TRUE_RET(write_flow_in_vnode != nullptr, nullptr);
  auto tensor_array_write_node =
    func_graph->NewCNode({tensor_array_write_vnode, tensor_array_node, write_index_vnode, cnode, write_flow_in_vnode});
  if (tensor_array_write_node == nullptr) {
    MS_LOG(ERROR) << "rensor_array_write_node is nullptr";
    return nullptr;
  }
  tensor_array_write_node->set_abstract(abstract->Clone());
  tensor_array_write_node->set_fullname_with_scope(cnode->fullname_with_scope() + "_tensor_array_write");

  if (int status = SetGraphOutput(func_graph, tensor_array_write_node); status != lite::RET_OK) {
    MS_LOG(ERROR) << "tensor::Tensor of abstract is nullptr";
    lite::ReturnCode::GetSingleReturnCode()->UpdateReturnCode(status);
    return nullptr;
  }

  return node;
}
}  // namespace mindspore::opt
