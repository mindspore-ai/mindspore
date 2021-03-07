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
#include "tools/optimizer/fusion/batchmatmul_fusion.h"
#include <memory>
#include <vector>
#include <algorithm>
#include "ops/mat_mul.h"
#include "schema/inner/model_generated.h"
#include "src/param_value_lite.h"
#include "utils/utils.h"
#include "tools/converter/quant_param_holder.h"
#include "tools/optimizer/common/gllo_utils.h"
#include "securec/include/securec.h"

namespace mindspore::opt {
namespace {
bool IsStackNode(const BaseRef &n) {
  if (utils::isa<AnfNodePtr>(n)) {
    return CheckPrimitiveType(utils::cast<AnfNodePtr>(n), prim::kPrimStack);
  }
  return false;
}
bool IsFullConnectNode(const BaseRef &n) {
  if (utils::isa<AnfNodePtr>(n)) {
    return CheckPrimitiveType(utils::cast<AnfNodePtr>(n), prim::kPrimFullConnection);
  }
  return false;
}
void *GetInputAddr(const AnfNodePtr &node, size_t input_index) {
  MS_ASSERT(node != nullptr);
  if (!node->isa<CNode>()) {
    MS_LOG(ERROR) << "GetInputAddr not cnode";
    return nullptr;
  }
  auto cnode = node->cast<CNodePtr>();
  if (input_index >= cnode->inputs().size()) {
    MS_LOG(ERROR) << "input index error";
    return nullptr;
  }
  if (cnode->input(input_index)->isa<Parameter>()) {
    auto param_input = cnode->input(input_index)->cast<ParameterPtr>();
    auto param_value = std::dynamic_pointer_cast<ParamValueLite>(param_input->default_param());
    if (param_value == nullptr) {
      MS_LOG(ERROR) << "param not paramValueLite";
      return nullptr;
    }
    return param_value->tensor_addr();
  }
  MS_LOG(ERROR) << "input not parameter";
  return nullptr;
}
STATUS GetRightMatmulInputParamter(const CNodePtr &stack_node, const ParameterPtr &rmatmul_input) {
  MS_ASSERT(stack_node != nullptr);
  MS_ASSERT(right_matmul_input != nullptr);
  auto joint_fullconnect_size = stack_node->inputs().size() - 1;
  auto fc = stack_node->input(1)->cast<CNodePtr>();
  auto fc_weight = fc->input(2)->cast<ParameterPtr>();
  auto fc_weight_param = std::dynamic_pointer_cast<ParamValueLite>(fc_weight->default_param());
  auto tensor_size = fc_weight_param->tensor_size();
  auto rmatmul_input_shape = fc_weight_param->tensor_shape();
  auto new_tensor_data = new (std::nothrow) int8_t[joint_fullconnect_size * tensor_size];
  if (new_tensor_data == nullptr) {
    MS_LOG(ERROR) << "tensor_data is nullptr";
    return RET_ERROR;
  }
  for (size_t i = 1; i < joint_fullconnect_size + 1; i++) {
    auto tensor_addr = GetInputAddr(stack_node->input(i), 2);
    if (tensor_addr == nullptr) {
      MS_LOG(ERROR) << "input tensor addr nullptr";
      delete[] new_tensor_data;
      return RET_ERROR;
    }
    if (EOK != memcpy_s(new_tensor_data + (i - 1) * tensor_size, tensor_size, tensor_addr, tensor_size)) {
      MS_LOG(ERROR) << "memcpy_s data failed";
      delete[] new_tensor_data;
      return RET_ERROR;
    }
  }
  rmatmul_input_shape.insert(rmatmul_input_shape.begin(), joint_fullconnect_size);
  auto type_ptr = TypeIdToType(fc_weight_param->tensor_type());
  std::vector<int64_t> shape_vector;
  (void)std::transform(rmatmul_input_shape.begin(), rmatmul_input_shape.end(), std::back_inserter(shape_vector),
                       [](const int32_t &value) { return static_cast<int64_t>(value); });
  auto abstract_tensor = std::make_shared<abstract::AbstractTensor>(type_ptr, shape_vector);
  rmatmul_input->set_abstract(abstract_tensor);
  rmatmul_input->set_name(stack_node->fullname_with_scope() + "right_parameter");
  ParamValueLitePtr param_value = std::make_shared<ParamValueLite>();
  MS_ASSERT(param_value != nullptr);
  param_value->set_tensor_shape(rmatmul_input_shape);
  param_value->set_tensor_type(fc_weight_param->tensor_type());
  param_value->set_format(fc_weight_param->format());
  param_value->SetTensorData(new_tensor_data, joint_fullconnect_size * tensor_size);
  rmatmul_input->set_default_param(param_value);
  return RET_OK;
}
}  // namespace
const BaseRef BatchMatMulFusion::DefinePattern() const {
  auto pack_var = std::make_shared<CondVar>(IsStackNode);
  auto left_fullconnect_var = std::make_shared<CondVar>(IsFullConnectNode);
  auto right_fullconnect_var = std::make_shared<CondVar>(IsFullConnectNode);
  auto other_fullconnect_var = std::make_shared<SeqVar>();
  return VectorRef({pack_var, left_fullconnect_var, right_fullconnect_var, other_fullconnect_var});
}

// slice +fullconnect ->batchmatmul
const AnfNodePtr BatchMatMulFusion::Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                            const EquivPtr &) const {
  MS_ASSERT(func_graph != nullptr);
  MS_ASSERT(node != nullptr);
  auto stack_cnode = node->cast<CNodePtr>();
  // check stack node all inputs must fullconnect
  for (size_t i = 1; i < stack_cnode->inputs().size(); i++) {
    auto input_node = stack_cnode->input(i);
    if (!IsFullConnectNode(input_node)) {
      MS_LOG(WARNING) << "batchmatmulfusion stack node all inputs must fullconnect type";
      return nullptr;
    }
  }
  auto fullconnect_node = stack_cnode->input(1);
  MS_ASSERT(fullconnnect_node != nullptr);
  auto fullconnect_cnode = fullconnect_node->cast<CNodePtr>();
  MS_ASSERT(fullconnect_cnode->inputs().size() == 3);
  auto left_slice_node = fullconnect_cnode->input(1);
  auto left_slice_cnode = left_slice_node->cast<CNodePtr>();
  if (!CheckPrimitiveType(left_slice_cnode, prim::kPrimSliceFusion)) {
    return nullptr;
  }
  auto left_matmul_input = left_slice_cnode->input(1);
  auto right_reshape_node = fullconnect_cnode->input(2);

  auto matmul_cvalue = new (std::nothrow) mindspore::ops::MatMul();
  if (matmul_cvalue == nullptr) {
    MS_LOG(ERROR) << "new MatMul failed";
    return nullptr;
  }
  // get matmul quantParams
  std::vector<schema::QuantParamT> jointed_quant_params;
  for (size_t i = 1; i < stack_cnode->inputs().size(); i++) {
    auto fullconnect_node2 = stack_cnode->input(i)->cast<CNodePtr>();
    auto fc_prim = GetValueNode<PrimitiveCPtr>(fullconnect_node2->input(0));
    auto fc_input_quantParams_valueptr = fc_prim->GetAttr("quant_params");
    if (fc_input_quantParams_valueptr == nullptr) {
      continue;
    }
    auto fc_input_quantParams_holder = fc_input_quantParams_valueptr->cast<lite::QuantParamHolderPtr>();
    if (fc_input_quantParams_holder == nullptr) {
      MS_LOG(ERROR) << "quant param is invalid.";
      return nullptr;
    }
    auto fc_input_quantParams = fc_input_quantParams_holder->input_quant_params();
    if (fc_input_quantParams.size() > 1 && !fc_input_quantParams[1].empty()) {
      jointed_quant_params.push_back(fc_input_quantParams[1][0]);
    }
  }
  auto quant_params_holder = std::make_shared<lite::QuantParamHolder>();
  auto fc_prim = GetValueNode<PrimitiveCPtr>(fullconnect_cnode->input(0));
  lite::QuantParamsVector rmatmul_quant_params;
  auto rmatmul_quant_params_valueptr = fc_prim->GetAttr("quant_params");
  if (rmatmul_quant_params_valueptr != nullptr) {
    auto rmatmul_quant_params_holder = rmatmul_quant_params_valueptr->cast<lite::QuantParamHolderPtr>();
    if (rmatmul_quant_params_holder == nullptr) {
      MS_LOG(ERROR) << "quant param is invalid.";
      return nullptr;
    }
    rmatmul_quant_params = rmatmul_quant_params_holder->input_quant_params();
    quant_params_holder->set_output_quant_params(rmatmul_quant_params_holder->output_quant_params());
  }
  rmatmul_quant_params.pop_back();
  rmatmul_quant_params.pop_back();
  // no bias quantParams
  rmatmul_quant_params.emplace_back(jointed_quant_params);
  quant_params_holder->set_input_quant_params(rmatmul_quant_params);
  matmul_cvalue->AddAttr("quant_params", quant_params_holder);
  auto matmul_value_node = NewValueNode(std::shared_ptr<ops::PrimitiveC>(matmul_cvalue));
  std::vector<AnfNodePtr> matmul_inputs = {matmul_value_node, left_matmul_input};

  // batchmatmul right node may be const
  if (right_reshape_node->isa<Parameter>()) {
    auto rmatmul_paramter = func_graph->add_parameter();
    if (GetRightMatmulInputParamter(stack_cnode, rmatmul_paramter) != RET_OK) {
      MS_LOG(ERROR) << "GetRightMatmulInputParamter failed";
      return node;
    }
    auto prim = GetValueNode<PrimitiveCPtr>(matmul_value_node);
    MS_ASSERT(prim != nullptr);
    auto prim_matmul = prim->cast<std::shared_ptr<mindspore::ops::MatMul>>();
    MS_ASSERT(prim_matmul != nullptr);
    prim_matmul->set_transpose_b(true);
    matmul_inputs.push_back(rmatmul_paramter);
  } else {
    auto right_reshape_cnode = right_reshape_node->cast<CNodePtr>();
    MS_ASSERT(right_reshape_cnode->inputs().size() > 1);
    auto right_transpose_node = right_reshape_cnode->input(1);
    auto right_transpose_cnode = right_transpose_node->cast<CNodePtr>();
    auto right_slice_node = right_transpose_cnode->input(1);
    auto right_slice_cnode = right_slice_node->cast<CNodePtr>();
    auto right_matmul_input = right_slice_cnode->input(1);
    matmul_inputs.push_back(right_matmul_input);
  }
  auto matmul_cnode = func_graph->NewCNode(matmul_inputs);
  matmul_cnode->set_fullname_with_scope("matmul_" + stack_cnode->fullname_with_scope());
  matmul_cnode->set_abstract(stack_cnode->abstract()->Clone());
  MS_LOG(INFO) << "stack node:" << stack_cnode->fullname_with_scope() << " batchmatmul fusion success";
  return matmul_cnode;
}
}  // namespace mindspore::opt
