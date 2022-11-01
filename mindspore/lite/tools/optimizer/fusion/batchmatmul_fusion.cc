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

#define USE_DEPRECATED_API
#include "tools/optimizer/fusion/batchmatmul_fusion.h"
#include <memory>
#include <vector>
#include <algorithm>
#include "ops/fusion/mat_mul_fusion.h"
#include "tools/common/tensor_util.h"
#include "tools/converter/quantizer/quant_param_holder.h"
#include "tools/optimizer/common/gllo_utils.h"
#include "securec/include/securec.h"
#include "nnacl/op_base.h"
#include "ops/op_utils.h"

namespace mindspore::opt {
namespace {
constexpr int64_t kFcRightInputDims = 3;
constexpr float kFpPrecision = 1e-6;
void *GetInputAddr(const AnfNodePtr &node, size_t input_index) {
  MS_ASSERT(node != nullptr);
  if (!node->isa<CNode>()) {
    MS_LOG(ERROR) << "GetInputAddr not cnode";
    return nullptr;
  }
  auto cnode = node->cast<CNodePtr>();
  if (input_index >= cnode->size()) {
    MS_LOG(ERROR) << "input index error";
    return nullptr;
  }
  if (cnode->input(input_index)->isa<Parameter>()) {
    auto param_input = cnode->input(input_index)->cast<ParameterPtr>();
    MS_CHECK_TRUE_RET(param_input->default_param() != nullptr, nullptr);
    auto tensor_info = param_input->default_param()->cast<tensor::TensorPtr>();
    if (tensor_info == nullptr) {
      MS_LOG(ERROR) << "param not tensor::Tensor";
      return nullptr;
    }
    return tensor_info->data_c();
  }
  MS_LOG(ERROR) << "input not parameter";
  return nullptr;
}
STATUS GetRightMatmulInputParamter(const CNodePtr &stack_node, const ParameterPtr &rmatmul_input) {
  MS_ASSERT(stack_node != nullptr);
  MS_ASSERT(rmatmul_input != nullptr);
  auto joint_fullconnect_size = stack_node->inputs().size() - 1;
  auto fc = stack_node->input(1)->cast<CNodePtr>();
  MS_CHECK_TRUE_RET(fc != nullptr, lite::RET_NULL_PTR);
  auto fc_weight = fc->input(kInputIndexTwo)->cast<ParameterPtr>();
  MS_CHECK_TRUE_RET(fc_weight != nullptr, lite::RET_NULL_PTR);
  auto fc_weight_param = std::dynamic_pointer_cast<tensor::Tensor>(fc_weight->default_param());
  MS_CHECK_TRUE_RET(fc_weight_param != nullptr, lite::RET_NULL_PTR);
  auto tensor_size = fc_weight_param->Size();
  auto rmatmul_input_shape = fc_weight_param->shape();

  rmatmul_input_shape.insert(rmatmul_input_shape.begin(), joint_fullconnect_size);
  std::vector<int64_t> shape_vector(rmatmul_input_shape.begin(), rmatmul_input_shape.end());
  auto tensor_info = lite::CreateTensorInfo(nullptr, 0, shape_vector, fc_weight_param->data_type());
  if (tensor_info == nullptr) {
    MS_LOG(ERROR) << "Create tensor info failed";
    return RET_ERROR;
  }
  for (size_t i = 1; i < joint_fullconnect_size + 1; i++) {
    auto tensor_addr = GetInputAddr(stack_node->input(i), kInputIndexTwo);
    if (tensor_addr == nullptr) {
      MS_LOG(ERROR) << "input tensor addr nullptr";
      return RET_ERROR;
    }
    if (EOK != memcpy_s(static_cast<int8_t *>(tensor_info->data_c()) + (i - 1) * tensor_size,
                        tensor_info->Size() - (i - 1) * tensor_size, tensor_addr, tensor_size)) {
      MS_LOG(ERROR) << "memcpy_s data failed";
      return RET_ERROR;
    }
  }
  auto status = lite::InitParameterFromTensorInfo(rmatmul_input, tensor_info);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "init parameter from tensor info failed";
    return RET_ERROR;
  }
  rmatmul_input->set_name(stack_node->fullname_with_scope());

  return RET_OK;
}

std::shared_ptr<ops::MatMulFusion> BuildMatMulPrim(const CNodePtr &stack_cnode) {
  auto matmul_cvalue = std::make_shared<ops::MatMulFusion>();
  if (matmul_cvalue == nullptr) {
    MS_LOG(ERROR) << "new MatMul failed";
    return nullptr;
  }
  auto matmul_prim_c = matmul_cvalue->GetPrim();
  MS_CHECK_TRUE_RET(matmul_prim_c != nullptr, nullptr);

  auto fullconnect_node = stack_cnode->input(1);
  auto fullconnect_cnode = fullconnect_node->cast<CNodePtr>();
  MS_CHECK_TRUE_RET(fullconnect_cnode != nullptr, nullptr);
  auto fc_prim = GetValueNode<PrimitiveCPtr>(fullconnect_cnode->input(0));
  MS_ASSERT(fc_prim != nullptr);
  lite::QuantParamsVector rmatmul_quant_params;
  auto rmatmul_quant_params_valueptr = fc_prim->GetAttr("quant_params");
  lite::QuantParamsVector output_quant_params;

  MS_CHECK_TRUE_RET(rmatmul_quant_params_valueptr != nullptr, nullptr);
  auto rmatmul_quant_params_holder = rmatmul_quant_params_valueptr->cast<lite::QuantParamHolderPtr>();
  if (rmatmul_quant_params_holder == nullptr) {
    MS_LOG(ERROR) << "quant param is invalid.";
    return nullptr;
  }
  rmatmul_quant_params = rmatmul_quant_params_holder->get_input_quant_params();
  output_quant_params = rmatmul_quant_params_holder->get_output_quant_params();

  // no bias quantParams
  rmatmul_quant_params.pop_back();
  auto quant_params_holder = std::make_shared<lite::QuantParamHolder>(rmatmul_quant_params, output_quant_params);
  MS_CHECK_TRUE_RET(quant_params_holder != nullptr, nullptr);
  (void)matmul_prim_c->AddAttr("quant_params", quant_params_holder);
  return matmul_cvalue;
}

bool IsTensorZero(const tensor::TensorPtr &tensor) {
  MS_ASSERT(tensor != nullptr);
  if (tensor->data_type() != TypeId::kNumberTypeFloat32) {
    return false;
  }
  auto data = reinterpret_cast<float *>(tensor->data_c());
  for (size_t i = 0; i < tensor->DataSize(); i++) {
    if (data[i] > kFpPrecision) {
      return false;
    }
  }
  return true;
}

bool IsFCNonBias(const CNodePtr &fc) {
  MS_ASSERT(fc != nullptr);
  if (fc->inputs().size() == kInputSizeThree) {
    return true;
  }
  auto bias_input = fc->inputs().at(kInputSizeThree);
  if (utils::isa<CNodePtr>(bias_input)) {
    return false;
  } else if (utils::isa<ParameterPtr>(bias_input)) {
    auto bias_param = utils::cast<ParameterPtr>(bias_input);
    if (!bias_param->has_default()) {
      return false;
    }
    auto bias_default_param = bias_param->default_param();
    if (bias_default_param == nullptr || !utils::isa<tensor::TensorPtr>(bias_default_param)) {
      return false;
    }
    auto bias_tensor = utils::cast<tensor::TensorPtr>(bias_default_param);
    if (!IsTensorZero(bias_tensor)) {
      return false;
    }
  } else if (utils::isa<ValuePtr>(bias_input)) {
    auto bias_value = utils::cast<ValuePtr>(bias_input);
    if (!utils::isa<tensor::TensorPtr>(bias_value)) {
      return false;
    }
    auto bias_tensor = utils::cast<tensor::TensorPtr>(bias_value);
    if (!IsTensorZero(bias_tensor)) {
      return false;
    }
  }
  return true;
}

bool ConnectTransposeConcat(const AnfNodePtr &node) {
  auto cnode = node->cast<CNodePtr>();
  if (cnode == nullptr) {
    MS_LOG(ERROR) << "cnode is null";
    return false;
  }
  auto right_transpose_node = cnode->input(1);
  MS_CHECK_TRUE_RET(right_transpose_node != nullptr, false);
  auto right_transpose_cnode = right_transpose_node->cast<CNodePtr>();
  MS_CHECK_TRUE_RET(right_transpose_cnode != nullptr, false);
  if (CheckPrimitiveType(right_transpose_cnode, prim::kPrimConcat)) {
    return true;
  }
  auto front_node = right_transpose_cnode->input(1);
  MS_CHECK_TRUE_RET(front_node != nullptr, false);
  auto front_cnode = front_node->cast<CNodePtr>();
  MS_CHECK_TRUE_RET(front_cnode != nullptr, false);
  if (CheckPrimitiveType(right_transpose_cnode, prim::kPrimTranspose) &&
      CheckPrimitiveType(front_cnode, prim::kPrimConcat)) {
    return true;
  }
  return false;
}

int ResetReshapeParameters(const AnfNodePtr &reshape_node) {
  auto reshape_cnode = reshape_node->cast<CNodePtr>();
  MS_ASSERT(reshape_cnode != nullptr);
  auto reshape_shape_param = reshape_cnode->input(kInputIndexTwo)->cast<ParameterPtr>();
  MS_ASSERT(reshape_shape_param != nullptr);
  auto shape_tensor = std::dynamic_pointer_cast<tensor::Tensor>(reshape_shape_param->default_param());
  auto rmatmul_input_shape = shape_tensor->shape();

  std::vector<int64_t> shape(1, 0);
  if (rmatmul_input_shape.size() <= 0) {
    MS_LOG(ERROR) << "Create tensor info failed";
    return RET_ERROR;
  } else if (shape[0] < kFcRightInputDims) {
    shape[0] = rmatmul_input_shape[0] + 1;
  }

  auto tensor_info = std::make_shared<tensor::Tensor>(shape_tensor->data_type(), shape);
  if (tensor_info == nullptr) {
    MS_LOG(ERROR) << "Create tensor info failed";
    return RET_ERROR;
  }

  int *tensor_data = reinterpret_cast<int *>(tensor_info->data_c());
  tensor_data[0] = 1;
  int *reshape_data = reinterpret_cast<int *>(shape_tensor->data_c());
  for (int64_t i = 1; i < shape[0]; ++i) {
    tensor_data[i] = reshape_data[i - 1];
  }

  auto ret = lite::InitParameterFromTensorInfo(reshape_shape_param, tensor_info);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "init parameter from tensor info failed";
    return RET_ERROR;
  }
  return RET_OK;
}
}  // namespace

const BaseRef BatchMatMulFusion::DefinePattern() const {
  auto is_stack = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimStack>);
  MS_CHECK_TRUE_RET(is_stack != nullptr, {});
  auto is_fullconnect1 = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimFullConnection>);
  MS_CHECK_TRUE_RET(is_fullconnect1 != nullptr, {});
  auto is_fullconnect2 = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimFullConnection>);
  MS_CHECK_TRUE_RET(is_fullconnect2 != nullptr, {});
  auto is_seq_var = std::make_shared<SeqVar>();
  MS_CHECK_TRUE_RET(is_seq_var != nullptr, {});
  return VectorRef({is_stack, is_fullconnect1, is_fullconnect2, is_seq_var});
}

bool BatchMatMulFusion::CheckCnodeProper(const CNodePtr &stack_cnode, const CNodePtr &fullconnect_cnode,
                                         const CNodePtr &left_slice_cnode) const {
  if (IsMarkedTrainOp(stack_cnode)) {
    return false;
  }
  // check stack node all inputs must fullconnect
  for (size_t i = 1; i < stack_cnode->inputs().size(); i++) {
    auto input_node = stack_cnode->input(i);
    if (!CheckPrimitiveType(input_node, prim::kPrimFullConnection)) {
      MS_LOG(WARNING) << "batchmatmulfusion stack node all inputs must fullconnect type";
      return false;
    }
  }

  if (IsMarkedTrainOp(fullconnect_cnode)) {
    return false;
  }
  if (!IsFCNonBias(fullconnect_cnode)) {
    return false;
  }

  if (IsMarkedTrainOp(left_slice_cnode)) {
    return false;
  }

  if (!CheckPrimitiveType(left_slice_cnode, prim::kPrimSliceFusion)) {
    if (!CheckPrimitiveType(left_slice_cnode, prim::kPrimReshape)) {
      return false;
    }
    auto up_slice_cnode = left_slice_cnode->input(1)->cast<CNodePtr>();
    if (IsMarkedTrainOp(up_slice_cnode)) {
      return false;
    }
    if (up_slice_cnode == nullptr || !CheckPrimitiveType(up_slice_cnode, prim::kPrimSliceFusion)) {
      return false;
    }
  }
  return true;
}

const AnfNodePtr BatchMatMulFusion::Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                            const EquivPtr &) const {
  MS_CHECK_TRUE_RET(func_graph != nullptr, nullptr);
  MS_CHECK_TRUE_RET(node != nullptr, nullptr);
  auto stack_cnode = node->cast<CNodePtr>();
  auto fullconnect_node = stack_cnode->input(1);
  auto fullconnect_cnode = fullconnect_node->cast<CNodePtr>();
  MS_CHECK_TRUE_RET(fullconnect_cnode != nullptr, nullptr);
  auto left_slice_node = fullconnect_cnode->input(1);
  auto left_slice_cnode = left_slice_node->cast<CNodePtr>();
  MS_CHECK_TRUE_RET(left_slice_cnode != nullptr, nullptr);
  if (!CheckCnodeProper(stack_cnode, fullconnect_cnode, left_slice_cnode)) {
    MS_LOG(WARNING) << stack_cnode->fullname_with_scope() << " can't fusion into matmul. Fusion failed";
    return nullptr;
  }
  if (CheckPrimitiveType(left_slice_cnode, prim::kPrimReshape)) {
    auto &left_reshape_cnode = left_slice_cnode;
    left_slice_cnode = left_reshape_cnode->input(1)->cast<CNodePtr>();
  }

  // slice +fullconnect ->batchmatmul
  auto left_matmul_input = left_slice_cnode->input(1);
  auto right_reshape_node = fullconnect_cnode->input(kInputIndexTwo);
  MS_ASSERT(right_reshape_node != nullptr);
  auto matmul_cvalue = BuildMatMulPrim(stack_cnode);
  MS_CHECK_TRUE_RET(matmul_cvalue != nullptr, nullptr);
  auto matmul_value_node = NewValueNode(matmul_cvalue->GetPrim());
  MS_CHECK_TRUE_RET(matmul_value_node != nullptr, nullptr);
  std::vector<AnfNodePtr> matmul_inputs = {matmul_value_node, left_matmul_input};

  // batchmatmul right node may be const
  bool right_transpose = false;
  if (right_reshape_node->isa<Parameter>()) {
    auto rmatmul_paramter = func_graph->add_parameter();
    MS_CHECK_TRUE_RET(rmatmul_paramter != nullptr, nullptr);
    if (GetRightMatmulInputParamter(stack_cnode, rmatmul_paramter) != RET_OK) {
      MS_LOG(ERROR) << "GetRightMatmulInputParamter failed";
      return node;
    }
    auto prim_matmul = ops::GetOperator<mindspore::ops::MatMulFusion>(matmul_value_node);
    MS_ASSERT(prim_matmul != nullptr);
    prim_matmul->set_transpose_b(true);
    matmul_inputs.push_back(rmatmul_paramter);
  } else if (ConnectTransposeConcat(right_reshape_node)) {
    right_transpose = true;
    auto ret = ResetReshapeParameters(right_reshape_node);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "reset reshape parameters failed";
      return nullptr;
    }
    matmul_inputs.push_back(right_reshape_node);
  } else {
    auto right_reshape_cnode = right_reshape_node->cast<CNodePtr>();
    MS_CHECK_TRUE_RET(right_reshape_cnode != nullptr, nullptr);
    if (IsMarkedTrainOp(right_reshape_cnode)) {
      return nullptr;
    }
    MS_ASSERT(right_reshape_cnode->inputs().size() > 1);
    auto right_transpose_node = right_reshape_cnode->input(1);
    MS_CHECK_TRUE_RET(right_transpose_node != nullptr, nullptr);
    auto right_transpose_cnode = right_transpose_node->cast<CNodePtr>();
    MS_CHECK_TRUE_RET(right_transpose_cnode != nullptr, nullptr);
    auto right_slice_node = right_transpose_cnode->input(1);
    MS_CHECK_TRUE_RET(right_slice_node != nullptr, nullptr);
    auto right_slice_cnode = right_slice_node->cast<CNodePtr>();
    MS_CHECK_TRUE_RET(right_slice_cnode != nullptr, nullptr);
    auto right_matmul_input = right_slice_cnode->input(1);
    matmul_inputs.push_back(right_matmul_input);
  }
  auto matmul_cnode = func_graph->NewCNode(matmul_inputs);
  MS_CHECK_TRUE_RET(matmul_cnode != nullptr, nullptr);
  matmul_cnode->set_fullname_with_scope(stack_cnode->fullname_with_scope());
  MS_CHECK_TRUE_RET(stack_cnode->abstract() != nullptr, nullptr);
  matmul_cnode->set_abstract(stack_cnode->abstract()->Clone());
  if (right_transpose) {
    auto matmul_primitive = ops::GetOperator<ops::MatMulFusion>(matmul_cnode->input(0));
    matmul_primitive->set_transpose_b(true);
  }
  MS_LOG(INFO) << "stack node:" << stack_cnode->fullname_with_scope() << " batchmatmul fusion success";
  return matmul_cnode;
}
}  // namespace mindspore::opt
