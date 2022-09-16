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

#define USE_DEPRECATED_API
#include "tools/optimizer/fusion/matmul_mul_fusion.h"
#include <memory>
#include <vector>
#include "ops/fusion/mat_mul_fusion.h"
#include "ops/fusion/mul_fusion.h"
#include "tools/optimizer/common/gllo_utils.h"
#include "nnacl/op_base.h"
#include "ops/op_utils.h"

namespace mindspore::opt {
namespace {
constexpr int64_t KMatmulWeightDims = 2;
constexpr size_t kMatMulNonBatchDims = 2;
constexpr size_t kSecondToLastDim = 2;
int CalNewCnodeScale(const CNodePtr &mul_cnode, const CNodePtr &matmul_cnode) {
  auto mul_weight_node = mul_cnode->input(kInputIndexTwo);
  std::shared_ptr<tensor::Tensor> mul_weight_tensor = GetTensorInfo(mul_weight_node);
  MS_CHECK_TRUE_RET(mul_weight_tensor != nullptr, RET_ERROR);
  if (mul_weight_tensor->data_type() != kNumberTypeFloat32) {
    MS_LOG(ERROR) << "only support float32 data type";
    return RET_ERROR;
  }

  auto mul_weight_data = reinterpret_cast<float *>(mul_weight_tensor->data_c());
  MS_CHECK_TRUE_RET(mul_weight_data != nullptr, RET_ERROR);

  auto matmul_weight_node = matmul_cnode->input(kInputIndexTwo);
  std::shared_ptr<tensor::Tensor> matmul_weight_tensor = GetTensorInfo(matmul_weight_node);
  MS_CHECK_TRUE_RET(matmul_weight_tensor != nullptr, RET_ERROR);
  if (matmul_weight_tensor->data_type() != kNumberTypeFloat32) {
    MS_LOG(ERROR) << "only support float32 data type";
    return RET_ERROR;
  }
  std::vector<int64_t> matmul_weight_shape = matmul_weight_tensor->shape();
  auto matmul_weight_data = reinterpret_cast<float *>(matmul_weight_tensor->data_c());
  MS_CHECK_TRUE_RET(matmul_weight_data != nullptr, RET_ERROR);

  auto matmul_prim = ops::GetOperator<ops::MatMulFusion>(matmul_cnode->input(0));
  MS_CHECK_TRUE_RET(matmul_prim->GetAttr(ops::kTransposeB) != nullptr, RET_ERROR);
  bool transpose_b = matmul_prim->get_transpose_b();

  int64_t bacth_size = 1;
  for (size_t i = 0; i < matmul_weight_shape.size() - kMatMulNonBatchDims; ++i) {
    bacth_size *= matmul_weight_shape[i];
  }
  int64_t row = matmul_weight_shape[matmul_weight_shape.size() - kMatMulNonBatchDims];
  int64_t col = matmul_weight_shape[matmul_weight_shape.size() - 1];
  for (int64_t i = 0; i < bacth_size; ++i) {
    for (int64_t j = 0; j < row; ++j) {
      for (int64_t k = 0; k < col; ++k) {
        if (transpose_b) {
          matmul_weight_data[i * row * col + j * col + k] *= mul_weight_data[j];
        } else {
          matmul_weight_data[i * row * col + j * col + k] *= mul_weight_data[k];
        }
      }
    }
  }
  return RET_OK;
}

int CalNewCnodeBias(const CNodePtr &mul_cnode, const CNodePtr &matmul_cnode) {
  auto mul_weight_node = mul_cnode->input(kInputIndexTwo);
  std::shared_ptr<tensor::Tensor> mul_weight_tensor = GetTensorInfo(mul_weight_node);
  MS_CHECK_TRUE_MSG(mul_weight_tensor != nullptr, RET_ERROR, "node's weight is invalid,please check your model file");
  if (mul_weight_tensor->data_type() != kNumberTypeFloat32) {
    MS_LOG(ERROR) << "only support float32 data type";
    return RET_ERROR;
  }
  std::vector<int64_t> mul_weight_shape = mul_weight_tensor->shape();
  auto mul_weight_data = reinterpret_cast<float *>(mul_weight_tensor->data_c());
  MS_CHECK_TRUE_RET(mul_weight_data != nullptr, RET_ERROR);

  auto mutmul_bias_node = matmul_cnode->input(kInputIndexThree);
  auto mutmul_bias_tensor = GetTensorInfo(mutmul_bias_node);
  if (mutmul_bias_tensor->data_type() != kNumberTypeFloat32) {
    MS_LOG(ERROR) << "only support float32 data type";
    return RET_ERROR;
  }
  std::vector<int64_t> matmul_bias_shape = mutmul_bias_tensor->shape();
  MS_CHECK_TRUE_RET(matmul_bias_shape[0] == mul_weight_shape[0], RET_ERROR);
  float *mutmul_bias_data = reinterpret_cast<float *>(mutmul_bias_tensor->data_c());
  MS_CHECK_TRUE_RET(mutmul_bias_data != nullptr, RET_ERROR);

  int64_t elem_size = matmul_bias_shape[0];
  for (int64_t i = 0; i < elem_size; ++i) {
    mutmul_bias_data[i] = mutmul_bias_data[i] * mul_weight_data[i];
  }
  return RET_OK;
}

bool IsPrimitiveProper(const CNodePtr &mul_cnode, const CNodePtr &matmul_cnode) {
  if (!CheckPrimitiveType(mul_cnode, prim::kPrimMulFusion)) {
    MS_LOG(INFO) << mul_cnode->fullname_with_scope() << "is not mul node";
    return false;
  }
  auto mul_weight_node = mul_cnode->input(kInputIndexTwo);
  if (!IsParamNode(mul_weight_node)) {
    MS_LOG(INFO) << mul_weight_node->fullname_with_scope() << "'s weight is not parameter";
    return false;
  }

  auto mul_primc = GetValueNode<PrimitiveCPtr>(mul_cnode->input(0));
  MS_CHECK_TRUE_RET(mul_primc != nullptr, false);
  if (IsQuantParameterNode(mul_primc)) {
    MS_LOG(INFO) << mul_cnode->fullname_with_scope() << " is quant node";
    return false;
  }

  if (!CheckPrimitiveType(matmul_cnode, prim::kPrimMatMulFusion)) {
    MS_LOG(INFO) << matmul_cnode->fullname_with_scope() << "is not matmul node";
    return false;
  }

  auto matmul_weight_node = matmul_cnode->input(kInputIndexTwo);
  if (!IsParamNode(matmul_weight_node)) {
    MS_LOG(INFO) << matmul_cnode->fullname_with_scope() << "'s weight is not parameter";
    return false;
  }

  if (matmul_cnode->size() > kInputSizeThree) {
    auto matmul_bias_node = matmul_cnode->input(kInputIndexThree);
    if (!IsParamNode(matmul_bias_node)) {
      MS_LOG(INFO) << matmul_cnode->fullname_with_scope() << "'s bias is not parameter";
      return false;
    }
  }

  auto matmul_primc = ops::GetOperator<ops::MatMulFusion>(matmul_cnode->input(0));
  MS_CHECK_TRUE_RET(matmul_primc != nullptr, false);
  if (matmul_primc->GetAttr(ops::kActivationType) != nullptr &&
      matmul_primc->get_activation_type() != ActivationType::NO_ACTIVATION) {
    MS_LOG(INFO) << matmul_cnode->fullname_with_scope() << " has activation attr";
    return false;
  }
  auto matmul_prim_c = matmul_primc->GetPrim();
  MS_CHECK_TRUE_RET(matmul_prim_c != nullptr, false);
  if (IsQuantParameterNode(matmul_prim_c)) {
    MS_LOG(INFO) << matmul_cnode->fullname_with_scope() << "is quant node";
    return false;
  }

  std::shared_ptr<tensor::Tensor> mul_weight_tensor = GetTensorInfo(mul_weight_node);
  MS_CHECK_TRUE_RET(mul_weight_tensor != nullptr, false);
  std::vector<int64_t> mul_weight_shape = mul_weight_tensor->shape();
  if (mul_weight_shape.size() != 1) {
    MS_LOG(INFO) << mul_cnode->fullname_with_scope() << "'s weight must be 1 dim";
    return false;
  }
  std::shared_ptr<tensor::Tensor> matmul_weight_tensor = GetTensorInfo(matmul_weight_node);
  MS_CHECK_TRUE_RET(matmul_weight_tensor != nullptr, false);
  std::vector<int64_t> matmul_weight_shape = matmul_weight_tensor->shape();
  MS_CHECK_TRUE_RET(matmul_weight_shape.size() >= KMatmulWeightDims, false);
  auto matmul_prim = ops::GetOperator<ops::MatMulFusion>(matmul_cnode->input(0));
  MS_CHECK_TRUE_RET(matmul_prim->GetAttr(ops::kTransposeB) != nullptr, false);
  int64_t last_dim_size = matmul_prim->get_transpose_b()
                            ? matmul_weight_shape[matmul_weight_shape.size() - kSecondToLastDim]
                            : matmul_weight_shape[matmul_weight_shape.size() - 1];
  if (mul_weight_shape[0] != last_dim_size) {
    MS_LOG(INFO) << "only support weight's dim is equal to the last dim of matmul output";
    return false;
  }
  return true;
}
}  // namespace

const BaseRef MatMulMulFusion::DefinePattern() const {
  auto is_mul = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimMulFusion>);
  MS_CHECK_TRUE_RET(is_mul != nullptr, {});
  auto is_matmul = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimMatMulFusion>);
  MS_CHECK_TRUE_RET(is_matmul != nullptr, {});
  auto is_param = std::make_shared<CondVar>(IsParamNode);
  MS_CHECK_TRUE_RET(is_param != nullptr, {});
  auto is_seq_var = std::make_shared<SeqVar>();
  MS_CHECK_TRUE_RET(is_seq_var != nullptr, {});
  VectorRef pattern_ref = VectorRef({is_mul, is_matmul, is_param, is_seq_var});
  return pattern_ref;
}

const AnfNodePtr MatMulMulFusion::Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                          const EquivPtr &equiv) const {
  if (func_graph == nullptr || node == nullptr) {
    return nullptr;
  }
  auto mul_cnode = node->cast<CNodePtr>();
  if (mul_cnode == nullptr || mul_cnode->size() != kInputSizeThree) {
    return nullptr;
  }

  auto matmul_node = mul_cnode->input(1);
  auto matmul_cnode = matmul_node->cast<CNodePtr>();
  MS_CHECK_TRUE_RET(matmul_cnode != nullptr, nullptr);
  if (IsMarkedTrainOp(mul_cnode) || IsMarkedTrainOp(matmul_cnode)) {
    return nullptr;
  }
  if (IsMultiOutputTensors(func_graph, matmul_cnode)) {
    return nullptr;
  }

  if (!IsPrimitiveProper(mul_cnode, matmul_cnode)) {
    return nullptr;
  }

  if (matmul_cnode->size() == kInputSizeFour && CalNewCnodeBias(mul_cnode, matmul_cnode) != RET_OK) {
    MS_LOG(ERROR) << mul_cnode->fullname_with_scope() << " failed to fusion bias with "
                  << matmul_cnode->fullname_with_scope();
    return nullptr;
  }
  if (CalNewCnodeScale(mul_cnode, matmul_cnode) != RET_OK) {
    MS_LOG(ERROR) << mul_cnode->fullname_with_scope() << " failed to fusion with "
                  << matmul_cnode->fullname_with_scope();
    return nullptr;
  }

  auto mul_primc = ops::GetOperator<ops::MulFusion>(mul_cnode->input(0));
  MS_CHECK_TRUE_RET(mul_primc != nullptr, nullptr);
  auto matmul_primc = ops::GetOperator<ops::MatMulFusion>(matmul_cnode->input(0));
  MS_CHECK_TRUE_RET(matmul_primc != nullptr, nullptr);
  if (mul_primc->GetAttr(ops::kActivationType) != nullptr &&
      mul_primc->get_activation_type() != ActivationType::NO_ACTIVATION) {
    matmul_primc->set_activation_type(mul_primc->get_activation_type());
  }
  // delete mul node
  auto manager = func_graph->manager();
  MS_CHECK_TRUE_RET(manager != nullptr, nullptr);
  (void)manager->Replace(mul_cnode, mul_cnode->input(1));
  return nullptr;
}
}  // namespace mindspore::opt
