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
#include "tools/optimizer/fusion/matmul_scale_fusion.h"
#include <memory>
#include <vector>
#include "ops/mat_mul.h"
#include "tools/optimizer/common/gllo_utils.h"
#include "tools/converter/quantizer/quant_param_holder.h"
#include "nnacl/op_base.h"
#include "ops/op_utils.h"

namespace mindspore::opt {
namespace {
constexpr size_t kMatMulNonBatchDims = 2;
}  // namespace

PrimitiveCPtr MatMulScaleFusion::BuildNewPrimitive(const CNodePtr &, const CNodePtr &prev_cnode) const {
  auto prim = prev_cnode->input(0);
  MS_CHECK_TRUE_MSG(prim != nullptr, nullptr, "failed to create matmul primitive");
  auto matmul_primc = GetValueNode<PrimitiveCPtr>(prim);
  MS_CHECK_TRUE_RET(matmul_primc != nullptr, nullptr);

  return matmul_primc;
}

int MatMulScaleFusion::CalNewBiasImpl(float *curr_weight_data, float *curr_bias_data,
                                      std::vector<int64_t> prev_bias_shape, float *prev_bias_data) const {
  int64_t elem_size = prev_bias_shape[0];
  for (int64_t i = 0; i < elem_size; ++i) {
    prev_bias_data[i] = prev_bias_data[i] * curr_weight_data[i];
    if (curr_bias_data != nullptr) {
      prev_bias_data[i] += curr_bias_data[i];
    }
  }
  return RET_OK;
}

int MatMulScaleFusion::CalNewScaleImpl(float *curr_weight_data, std::vector<int64_t> prev_weight_shape,
                                       float *prev_weight_data, const AnfNodePtr &prim) const {
  auto matmul_prim = ops::GetOperator<ops::MatMul>(prim);
  auto trans_attr = matmul_prim->GetAttr(ops::kTransposeB);
  MS_CHECK_TRUE_RET(trans_attr != nullptr, RET_ERROR);
  bool transpose_b = matmul_prim->get_transpose_b();

  int64_t bacth_size = 1;
  for (size_t i = 0; i < prev_weight_shape.size() - kMatMulNonBatchDims; ++i) {
    bacth_size *= prev_weight_shape[i];
  }
  int64_t row = prev_weight_shape[prev_weight_shape.size() - kMatMulNonBatchDims];
  int64_t col = prev_weight_shape[prev_weight_shape.size() - 1];
  for (int64_t i = 0; i < bacth_size; ++i) {
    for (int64_t j = 0; j < row; ++j) {
      for (int64_t k = 0; k < col; ++k) {
        if (transpose_b) {
          prev_weight_data[bacth_size * row * col + j * col + k] *= curr_weight_data[j];
        } else {
          prev_weight_data[bacth_size * row * col + j * col + k] *= curr_weight_data[k];
        }
      }
    }
  }
  return RET_OK;
}

bool MatMulScaleFusion::CheckPrevCnodeProper(const CNodePtr &prev_cnode) const {
  if (!CheckPrimitiveType(prev_cnode, prim::kPrimMatMulFusion)) {
    MS_LOG(INFO) << prev_cnode->fullname_with_scope() << "is not matmul node";
    return false;
  }

  auto matmul_weight_node = prev_cnode->input(kInputIndexTwo);
  if (!IsParamNode(matmul_weight_node)) {
    MS_LOG(INFO) << prev_cnode->fullname_with_scope() << "'s weight is not parameter";
    return false;
  }

  if (prev_cnode->size() > kInputSizeThree) {
    auto prev_bias_node = prev_cnode->input(kInputIndexThree);
    if (!IsParamNode(prev_bias_node)) {
      MS_LOG(INFO) << prev_cnode->fullname_with_scope() << "'s bias is not parameter";
      return false;
    }
  }

  auto matmul_primc = GetValueNode<PrimitiveCPtr>(prev_cnode->input(0));  // previous fc primitive
  MS_CHECK_TRUE_RET(matmul_primc != nullptr, false);
  if (IsQuantParameterNode(matmul_primc)) {
    MS_LOG(INFO) << prev_cnode->fullname_with_scope() << "is quant node";
    return false;
  }

  return true;
}

const BaseRef MatMulScaleFusion::DefinePattern() const {
  auto is_fc = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimMatMulFusion>);
  MS_CHECK_TRUE_RET(is_fc != nullptr, {});
  auto is_scale = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimScaleFusion>);
  MS_CHECK_TRUE_RET(is_scale != nullptr, {});
  auto is_param = std::make_shared<CondVar>(IsParamNode);
  MS_CHECK_TRUE_RET(is_param != nullptr, {});
  auto is_seq_var = std::make_shared<SeqVar>();
  MS_CHECK_TRUE_RET(is_seq_var != nullptr, {});
  return VectorRef({is_scale, is_fc, is_param, is_seq_var});
}
}  // namespace mindspore::opt
