/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CORE_OPS_FUNC_IMPL_FUSED_INFER_ATTENTION_SCORE_H_
#define MINDSPORE_CORE_OPS_FUNC_IMPL_FUSED_INFER_ATTENTION_SCORE_H_
#include <vector>
#include <set>
#include "ops/ops_func_impl/op_func_impl.h"

namespace mindspore {
namespace ops {
enum FusedInferAttentionScoreInputIndex : size_t {
  kFusedInferAttentionScoreInputQueryIndex = 0,
  kFusedInferAttentionScoreInputKeyIndex,
  kFusedInferAttentionScoreInputValueIndex,
  kFusedInferAttentionScoreInputPseShiftIndex,
  kFusedInferAttentionScoreInputAttnMaskIndex,
  kFusedInferAttentionScoreInputActualSeqLengthsIndex,
  kFusedInferAttentionScoreInputActualSeqLengthsKvIndex,
  kFusedInferAttentionScoreInputDequantScale1Index,
  kFusedInferAttentionScoreInputQuantScale1Index,
  kFusedInferAttentionScoreInputDequantScale2Index,
  kFusedInferAttentionScoreInputQuantScale2Index,
  kFusedInferAttentionScoreInputQuantOffset2Index,
  kFusedInferAttentionScoreInputAntiquantScaleIndex,
  kFusedInferAttentionScoreInputAntiquantOffsetIndex,
  kFusedInferAttentionScoreInputBlockTableIndex,
  kFusedInferAttentionScoreInputQueryPaddingSizeIndex,
  kFusedInferAttentionScoreInputKvPaddingSizeIndex,
  kFusedInferAttentionScoreInputKeyAntiquantScaleIndex,
  kFusedInferAttentionScoreInputKeyAntiquantOffsetIndex,
  kFusedInferAttentionScoreInputValueAntiquantScaleIndex,
  kFusedInferAttentionScoreInputValueAntiquantOffsetIndex,
  kFusedInferAttentionScoreInputKeySharedPrefixIndex,
  kFusedInferAttentionScoreInputValueSharedPrefixIndex,
  kFusedInferAttentionScoreInputActualSharedPrefixLenIndex,
  // attrs
  kFusedInferAttentionScoreInputNumHeadsIndex,
  kFusedInferAttentionScoreInputScaleIndex,
  kFusedInferAttentionScoreInputPreTokensIndex,
  kFusedInferAttentionScoreInputNextTokensIndex,
  kFusedInferAttentionScoreInputLayoutIndex,
  kFusedInferAttentionScoreInputNumKeyValueHeadsIndex,
  kFusedInferAttentionScoreInputSparseModeIndex,
  kFusedInferAttentionScoreInputInnerPreciseIndex,
  kFusedInferAttentionScoreInputBlockSizeIndex,
  kFusedInferAttentionScoreInputAntiquantModeIndex,
  kFusedInferAttentionScoreInputSoftmaxLseFlagIndex,
  kFusedInferAttentionScoreInputKeyAntiquantModeIndex,
  kFusedInferAttentionScoreInputValueAntiquantModeIndex,
  kFusedInferAttentionScoreInputsNum,
};
enum FusedInferAttentionScoreOutputIndex : size_t {
  kFusedInferAttentionScoreOutputAttentionOutIndex = 0,
  kFusedInferAttentionScoreOutputSoftmaxLseIndex,
  kFusedInferAttentionScoreOutputsNum,
};

class OPS_API FusedInferAttentionScoreFuncImpl : public OpFuncImpl {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override;
  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override;
  std::set<int64_t> GetValueDependArgIndices() const override;
};
}  // namespace ops
}  // namespace mindspore
#endif  // MINDSPORE_CORE_OPS_FUNC_IMPL_FUSED_INFER_ATTENTION_SCORE_H_
