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
#ifndef MINDSPORE_CORE_OPS_OPS_FUNC_IMPL_FLASH_ATTENTION_SCORE_H_
#define MINDSPORE_CORE_OPS_OPS_FUNC_IMPL_FLASH_ATTENTION_SCORE_H_

#include <vector>
#include "ops/ops_func_impl/op_func_impl.h"

namespace mindspore {
namespace ops {
enum FlashAttentionScoreInputIndex : size_t {
  kFlashAttentionScoreInputQueryIndex = 0,
  kFlashAttentionScoreInputKeyIndex,
  kFlashAttentionScoreInputValueIndex,
  kFlashAttentionScoreInputRealShiftIndex,
  kFlashAttentionScoreInputDropMaskIndex,
  kFlashAttentionScoreInputPaddingMaskIndex,
  kFlashAttentionScoreInputAttnMaskIndex,
  kFlashAttentionScoreInputPrefixIndex,
  kFlashAttentionScoreInputActualSeqQlenIndex,
  kFlashAttentionScoreInputActualSeqKVlenIndex,
  kFlashAttentionScoreInputHeadNumIndex,
  kFlashAttentionScoreInputKeepProbIndex,
  kFlashAttentionScoreInputScaleValueIndex,
  kFlashAttentionScoreInputPreTokensIndex,
  kFlashAttentionScoreInputNextTokensIndex,
  kFlashAttentionScoreInputInnerPreciseIndex,
  kFlashAttentionScoreInputLayoutIndex,
  kFlashAttentionScoreInputSparseModeIndex,
  kFlashAttentionScoreInputsNum,
};
enum FlashAttentionScoreOutputIndex : size_t {
  kFlashAttentionScoreOutputSoftmaxMaxIndex = 0,
  kFlashAttentionScoreOutputSoftmaxSumIndex,
  kFlashAttentionScoreOutputSoftmaxOutIndex,
  kFlashAttentionScoreOutputAttentionOutIndex,
  kFlashAttentionScoreOutputsNum,
};
enum FlashAttentionScoreSparseMode : int64_t {
  kSparseDefaultMask = 0,
  kSparseAllMask,
  kSparseLeftUpCausal,
  kSparseRightDownCausal,
  kSparseBand,
  kSparsePrefix,
  kSparseGlobal,
  kSparseDilated,
  kSparseBlockLocal,
};
class MIND_API FlashAttentionScoreFuncImpl : public OpFuncImpl {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override;
  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override;
};
}  // namespace ops
}  // namespace mindspore
#endif  // MINDSPORE_CORE_OPS_OPS_FUNC_IMPL_FLASH_ATTENTION_SCORE_H_
