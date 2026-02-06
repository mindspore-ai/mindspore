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
#ifndef MINDSPORE_CORE_OPS_OPS_FUNC_IMPL_FLASH_ATTENTION_SCORE_GRAD_H_
#define MINDSPORE_CORE_OPS_OPS_FUNC_IMPL_FLASH_ATTENTION_SCORE_GRAD_H_

#include <vector>
#include "ops/ops_func_impl/op_func_impl.h"

namespace mindspore {
namespace ops {
enum FASGradInputIndex : size_t {
  kFASGradInputQueryIndex = 0,
  kFASGradInputKeyIndex,
  kFASGradInputValueIndex,
  kFASGradInputDyIndex,
  kFASGradInputPseShiftIndex,
  kFASGradInputDropMaskIndex,
  kFASGradInputPaddingMaskIndex,
  kFASGradInputAttnMaskIndex,
  kFASGradInputSoftmaxMaxIndex,
  kFASGradInputSoftmaxSumIndex,
  kFASGradInputSoftmaxOutIndex,
  kFASGradInputAttentionInIndex,
  kFASGradInputPrefixIndex,
  kFASGradInputActualSeqQlenIndex,
  kFASGradInputActualSeqKVlenIndex,
  kFASGradInputHeadNumIndex,
  kFASGradInputKeepProbIndex,
  kFASGradInputScaleValueIndex,
  kFASGradInputPreTokensIndex,
  kFASGradInputNextTokensIndex,
  kFASGradInputInnerPreciseIndex,
  kFASGradInputLayoutIndex,
  kFASGradInputSparseModeIndex,
  kFASGradInputsNum,
};
enum FASGradOutputIndex : size_t {
  kFASGradOutputDqIndex = 0,
  kFASGradOutputDkIndex,
  kFASGradOutputDvIndex,
  kFASGradOutputDpseIndex,
  kFASGradOutputsNum,
};
enum FASGradSparseMode : int64_t {
  kFAGSparseDefaultMask = 0,
  kFAGSparseAllMask,
  kFAGSparseLeftUpCausal,
  kFAGSparseRightDownCausal,
  kFAGSparseBand,
  kFAGSparsePrefix,
  kFAGSparseGlobal,
  kFAGSparseDilated,
  kFAGSparseBlockLocal,
};

class OPS_API FlashAttentionScoreGradFuncImpl : public OpFuncImpl {
 public:
  ShapeArray InferShape(const PrimitivePtr &primitive, const InferInfoPtrList &input_infos) const override;
  std::vector<TypeId> InferType(const PrimitivePtr &primitive, const InferInfoPtrList &input_infos) const override;
  bool GeneralInferRegistered() const override { return true; }
};
}  // namespace ops
}  // namespace mindspore
#endif  // MINDSPORE_CORE_OPS_OPS_FUNC_IMPL_FLASH_ATTENTION_SCORE_GRAD_H_
