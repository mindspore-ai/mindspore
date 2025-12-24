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

#ifndef MINDSPORE_CORE_OPS_OPS_FUNC_IMPL_PAGED_ATTENTION_H_
#define MINDSPORE_CORE_OPS_OPS_FUNC_IMPL_PAGED_ATTENTION_H_
#include <algorithm>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>
#include "mindapi/base/types.h"
#include "ops/base_operator.h"
#include "ops/ops_func_impl/op_func_impl.h"

namespace mindspore {
namespace ops {
enum PagedAttentionInputIndex : size_t {
  kPagedAttentionInputQueryIndex = 0,
  kPagedAttentionInputKeyCacheIndex,
  kPagedAttentionInputValueCacheIndex,
  kPagedAttentionInputBlockTablesIndex,
  kPagedAttentionInputContextLensIndex,
  kPagedAttentionInputAntiquantScaleIndex,
  kPagedAttentionInputAntiquantOffsetIndex,
  kPagedAttentionInputAttnMaskIndex,
  kPagedAttentionInputQueryLensIndex,
  kPagedAttentionInputAlibiMaskIndex,
  kPagedAttentionInputNumHeadIndex,
  kPagedAttentionInputScaleValueIndex,
  kPagedAttentionInputNumKVHeadIndex,
  kPagedAttentionInputKVCacheQuantModeIndex,
  kPagedAttentionInputMaskModeIndex,
  kPagedAttentionInputMlaVDimIndex,
  kPagedAttentionInputsNum
};

class OPS_API PagedAttentionFuncImpl : public OpFuncImpl {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override;
  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override;
  std::set<int64_t> GetValueDependArgIndices() const override {
    return {kPagedAttentionInputContextLensIndex, kPagedAttentionInputQueryLensIndex};
  };
};
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_OPS_FUNC_IMPL_PAGED_ATTENTION_H_
