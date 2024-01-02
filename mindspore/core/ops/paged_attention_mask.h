/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CORE_OPS_PAGED_ATTENTION_MASK_H_
#define MINDSPORE_CORE_OPS_PAGED_ATTENTION_MASK_H_
#include <map>
#include <vector>
#include <memory>

#include "ops/base_operator.h"
#include "mindapi/base/types.h"
#include "ops/primitive_c.h"
#include "abstract/abstract_value.h"
#include "mindspore/core/ops/op_name.h"

namespace mindspore {
namespace ops {
constexpr auto kNamePagedAttentionMask = "PagedAttentionMask";
enum PagedAttentionMaskInputIndex : size_t {
  kPagedAttentionMaskInputQueryIndex = 0,
  kPagedAttentionMaskInputKeyCacheIndex,
  kPagedAttentionMaskInputValueCacheIndex,
  kPagedAttentionMaskInputBlockTablesIndex,
  kPagedAttentionMaskInputContextLensIndex,
  kPagedAttentionMaskInputAlibiMaskIndex,
  kPagedAttentionMaskInputsNum
};
enum PagedAttentionMaskOutputIndex : size_t {
  kPagedAttentionMaskOutputAttentionOutIndex = 0,
  kPagedAttentionMaskOutputsNum
};

/// \brief PagedAttentionMask.
/// Refer to Python API @ref mindspore.ops.PagedAttentionMask for more details.
class MIND_API PagedAttentionMask : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(PagedAttentionMask);
  /// \brief Constructor.
  PagedAttentionMask() : BaseOperator(kNamePagedAttentionMask) {
    InitIOName({"query", "key_cache", "value_cache", "block_tables", "context_lens", "alibi_mask"}, {"attention_out"});
  }
};

AbstractBasePtr PagedAttentionMaskInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                        const std::vector<AbstractBasePtr> &input_args);
using PagedAttentionMaskPtr = std::shared_ptr<PagedAttentionMask>;
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_PAGED_ATTENTION_MASK_H_
