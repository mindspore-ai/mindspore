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
#ifndef MINDSPORE_CORE_OPS_PAGED_ATTENTION_H_
#define MINDSPORE_CORE_OPS_PAGED_ATTENTION_H_
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
constexpr auto kNamePagedAttention = "PagedAttention";
enum PagedAttentionInputIndex : size_t {
  kPagedAttentionInputQueryIndex = 0,
  kPagedAttentionInputKeyCacheIndex,
  kPagedAttentionInputValueCacheIndex,
  kPagedAttentionInputBlockTablesIndex,
  kPagedAttentionInputContextLensIndex,
  kPagedAttentionInputsNum
};
enum PagedAttentionOutputIndex : size_t { kPagedAttentionOutputAttentionOutIndex = 0, kPagedAttentionOutputsNum };

/// \brief PagedAttention.
/// Refer to Python API @ref mindspore.ops.PagedAttention for more details.
class MIND_API PagedAttention : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(PagedAttention);
  /// \brief Constructor.
  PagedAttention() : BaseOperator(kNamePagedAttention) {
    InitIOName({"query", "key_cache", "value_cache", "block_tables", "context_lens"}, {"attention_out"});
  }
};

AbstractBasePtr PagedAttentionInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args);
using PagedAttentionPtr = std::shared_ptr<PagedAttention>;
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_PAGED_ATTENTION_H_
