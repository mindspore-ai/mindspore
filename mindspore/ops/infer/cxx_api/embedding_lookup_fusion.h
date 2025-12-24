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

#ifndef MINDSPORE_CORE_OPS_EMBEDDING_LOOKUP_FUSION_H_
#define MINDSPORE_CORE_OPS_EMBEDDING_LOOKUP_FUSION_H_
#include <map>
#include <memory>
#include <string>
#include <vector>
#include "mindapi/base/types.h"
#include "ops/base_operator.h"

namespace mindspore {
namespace ops {
constexpr auto kNameEmbeddingLookupFusion = "EmbeddingLookupFusion";
/// \brief EmbeddingLookupFusion defined EmbeddingLookup operator prototype of lite.
class OPS_API EmbeddingLookupFusion : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(EmbeddingLookupFusion);
  /// \brief Constructor.
  EmbeddingLookupFusion() : BaseOperator(kNameEmbeddingLookupFusion) {
    InitIOName({"params", "indices", "offset"}, {"output"});
  }

  /// \brief Method to init the op's attributes.
  ///
  /// \param[in] max_norm Define the max l2-norm value of each embedding. Each embedding will be clip if l2-norm is
  ///            larger than this value.
  void Init(const float max_norm = 0.0);

  /// \brief Method to set max_norm attribute.
  ///
  /// \param[in] max_norm Define the max l2-norm value of each embedding. Each embedding will be clip if l2-norm is
  ///            larger than this value.
  void set_max_norm(const float max_norm);

  /// \brief Method to get max_norm attribute.
  ///
  /// \return a value which indicates a max l2-norm value.
  float get_max_norm() const;
};
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_EMBEDDING_LOOKUP_FUSION_H_
