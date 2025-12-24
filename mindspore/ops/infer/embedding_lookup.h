/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CORE_OPS_EMBEDDING_LOOKUP_H_
#define MINDSPORE_CORE_OPS_EMBEDDING_LOOKUP_H_
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "mindapi/base/types.h"
#include "ops/base_operator.h"

namespace mindspore {
namespace ops {
constexpr auto kNameEmbeddingLookup = "EmbeddingLookup";
/// \brief Returns a slice of input tensor based on the specified indices.
/// Refer to Python API @ref mindspore.ops.EmbeddingLookup for more details.
class OPS_API EmbeddingLookup : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(EmbeddingLookup);
  /// \brief Constructor.
  EmbeddingLookup() : BaseOperator(kNameEmbeddingLookup) { InitIOName({"params", "indices", "offset"}, {"output"}); }
  /// \brief Init. Refer to the parameters of Python API @ref mindspore.ops.EmbeddingLookup for the inputs.
  explicit EmbeddingLookup(const std::string k_name) : BaseOperator(k_name) {
    InitIOName({"params", "indices", "offset"}, {"output"});
  }
  void Init(const bool setattr_flag = true);
  /// \brief Set setattr_flag.
  void set_setattr_flag(const bool setattr_flag);
  /// \brief Set offset.
  void set_offset(const int64_t offset);
  /// \brief Get setattr_flag.
  ///
  /// \return setattr_flag.
  bool get_setattr_flag() const;
  ///
  /// \return offset.
  int64_t get_offset();
};
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_EMBEDDING_LOOKUP_H_
