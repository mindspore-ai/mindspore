/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CORE_OPS_CONVERT_TO_DYNAMIC_H_
#define MINDSPORE_CORE_OPS_CONVERT_TO_DYNAMIC_H_

#include <memory>
#include <string>
#include <vector>

#include "mindapi/base/types.h"
#include "ops/base_operator.h"

namespace mindspore {
namespace ops {
constexpr auto kNameConvertToDynamicRank = "ConvertToDynamic";
/// \brief Convert to dynamic rank.
class OPS_API ConvertToDynamic : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(ConvertToDynamic);
  /// \brief Constructor.
  ConvertToDynamic() : BaseOperator(kNameConvertToDynamicRank) { InitIOName({"input"}, {"output"}); }

  /// \brief Init.
  void Init() const {}

  /// \brief Method to set is_dynamic_rank attribute.
  ///
  /// \param[in] is_dynamic_rank Define whether convert to dynamic rank, default false.
  void set_is_dynamic_rank(const bool is_dynamic_rank);

  /// \brief Method to get is_dynamic_rank attribute.
  ///
  /// \return is_dynamic_rank attribute.
  bool get_is_dynamic_rank() const;
};

using PrimConvertToDynamicRankPtr = std::shared_ptr<ConvertToDynamic>;
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_CONVERT_TO_DYNAMIC_H_
