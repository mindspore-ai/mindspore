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

#ifndef MINDSPORE_CORE_OPS_ASSERT_H_
#define MINDSPORE_CORE_OPS_ASSERT_H_
#include <memory>
#include <vector>
#include "mindapi/base/types.h"
#include "ops/base_operator.h"

namespace mindspore {
namespace ops {
constexpr auto kNameAssert = "Assert";
/// \brief Assert defined Assert operator prototype of lite.
class OPS_API Assert : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(Assert);
  /// \brief Constructor.
  Assert() : BaseOperator(kNameAssert) {}

  /// \brief Method to init the op's attributes.
  ///
  /// \param[in] summarize Define print the number of each tensor.
  void Init(const int64_t summarize = 3);

  /// \brief Method to set summarize attributes.
  ///
  /// \param[in] summarize Define print the number of each tensor.
  void set_summarize(const int64_t summarize);

  /// \brief Method to get summarize attributes.
  ///
  /// \return summarize attributes.
  int64_t get_summarize() const;
};

OPS_API abstract::AbstractBasePtr AssertInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                              const std::vector<abstract::AbstractBasePtr> &input_args);
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_ASSERT_H_
