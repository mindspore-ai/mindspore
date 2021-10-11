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
#include <vector>
#include <memory>
#include "ops/primitive_c.h"
#include "abstract/abstract_value.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace ops {
constexpr auto kNameAssert = "Assert";
/// \brief Assert defined Assert operator prototype of lite.
class MS_CORE_API Assert : public PrimitiveC {
 public:
  /// \brief Constructor.
  Assert() : PrimitiveC(kNameAssert) {}

  /// \brief Destructor.
  ~Assert() = default;

  MS_DECLARE_PARENT(Assert, PrimitiveC);

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

AbstractBasePtr AssertInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                            const std::vector<AbstractBasePtr> &input_args);
using PrimAssertPtr = std::shared_ptr<Assert>;
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_ASSERT_H_
