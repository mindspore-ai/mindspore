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

#ifndef MINDSPORE_CORE_OPS_UNSQUEEZE_H_
#define MINDSPORE_CORE_OPS_UNSQUEEZE_H_

#include <memory>
#include <vector>

#include "mindapi/base/types.h"
#include "ops/base_operator.h"

namespace mindspore {
namespace ops {
constexpr auto kNameUnsqueeze = "Unsqueeze";
/// \brief Unsqueeze defined the Unsqueeze operator prototype of lite.
class OPS_API Unsqueeze : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(Unsqueeze);
  /// \brief Constructor.
  Unsqueeze() : BaseOperator(kNameUnsqueeze) {}

  /// \brief Method to init the op's attributes
  ///
  /// \param[in] axis Define a vector to indicate on which dimensions to expand.
  void Init(const std::vector<int64_t> axis);

  /// \brief Method to set axis attribute.
  ///
  /// \param[in] axis Define a vector to indicate on which dimensions to expand.
  void set_axis(const std::vector<int64_t> axis);

  /// \brief Method to get axis attribute.
  ///
  /// \return dimensions info of expanding.
  std::vector<int64_t> get_axis() const;
};
OPS_API abstract::AbstractBasePtr UnsqueezeInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                                 const std::vector<abstract::AbstractBasePtr> &input_args);
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_UNSQUEEZE_H_
