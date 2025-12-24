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

#ifndef MINDSPORE_CORE_OPS_NON_MAX_SUPPRESSION_H_
#define MINDSPORE_CORE_OPS_NON_MAX_SUPPRESSION_H_

#include <algorithm>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "mindapi/base/types.h"
#include "ops/base_operator.h"

namespace mindspore {
namespace ops {
constexpr auto kNameNonMaxSuppression = "NonMaxSuppression";
/// \brief NonMaxSuppression QuantDTypeCast the NonMaxSuppression operator prototype.
class OPS_API NonMaxSuppression : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(NonMaxSuppression);
  /// \brief Constructor.
  NonMaxSuppression() : BaseOperator(kNameNonMaxSuppression) {}

  /// \brief Method to init the op's attributes.
  ///
  /// \param[in] center_point_box Define a value to indicate the format of the box data. If the value is 0, the box data
  ///            is supplied by diagonal point, such as [y1, x1, y2, x2](the pair [y1, x1] is picture coordinate). If
  ///            the value is 1, the box data is supplied as [x_center, y_center, width, height].
  void Init(const int64_t center_point_box = 0);

  /// \brief Method to set center_point_box attribute.
  ///
  /// \param[in] center_point_box a value to indicate the format of the box data.
  void set_center_point_box(const int64_t center_point_box);

  /// \brief Method to get center_point_box attribute.
  ///
  /// \return an integer value.
  int64_t get_center_point_box() const;
};
OPS_API abstract::AbstractBasePtr NonMaxSuppressionInfer(const abstract::AnalysisEnginePtr &,
                                                         const PrimitivePtr &primitive,
                                                         const std::vector<abstract::AbstractBasePtr> &input_args);
using PrimNonMaxSuppressionPtr = std::shared_ptr<NonMaxSuppression>;
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_NON_MAX_SUPPRESSION_H_
