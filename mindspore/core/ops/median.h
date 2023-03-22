/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CORE_OPS_MEDIAN_H_
#define MINDSPORE_CORE_OPS_MEDIAN_H_

#include <map>
#include <vector>
#include <string>
#include <memory>

#include "ops/base_operator.h"
#include "mindapi/base/types.h"

namespace mindspore {
namespace ops {
constexpr auto kNameMedian = "Median";
/// \brief Median operation. Refer to Python API @ref mindspore.ops.Median for more details.
class MIND_API Median : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(Median);
  /// \brief Constructor.
  Median() : BaseOperator(kNameMedian) { InitIOName({"x"}, {"y", "indices"}); }
  /// \brief Init. Refer to the parameters of Python API @ref mindspore.ops.Median for the inputs.
  void Init(const bool global_median = false, const int64_t axis = 0, const bool keep_dims = false,
            const bool ignore_nan = false);
  /// \brief Set global_median.
  void set_global_median(const bool global_median);
  /// \brief Set keep_dims.
  void set_keep_dims(const bool keep_dims);
  /// \brief Set ignore_nan.
  void set_ignore_nan(const bool ignore_nan);
  /// \brief Set axis.
  void set_axis(const int64_t &axis);
  /// \brief Get global_median.
  ///
  /// \return global_median.
  bool get_global_median() const;
  /// \brief Get keep_dims.
  ///
  /// \return keep_dims.
  bool get_keep_dims() const;
  /// \brief Get ignore_nan.
  ///
  /// \return ignore_nan.
  bool get_ignore_nan() const;
  /// \brief Get axis.
  ///
  /// \return axis.
  int64_t get_axis() const;
};

MIND_API abstract::AbstractBasePtr MedianInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                               const std::vector<abstract::AbstractBasePtr> &input_args);
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_MEDIAN_H_
