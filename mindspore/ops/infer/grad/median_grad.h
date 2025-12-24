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

#ifndef MINDSPORE_CORE_OPS_MEDIAN_GRAD_H_
#define MINDSPORE_CORE_OPS_MEDIAN_GRAD_H_

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "mindapi/base/types.h"
#include "ops/base_operator.h"

namespace mindspore {
namespace ops {
constexpr auto kNameMedianGrad = "MedianGrad";
class OPS_API MedianGrad : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(MedianGrad);
  /// \brief Constructor.
  MedianGrad() : BaseOperator(kNameMedianGrad) { InitIOName({"y_grad", "x", "y", "indices"}, {"x_grad"}); }
  /// \brief Init. Refer to the parameters of Python API @ref mindspore.ops.Median for the inputs.
  void Init(const bool global_median = false, const int64_t axis = 0, const bool keep_dims = false);
  /// \brief Set global_median.
  void set_global_median(const bool global_median);
  /// \brief Set keep_dims.
  void set_keep_dims(const bool keep_dims);
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
  /// \brief Get axis.
  ///
  /// \return axis.
  int64_t get_axis() const;
};

OPS_API abstract::AbstractBasePtr MedianGradInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                                  const std::vector<abstract::AbstractBasePtr> &input_args);
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_MEDIAN_GRAD_H_
