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

#ifndef MINDSPORE_CORE_OPS_HISTOGRAM_H_
#define MINDSPORE_CORE_OPS_HISTOGRAM_H_
#include <memory>
#include <vector>

#include "abstract/abstract_value.h"
#include "mindapi/base/types.h"
#include "ops/base_operator.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace ops {
constexpr auto kNameHistogram = "Histogram";
/// \brief Computes the histogram of a tensor.
/// Refer to Python API @ref mindspore.ops.Histogram for more details.
class OPS_API Histogram : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(Histogram);

  /// \brief Constructor.
  Histogram() : BaseOperator(kNameHistogram) { InitIOName({"x"}, {"y"}); }

  /// \brief Init.
  void Init() const {}

  void set_bins(const int64_t bins);
  int64_t get_bins() const;
  void set_min(const float min);
  float get_min() const;
  void set_max(const float max);
  float get_max() const;
};
AbstractBasePtr HistogramInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                               const std::vector<AbstractBasePtr> &input_args);

using PrimHistogramPtr = std::shared_ptr<Histogram>;
}  // namespace ops
}  // namespace mindspore
#endif  // MINDSPORE_CORE_OPS_HISTOGRAM_H_
