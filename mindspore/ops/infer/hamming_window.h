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

#ifndef MINDSPORE_CORE_OPS_HAMMING_WINDOW_H_
#define MINDSPORE_CORE_OPS_HAMMING_WINDOW_H_

#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "mindapi/base/types.h"
#include "ops/base_operator.h"

namespace mindspore {
namespace ops {
constexpr auto kNameHammingWindow = "HammingWindow";
/// \brief Computes batched the hamming window function with input window length.
/// Refer to Python API @ref mindspore.ops.HammingWindow for more details.
class OPS_API HammingWindow : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(HammingWindow);
  /// \brief Constructor.
  HammingWindow() : BaseOperator(kNameHammingWindow) { InitIOName({"length"}, {"y"}); }
  void Init(const bool periodic = true, const float alpha = 0.54, const float beta = 0.46);
  void set_periodic(const bool periodic);
  bool get_periodic() const;
  void set_alpha(const float alpha);
  float get_alpha() const;
  void set_beta(const float beta);
  float get_beta() const;
};

OPS_API abstract::AbstractBasePtr HammingWindowInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                                     const std::vector<abstract::AbstractBasePtr> &input_args);
}  // namespace ops
}  // namespace mindspore
#endif  // MINDSPORE_CORE_OPS_HAMMING_WINDOW_H_
