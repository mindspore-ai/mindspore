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
#ifndef MINDSPORE_CORE_OPS_FAKE_QUANT_WITH_MIN_MAX_VARS_H_
#define MINDSPORE_CORE_OPS_FAKE_QUANT_WITH_MIN_MAX_VARS_H_
#include <memory>
#include <vector>

#include "mindapi/base/types.h"
#include "ops/base_operator.h"

namespace mindspore {
namespace ops {
constexpr auto kNameFakeQuantWithMinMaxVars = "FakeQuantWithMinMaxVars";
/// \brief Fake-quantize the input by minimum and maximum.
/// Refer to Python API @ref mindspore.ops.FakeQuantWithMinMaxVars for more details.
class OPS_API FakeQuantWithMinMaxVars : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(FakeQuantWithMinMaxVars);
  /// \brief Constructor.
  FakeQuantWithMinMaxVars() : BaseOperator(kNameFakeQuantWithMinMaxVars) {}
  /// \brief Init. Refer to the parameters of Python API @ref mindspore.ops.FakeQuantWithMinMaxVars for the inputs.
  void Init(const bool narrow_range = false, const int64_t num_bits = 8);
  /// \brief Set narrow_range.
  void set_narrow_range(const bool narrow_range);
  /// \brief Set num_bits.
  void set_num_bits(const int64_t num_bits);
  /// \brief Get narrow_range.
  ///
  /// \return narrow_range.
  bool get_narrow_range() const;
  /// \brief Get num_bits.
  ///
  /// \return num_bits.
  int64_t get_num_bits() const;
};
OPS_API abstract::AbstractBasePtr FakeQuantWithMinMaxVarsInfer(
  const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
  const std::vector<abstract::AbstractBasePtr> &input_args);
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_FAKE_QUANT_WITH_MIN_MAX_VARS_H_
