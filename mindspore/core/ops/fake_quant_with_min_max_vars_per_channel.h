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

#ifndef MINDSPORE_CORE_OPS_FAKE_QUANT_WITH_MIN_MAX_VARS_PER_CHANNEL_H_
#define MINDSPORE_CORE_OPS_FAKE_QUANT_WITH_MIN_MAX_VARS_PER_CHANNEL_H_
#include <map>
#include <vector>
#include <string>
#include <memory>
#include "ops/primitive_c.h"
#include "abstract/abstract_value.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace ops {
constexpr auto kNameFakeQuantWithMinMaxVarsPerChannel = "FakeQuantWithMinMaxVarsPerChannel";
/// \brief Fake-quantize the input and one of shape: [d], [b, d], [b, h, w, d] by per-channel minimum and maximum.
/// Refer to Python API @ref mindspore.ops.FakeQuantWithMinMaxVarsPerChannel for more details.
class MS_CORE_API FakeQuantWithMinMaxVarsPerChannel : public PrimitiveC {
 public:
  /// \brief Constructor.
  FakeQuantWithMinMaxVarsPerChannel() : PrimitiveC(kNameFakeQuantWithMinMaxVarsPerChannel) {}
  /// \brief Destructor.
  ~FakeQuantWithMinMaxVarsPerChannel() = default;
  MS_DECLARE_PARENT(FakeQuantWithMinMaxVarsPerChannel, PrimitiveC);
  /// \brief Init. Refer to the parameters of Python API @ref mindspore.ops.FakeQuantWithMinMaxVarsPerChannel
  /// for the inputs.
  void Init(const int64_t num_bits = 8, const bool narrow_range = false);
  /// \brief Set num_bits.
  void set_num_bits(const int64_t num_bits);
  /// \brief Set narrow_range.
  void set_narrow_range(const bool narrow_range);
  /// \brief Get num_bits.
  ///
  /// \return num_bits.
  int64_t get_num_bits() const;
  /// \brief Get narrow_range.
  ///
  /// \return narrow_range.
  bool get_narrow_range() const;
};

AbstractBasePtr FakeQuantWithMinMaxVarsPerChannelInfer(const abstract::AnalysisEnginePtr &,
                                                       const PrimitivePtr &primitive,
                                                       const std::vector<AbstractBasePtr> &input_args);
using PrimFakeQuantWithMinMaxVarsPerChannelPtr = std::shared_ptr<FakeQuantWithMinMaxVarsPerChannel>;
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_FAKE_QUANT_WITH_MIN_MAX_VARS_PER_CHANNEL_H_
