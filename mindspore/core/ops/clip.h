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
#ifndef MINDSPORE_CORE_OPS_CLIP_H_
#define MINDSPORE_CORE_OPS_CLIP_H_
#include <memory>

#include "ops/primitive_c.h"
#include "abstract/abstract_value.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace ops {
constexpr auto kNameClip = "Clip";
/// \brief Clip defined Clip operator prototype.
class MS_CORE_API Clip : public PrimitiveC {
 public:
  /// \brief Constructor.
  Clip() : PrimitiveC(kNameClip) {}

  /// \brief Destructor.
  ~Clip() = default;

  MS_DECLARE_PARENT(Clip, PrimitiveC);

  /// \brief Method to init the op's attributes.
  ///
  /// \param[in] max Define the upper bound. If value is larger than the upper bound, it will be set as this upper
  ///            bound.
  /// \param[in] min Define the lower bound. If value is less than the lower bound, it will be set as this lower bound.
  void Init(const float max, const float min);

  /// \brief Method to set max attribute.
  ///
  /// \param[in] max Define the upper bound. If value is larger than the upper bound, it will be set as this upper
  ///            bound.
  void set_max(const float max);

  /// \brief Method to set min attribute.
  ///
  /// \param[in] min Define the lower bound. If value is less than the lower bound, it will be set as this lower bound.
  void set_min(const float min);

  /// \brief Method to get max attribute.
  ///
  /// \return a value to indicate upper bound.
  float get_max() const;

  /// \brief Method to get min attribute.
  ///
  /// \return a value to indicate lower bound.
  float get_min() const;
};

using PrimClipPtr = std::shared_ptr<Clip>;
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_CLIP_H_
