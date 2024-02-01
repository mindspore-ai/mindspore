/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CORE_OPS_SILENT_CHECK_H_
#define MINDSPORE_CORE_OPS_SILENT_CHECK_H_
#include <map>
#include <memory>
#include <string>
#include "mindapi/base/types.h"
#include "ops/base_operator.h"

namespace mindspore {
namespace ops {
constexpr auto kNameSilentCheck = "SilentCheck";
/// \brief SilentCheck defined the SilentCheck operator prototype.
class MIND_API SilentCheck : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(SilentCheck);
  /// \brief Constructor.
  SilentCheck() : BaseOperator(kNameSilentCheck) {}

  void Init(const int64_t c_min_steps, const float c_thresh, const float c_coeff);

  /// \brief Method to set c_min_steps attribute.
  ///
  /// \param[in] c_min_steps Define a value xxx
  void set_c_min_steps(const int64_t c_min_steps);

  /// \brief Method to get c_min_steps attribute.
  ///
  /// \return a value.
  int64_t get_c_min_steps() const;

  /// \brief Method to set c_thresh attribute.
  ///
  /// \param[in] c_thresh Define a value xxx
  void set_c_thresh(const float c_thresh);

  /// \brief Method to get c_thresh attribute.
  ///
  /// \return a value.
  float get_c_thresh() const;

  /// \brief Method to set c_coeff attribute.
  ///
  /// \param[in] c_coeff Define a value xxx
  void set_c_coeff(const float c_coeff);

  /// \brief Method to get c_coeff attribute.
  ///
  /// \return a value.
  float get_c_coeff() const;
};
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_SILENT_CHECK_H_
