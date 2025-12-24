/**
 * Copyright 2022-2025 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CORE_OPS_DILATION2D_BACKPROP_INPUT_H_
#define MINDSPORE_CORE_OPS_DILATION2D_BACKPROP_INPUT_H_

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "mindapi/base/types.h"
#include "ops/base_operator.h"

namespace mindspore {
namespace ops {
constexpr auto kNameDilation2DBackpropInput = "Dilation2DBackpropInput";
class OPS_API Dilation2DBackpropInput : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(Dilation2DBackpropInput);
  Dilation2DBackpropInput() : BaseOperator(kNameDilation2DBackpropInput) {
    InitIOName({"x", "filter", "out_backprop"}, {"y"});
  }
  explicit Dilation2DBackpropInput(const std::string k_name) : BaseOperator(k_name) {
    InitIOName({"x", "filter", "out_backprop"}, {"y"});
  }
  /// \brief Get stride.
  ///
  /// \return stride.
  std::vector<int64_t> get_stride() const;
  /// \brief Get dilation.
  ///
  /// \return dilation.
  std::vector<int64_t> get_dilation() const;
  /// \brief Get pad_mode.
  ///
  /// \return pad_mode.
  std::string get_pad_mode() const;
  /// \brief Get format.
  ///
  /// \return format.
  std::string get_format() const;
};

OPS_API abstract::AbstractBasePtr Dilation2DBackpropInputInfer(
  const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
  const std::vector<abstract::AbstractBasePtr> &input_args);
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_DILATION2D_BACKPROP_INPUT_H_
