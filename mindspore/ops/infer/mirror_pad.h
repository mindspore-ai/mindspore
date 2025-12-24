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

#ifndef MINDSPORE_CORE_OPS_MIRROR_PAD_H_
#define MINDSPORE_CORE_OPS_MIRROR_PAD_H_
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "mindapi/base/types.h"
#include "ops/base_operator.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace ops {
constexpr auto kNameMirrorPad = "MirrorPad";
/// \brief Pads the input tensor according to the paddings. Refer to Python API
/// @ref mindspore.ops.MirrorPad for more details.
class OPS_API MirrorPad : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(MirrorPad);
  /// \brief Constructor.
  MirrorPad() : BaseOperator(kNameMirrorPad) { InitIOName({"x", "paddings"}, {"y"}); }
  explicit MirrorPad(const std::string k_name) : BaseOperator(k_name) {}
  /// \brief Init. Refer to the parameters of Python API @ref mindspore.ops.MirrorPad for the inputs.
  void Init(const std::string &mode) { set_mode(mode); }
  /// \brief Set mode.
  void set_mode(const std::string &mode);
  /// \brief get mode.
  std::string get_mode() const;
};
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_MirrorPad_H_
