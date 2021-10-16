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

#ifndef MINDSPORE_CORE_OPS_MOD_H_
#define MINDSPORE_CORE_OPS_MOD_H_
#include "ops/primitive_c.h"
#include "abstract/abstract_value.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace ops {
constexpr auto kNameMod = "Mod";
/// \brief Computes the remainder of dividing the first input tensor by the second input tensor element-wise.
/// Refer to Python API @ref mindspore.ops.Mod for more details.
class MS_CORE_API Mod : public PrimitiveC {
 public:
  /// \brief Constructor.
  Mod() : PrimitiveC(kNameMod) { InitIOName({"x", "y"}, {"output"}); }
  /// \brief Destructor.
  ~Mod() = default;
  MS_DECLARE_PARENT(Mod, PrimitiveC);
  /// \brief Init. Refer to the parameters of Python API @ref mindspore.ops.Mod for the inputs.
  void Init() {}
};
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_MOD_H_
