/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CORE_OPS_COMPLEX_H_
#define MINDSPORE_CORE_OPS_COMPLEX_H_
#include <memory>
#include <vector>

#include "abstract/abstract_value.h"
#include "ops/primitive_c.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace ops {
/// \brief Returns a complex Tensor from the real and imaginary part.
/// Refer to Python API @ref mindspore.ops.Complex for more details.
class MS_CORE_API Complex : public PrimitiveC {
 public:
  /// \brief Constructor.
  Complex() : PrimitiveC(prim::kPrimSquare->name()) { InitIOName({"s", "input_imag"}, {"output"}); }
  /// \brief Destructor.
  ~Complex() = default;
  MS_DECLARE_PARENT(Complex, PrimitiveC);
  /// \brief Init. Refer to the parameters of Python API @ref mindspore.ops.Complex for the inputs.
  void Init() {}
};
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_COMPLEX_H_
