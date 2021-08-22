/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CORE_OPS_GLU_H_
#define MINDSPORE_CORE_OPS_GLU_H_
#include <vector>
#include <memory>
#include "ops/primitive_c.h"
#include "ops/op_utils.h"
#include "abstract/abstract_value.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace ops {
constexpr auto kNameGLU = prim::kGLU;
class MS_CORE_API GLU : public PrimitiveC {
 public:
  GLU() : PrimitiveC(kNameGLU) { InitIOName({"x"}, {"output"}); }
  ~GLU() = default;
  MS_DECLARE_PARENT(GLU, PrimitiveC);
  void Init(int64_t axis);
  void set_axis(int64_t axis);
  int64_t get_axis() const;
};

using PrimGLUPtr = std::shared_ptr<GLU>;
}  // namespace ops
}  // namespace mindspore
#endif  // MINDSPORE_CORE_OPS_GLU_H_
