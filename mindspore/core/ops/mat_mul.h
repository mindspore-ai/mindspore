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

#ifndef MINDSPORE_CORE_OPS_MAT_MUL_H_
#define MINDSPORE_CORE_OPS_MAT_MUL_H_
#include <vector>
#include <memory>

#include "ops/primitive_c.h"
#include "ops/op_utils.h"
#include "abstract/abstract_value.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace ops {
constexpr auto kNameMatMul = "MatMul";
class MatMul : public PrimitiveC {
 public:
  MatMul() : PrimitiveC(kNameMatMul) { InitIOName({"x1", "x2"}, {"output"}); }
  ~MatMul() = default;
  MS_DECLARE_PARENT(MatMul, PrimitiveC);
  void Init(bool transpose_a = false, bool transpose_b = false);
  void set_transpose_a(bool transpose_a);
  void set_transpose_b(bool transpose_b);
  bool get_transpose_a() const;
  bool get_transpose_b() const;
};
using PrimMatMulPtr = std::shared_ptr<MatMul>;
}  // namespace ops
}  // namespace mindspore
#endif  // MINDSPORE_CORE_OPS_MAT_MUL_H_
