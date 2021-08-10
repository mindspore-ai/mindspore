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

#ifndef MINDSPORE_CORE_OPS_AFFINE_H_
#define MINDSPORE_CORE_OPS_AFFINE_H_
#include <vector>
#include <string>
#include "ops/primitive_c.h"
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace ops {

constexpr auto kNameAffine = "Affine";
constexpr auto kAffineContext = "context";
constexpr auto kAffineOutputDim = "output_dim";

class MS_CORE_API Affine : public PrimitiveC {
 public:
  Affine() : PrimitiveC(kNameAffine) { InitIOName({"x1", "x2"}, {"outputs"}); }
  ~Affine() = default;
  MS_DECLARE_PARENT(Affine, PrimitiveC);
  void Init(const std::vector<int64_t> &contexts, int64_t output_dim, bool transpose_a = false,
            bool transpose_b = false);
  void set_context(const std::vector<int64_t> &);
  void set_output_dim(int64_t output_dim);
  void set_transpose_a(bool transpose_a);
  void set_transpose_b(bool transpose_b);
  void set_activation_type(const ActivationType &activation_type);

  bool get_transpose_a() const;
  bool get_transpose_b() const;
  std::vector<int64_t> get_context() const;
  int64_t get_output_dim() const;
  ActivationType get_activation_type() const;
};
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_AFFINE_H_
