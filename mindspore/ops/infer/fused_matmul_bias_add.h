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

#ifndef MINDSPORE_CORE_OPS_FUSED_MAT_MUL_BIAS_ADD_H_
#define MINDSPORE_CORE_OPS_FUSED_MAT_MUL_BIAS_ADD_H_
#include <memory>
#include <string>
#include "mindapi/base/types.h"
#include "ops/base_operator.h"

namespace mindspore {
namespace ops {
constexpr auto kNameFusedMatMulBiasAdd = "FusedMatMulBiasAdd";
/// \brief Multiplies matrix a and matrix b. Refer to Python API @ref mindspore.ops.FusedMatMulBiasAdd for more details.
class OPS_API FusedMatMulBiasAdd : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(FusedMatMulBiasAdd);
  /// \brief Constructor.
  FusedMatMulBiasAdd() : BaseOperator(kNameFusedMatMulBiasAdd) { InitIOName({"x1", "x2"}, {"output"}); }
  explicit FusedMatMulBiasAdd(const std::string k_name) : BaseOperator(k_name) { InitIOName({"x", "x2"}, {"output"}); }
  /// \brief Init. Refer to the parameters of Python API @ref mindspore.ops.FusedMatMulBiasAdd for the inputs.
  void Init(bool transpose_a = false, bool transpose_b = false);
  /// \brief Set transpose_a.
  void set_transpose_a(bool transpose_a);
  /// \brief Set transpose_b.
  void set_transpose_b(bool transpose_b);
  /// \brief Get transpose_a.

  /// \return transpose_a.
  bool get_transpose_a() const;
  /// \brief Get transpose_b.
  ///
  /// \return transpose_b.
  bool get_transpose_b() const;
};
}  // namespace ops
}  // namespace mindspore
#endif  // MINDSPORE_CORE_OPS_FUSED_MAT_MUL_BIAS_ADD_H_
