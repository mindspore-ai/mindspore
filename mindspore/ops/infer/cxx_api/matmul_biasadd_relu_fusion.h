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

#ifndef MINDSPORE_CORE_OPS_FUSION_MATMUL_BIASADD_RELU_FUSION_H_
#define MINDSPORE_CORE_OPS_FUSION_MATMUL_BIASADD_RELU_FUSION_H_
#include "mindapi/base/types.h"
#include "ops/base_operator.h"

namespace mindspore {
namespace ops {
constexpr auto kNameMatMulBiasAddReluFusion = "MatMulBiasAddReluFusion";
/// \brief MatMulBiasAddReluFusion.
class OPS_API MatMulBiasAddReluFusion : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(MatMulBiasAddReluFusion);
  /// \brief Constructor.
  MatMulBiasAddReluFusion() : BaseOperator(kNameMatMulBiasAddReluFusion) {}
  /// \brief Init. Refer to the parameters of Python API @ref mindspore.ops.MatMulBiasAddReluFusion for the inputs.
  void Init(bool transpose_a = false, bool transpose_b = false);
  /// \brief Set transpose_a.
  void set_transpose_a(bool transpose_a);
  /// \brief Set transpose_b.
  void set_transpose_b(bool transpose_b);
  /// \brief Get transpose_a.
  ///
  /// \return transpose_a.
  bool get_transpose_a() const;
  /// \brief Get transpose_b.
  ///
  /// \return transpose_b.
  bool get_transpose_b() const;
};
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_FUSION_MATMUL_BIASADD_RELU_FUSION_H_
