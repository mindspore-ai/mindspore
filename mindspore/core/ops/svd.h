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

#ifndef MINDSPORE_CORE_OPS_SVD_H_
#define MINDSPORE_CORE_OPS_SVD_H_
#include <string>
#include <memory>

#include "ops/base_operator.h"

namespace mindspore {
namespace ops {
constexpr auto kNameSvd = "Svd";
constexpr auto kAttrComputeUV = "compute_uv";
constexpr auto kAttrFullMatrices = "full_matrices";
/// \brief Returns the singular value decompositions of one or more matrices.
/// Refer to Python API @ref mindspore.ops.svd for more details.
class MIND_API Svd : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(Svd);
  /// \brief Constructor.
  Svd() : BaseOperator(kNameSvd) { InitIOName({"a"}, {"s", "u", "v"}); }
  explicit Svd(const std::string k_name) : BaseOperator(k_name) { InitIOName({"a"}, {"s", "u", "v"}); }
  /// \brief Init. Refer to the parameters of Python API @ref mindspore.ops.svd for the inputs.
  void Init(const bool full_matrices = false, const bool compute_uv = true);
  /// \brief Set axis.
  void set_full_matrices(const bool full_matrices);
  /// \brief Set output_type.
  void set_compute_uv(const bool compute_uv);
  /// \brief Get full_matrices.
  ///
  /// \return full_matrices.
  bool full_matrices() const;
  /// \brief Get compute_uv.
  ///
  /// \return compute_uv.
  bool compute_uv() const;
};
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_SVD_H_
