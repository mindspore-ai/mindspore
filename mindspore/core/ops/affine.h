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
#include "ops/base_operator.h"
#include "mindapi/base/types.h"

namespace mindspore {
namespace ops {
constexpr auto kNameAffine = "Affine";
constexpr auto kAffineContext = "context";
constexpr auto kAffineOutputDim = "output_dim";

/// \brief Assert defined Affine operator prototype of lite.
class MIND_API Affine : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(Affine);
  /// \brief Constructor.
  Affine() : BaseOperator(kNameAffine) { InitIOName({"x1", "x2"}, {"outputs"}); }
  /// \brief Method to init the op's attributes.
  void Init(const std::vector<int64_t> &contexts, int64_t output_dim, bool transpose_a = false,
            bool transpose_b = false);
  /// \brief Method to set context attributes.
  ///
  /// \param[in] keep_dims Define the context.
  void set_context(const std::vector<int64_t> &);
  /// \brief Method to set output_dim attributes.
  ///
  /// \param[in] output_dim Define the output dim.
  void set_output_dim(int64_t output_dim);
  /// \brief Method to set transpose_a attributes.
  ///
  /// \param[in] transpose_a Define the if transpose a tensor.
  void set_transpose_a(bool transpose_a);
  /// \brief Method to set transpose_b attributes.
  ///
  /// \param[in] transpose_b Define the if transpose b tensor.
  void set_transpose_b(bool transpose_b);
  /// \brief Method to set activation_type attributes.
  ///
  /// \param[in] activation_type Define the activation type.
  void set_activation_type(const ActivationType &activation_type);

  /// \brief Method to get transpose_a attributes.
  bool get_transpose_a() const;
  /// \brief Method to get transpose_b attributes.
  bool get_transpose_b() const;
  /// \brief Method to get context attributes.
  std::vector<int64_t> get_context() const;
  /// \brief Method to get output_dim attributes.
  int64_t get_output_dim() const;
  /// \brief Method to get activation_type attributes.
  ActivationType get_activation_type() const;
};
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_AFFINE_H_
