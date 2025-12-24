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

#ifndef MINDSPORE_CORE_OPS_SPLICE_H_
#define MINDSPORE_CORE_OPS_SPLICE_H_
#include <memory>
#include <vector>

#include "mindapi/base/types.h"
#include "ops/base_operator.h"

namespace mindspore {
namespace ops {
constexpr auto kNameSplice = "Splice";
/// \brief All defined All operator prototype of lite.
class OPS_API Splice : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(Splice);
  /// \brief Constructor.
  Splice() : BaseOperator(kNameSplice) { InitIOName({"inputs"}, {"outputs"}); }

  /// \brief Method to init the op's attributes.
  ///
  /// \param[in] contexts Define the contexts.
  /// \param[in] forward_indexes Define the forward indexes.
  /// \param[in] output_dims Define the output dims.
  void Init(const std::vector<int64_t> &contexts, const std::vector<int64_t> &forward_indexes, int64_t output_dims);

  /// \brief Method to set contexts attributes.
  ///
  /// \param[in] contexts Define the contexts.
  void set_context(const std::vector<int64_t> &contexts);

  /// \brief Method to set forward_indexes attributes.
  ///
  /// \param[in] forward_indexes Define the forward_indexes.
  void set_forward_indexes(const std::vector<int64_t> &forward_indexes);

  /// \brief Method to set output_dim attributes.
  ///
  /// \param[in] output_dim Define the output_dim.
  void set_output_dim(int64_t output_dim);

  /// \brief Method to set context attributes.
  ///
  /// \param[in] context Define the context.
  std::vector<int64_t> get_context() const;

  /// \brief Method to set forward_indexes attributes.
  ///
  /// \param[in] forward_indexes Define the forward_indexes.
  std::vector<int64_t> get_forward_indexes() const;

  /// \brief Method to set output_dim attributes.
  ///
  /// \param[in] output_dim Define the output_dim.
  int64_t get_output_dim() const;
};
OPS_API abstract::AbstractBasePtr SpliceInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                              const std::vector<abstract::AbstractBasePtr> &input_args);
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_SPLICE_H_
