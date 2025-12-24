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

#ifndef MINDSPORE_CORE_OPS_REVERSE_SEQUENCE_H_
#define MINDSPORE_CORE_OPS_REVERSE_SEQUENCE_H_
#include <memory>
#include <vector>

#include "mindapi/base/types.h"
#include "ops/base_operator.h"

namespace mindspore {
namespace ops {
constexpr auto kNameReverseSequence = "ReverseSequence";
/// \brief Reverses variable length slices.
/// Refer to Python API @ref mindspore.ops.ReverseSequence for more details.
class OPS_API ReverseSequence : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(ReverseSequence);
  /// \brief Constructor.
  ReverseSequence() : BaseOperator(kNameReverseSequence) { InitIOName({"x", "seq_lengths"}, {"y"}); }
  /// \brief Init. Refer to the parameters of Python API @ref mindspore.ops.ReverseSequence for the inputs.
  void Init(const int64_t seq_dim, const int64_t batch_dim = 0);
  /// \brief Set seq_dim.
  void set_seq_dim(const int64_t seq_dim);
  /// \brief Set batch_dim.
  void set_batch_dim(const int64_t batch_dim);
  /// \brief Get seq_dim.
  ///
  /// \return seq_dim.
  int64_t get_seq_dim() const;
  /// \brief Get batch_dim.
  ///
  /// \return batch_dim.
  int64_t get_batch_dim() const;
};
OPS_API abstract::AbstractBasePtr ReverseSequenceInfer(const abstract::AnalysisEnginePtr &,
                                                       const PrimitivePtr &primitive,
                                                       const std::vector<abstract::AbstractBasePtr> &input_args);
using PrimReverseSequence = std::shared_ptr<ReverseSequence>;
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_REVERSE_SEQUENCE_H_
