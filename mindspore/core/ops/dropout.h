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

#ifndef MINDSPORE_CORE_OPS_DROPOUT_H_
#define MINDSPORE_CORE_OPS_DROPOUT_H_
#include <vector>
#include <memory>

#include "ops/primitive_c.h"
#include "abstract/abstract_value.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace ops {
constexpr auto kNameDropout = "Dropout";
/// \brief During training, randomly zeroes some of the elements of the input tensor with probability 1-keep_prob
//// from a Bernoulli distribution. Refer to Python API @ref mindspore.ops.Dropout for more details.
class MS_CORE_API Dropout : public PrimitiveC {
 public:
  /// \brief Constructor.
  Dropout() : PrimitiveC(kNameDropout) {}
  /// \brief Destructor.
  ~Dropout() = default;
  MS_DECLARE_PARENT(Dropout, PrimitiveC);
  /// \brief Init. Refer to the parameters of Python API @ref mindspore.ops.Dropout for the inputs.
  void Init(const float keep_prob = 0.5);
  /// \brief Set keep_prob.
  void set_keep_prob(const float keep_prob);
  /// \brief Get keep_prob.
  ///
  /// \return keep_prob.
  float get_keep_prob() const;
};
AbstractBasePtr DropoutInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                             const std::vector<AbstractBasePtr> &input_args);
using PrimDropoutPtr = std::shared_ptr<Dropout>;
}  // namespace ops
}  // namespace mindspore
#endif  // MINDSPORE_CORE_OPS_DROPOUT_H_
