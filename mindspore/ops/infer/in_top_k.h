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

#ifndef MINDSPORE_CORE_OPS_IN_TOP_K_H_
#define MINDSPORE_CORE_OPS_IN_TOP_K_H_
#include <memory>
#include <string>
#include <vector>

#include "mindapi/base/types.h"
#include "ops/base_operator.h"

namespace mindspore {
namespace ops {
constexpr auto kNameInTopK = "InTopK";
/// \brief Determines whether the targets are in the top `k` predictions.
/// Refer to Python API @ref mindspore.ops.InTopK for more details.
class OPS_API InTopK : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(InTopK);
  /// \brief Constructor.
  explicit InTopK(const std::string &k_name = kNameInTopK) : BaseOperator(k_name) {
    InitIOName({"x1", "x2", "k"}, {"y"});
  }
  /// \brief Init. Refer to the parameters of Python API @ref mindspore.ops.InTopK for the inputs.
  void Init(const int64_t k);
  /// \brief Set k.
  void set_k(const int64_t k);
  /// \brief Get k.
  ///
  /// \return k.
  int64_t get_k() const;
};

abstract::AbstractBasePtr InTopKInferFunc(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                          const std::vector<abstract::AbstractBasePtr> &input_args);
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_IN_TOP_K_H_
