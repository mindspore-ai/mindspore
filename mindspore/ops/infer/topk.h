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

#ifndef MINDSPORE_CORE_OPS_TOPK_H_
#define MINDSPORE_CORE_OPS_TOPK_H_
#include <memory>
#include <string>
#include <vector>

#include "mindapi/base/types.h"
#include "ops/base_operator.h"

namespace mindspore {
namespace ops {
constexpr auto kNameTopK = "TopK";
/// \brief Finds values and indices of the k largest entries along the last dimension.
/// Refer to Python API @ref mindspore.ops.TopK for more details.
class OPS_API TopK : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(TopK);
  /// \brief Constructor.
  explicit TopK(const std::string &k_name = kNameTopK) : BaseOperator(k_name) {
    InitIOName({"input", "k"}, {"values", "indices"});
  }
  /// \brief Init. Refer to the parameters of Python API @ref mindspore.ops.TopK for the inputs.
  void Init(const bool sorted = false);
  /// \brief Set sorted.
  void set_sorted(const bool sorted);
  /// \brief Get sorted.
  ///
  /// \return sorted.
  bool get_sorted() const;
  bool get_attr(const char *attr) const;
};
OPS_API abstract::AbstractBasePtr TopKInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                            const std::vector<abstract::AbstractBasePtr> &input_args);
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_TOPK_H_
