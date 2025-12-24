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

#ifndef MINDSPORE_CORE_OPS_INPLACE_SUB_H_
#define MINDSPORE_CORE_OPS_INPLACE_SUB_H_
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>
#include "mindapi/base/types.h"
#include "ops/base_operator.h"

namespace mindspore {
namespace ops {
constexpr auto kNameInplaceSub = "InplaceSub";
/// \brief InplaceSub operation. Refer to Python API @ref mindspore.ops.InplaceSub for more details.
class OPS_API InplaceSub : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(InplaceSub);
  /// \brief Constructor.
  InplaceSub() : BaseOperator(kNameInplaceSub) { InitIOName({"x", "v"}, {"y"}); }
  /// \brief Init. Refer to the parameters of Python API @ref mindspore.ops.InplaceSub for the inputs.
  void Init(const std::vector<int64_t> indices) { set_indices(indices); }
  /// \brief Set indices.
  void set_indices(std::vector<int64_t> indices);
  /// \brief Get indices.
  ///
  /// \return indices.
  std::vector<int64_t> get_indices() const;
};

OPS_API abstract::AbstractBasePtr InplaceSubInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                                  const std::vector<abstract::AbstractBasePtr> &input_args);
using PrimInplaceSubPtr = std::shared_ptr<InplaceSub>;
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_INPLACE_SUB_H_
