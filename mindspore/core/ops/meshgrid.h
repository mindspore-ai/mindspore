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

#ifndef MINDSPORE_CORE_OPS_MESHGRID_H_
#define MINDSPORE_CORE_OPS_MESHGRID_H_

#include <memory>
#include <string>
#include <vector>
#include "ops/base_operator.h"

namespace mindspore {
namespace ops {
constexpr auto kNameMeshgrid = "Meshgrid";
/// \brief Computes the maximum of input tensors element-wise.
/// Refer to Python API @ref mindspore.ops.Meshgrid for more details.
class MIND_API Meshgrid : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(Meshgrid);
  /// \brief Constructor.
  Meshgrid() : BaseOperator(kNameMeshgrid) { InitIOName({"inputs"}, {"outputs"}); }
  /// \brief Init. Refer to the parameters of Python API @ref mindspore.ops.Meshgrid for the inputs.
  void Init(const std::string &indexing = "xy");
  /// \brief Method to set lambd. The Default value is 0.5.
  void set_indexing(const std::string &indexing = "xy");
  /// \brief Method to get lambd.
  std::string get_indexing() const;
};

MIND_API abstract::AbstractBasePtr MeshgridInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                                 const std::vector<abstract::AbstractBasePtr> &input_args);
using kPrimMeshgridPtr = std::shared_ptr<Meshgrid>;
}  // namespace ops
}  // namespace mindspore
#endif  // MINDSPORE_CORE_OPS_MESHGRID_H_
