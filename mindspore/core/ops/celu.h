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

#ifndef MINDSPORE_CORE_OPS_CELU_H_
#define MINDSPORE_CORE_OPS_CELU_H_

#include <map>
#include <memory>
#include <vector>
#include <string>
#include "ops/primitive_c.h"
#include "abstract/abstract_value.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace ops {
constexpr auto kNameCeLU = "CeLU";
/// \brief Computes CeLU (Continuously differentiable exponential linear units) of input tensors element-wise.
/// Refer to Python API @ref mindspore.ops.CeLU for more details.
class CeLU : public PrimitiveC {
 public:
  /// \brief Constructor.
  CeLU() : PrimitiveC(kNameCeLU) { InitIOName({"x"}, {"output"}); }
  /// \brief Destructor.
  ~CeLU() = default;
  MS_DECLARE_PARENT(CeLU, PrimitiveC);
  /// \brief Init. Refer to the parameters of Python API @ref mindspore.ops.CeLU for the inputs.
  void Init() {}
};

AbstractBasePtr CeLUInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args);
using PrimCeLUPtr = std::shared_ptr<CeLU>;
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_CELU_H_
