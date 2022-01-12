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
#ifndef MINDSPORE_CORE_OPS_EINSUM_H_
#define MINDSPORE_CORE_OPS_EINSUM_H_

#include <vector>
#include <memory>
#include <unordered_map>
#include <string>
#include "ops/primitive_c.h"
#include "abstract/abstract_value.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace ops {
constexpr auto kNameEinsum = "Einsum";

/// \brief .
/// Refer to Python API @ref mindspore.ops.Einsum for more details.
class Einsum : public PrimitiveC {
 public:
  /// \brief Constructor.
  Einsum() : PrimitiveC(kNameEinsum) { InitIOName({"x"}, {"output"}); }
  /// \brief Destructor.
  ~Einsum() = default;
  MS_DECLARE_PARENT(Einsum, PrimitiveC);
  /// \brief Init. Refer to the parameters of Python API @ref mindspore.ops.Einsum for the inputs.
  void Init(const std::string &equation);
  /// \brief Set equation.
  void set_equation(const std::string &equation);
  /// \brief Get equation.
  ///
  /// \return equation.
  std::string get_equation() const;
};

AbstractBasePtr EinsumInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                            const std::vector<AbstractBasePtr> &input_args);
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_EINSUM_H_
