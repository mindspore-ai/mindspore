/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CORE_OPS_BIAS_ADD_H_
#define MINDSPORE_CORE_OPS_BIAS_ADD_H_
#include <map>
#include <vector>
#include <string>
#include <memory>
#include "ops/primitive_c.h"
#include "abstract/abstract_value.h"
#include "utils/check_convert_utils.h"
// Add
#include "ops/op_utils.h"

namespace mindspore {
namespace ops {
constexpr auto kNameBiasAdd = prim::kBiasAdd;
/// \brief Returns sum of input and bias tensor. Refer to Python API @ref mindspore.ops.BiasAdd for more details.
class MS_CORE_API BiasAdd : public PrimitiveC {
 public:
  /// \brief Constructor.
  BiasAdd() : PrimitiveC(prim::kPrimBiasAdd->name()) { InitIOName({"x", "b"}, {"output"}); }
  /// \brief Destructor.
  ~BiasAdd() = default;
  MS_DECLARE_PARENT(BiasAdd, PrimitiveC);
  /// \brief Init. Refer to the parameters of Python API @ref mindspore.ops.BiasAdd for the inputs.
  void Init(const Format &format = NCHW);
  /// \brief Set format.
  void set_format(const Format &format);
  /// \brief Get format.
  ///
  /// \return format.
  Format get_format() const;
};
AbstractBasePtr BiasAddInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                             const std::vector<AbstractBasePtr> &input_args);
using PrimBiasAddPtr = std::shared_ptr<BiasAdd>;
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_BIAS_ADD_H_
