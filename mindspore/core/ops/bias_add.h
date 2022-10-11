/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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
#include "ops/base_operator.h"
#include "mindapi/base/types.h"
#include "mindapi/base/format.h"

namespace mindspore {
namespace ops {
constexpr auto kNameBiasAdd = "BiasAdd";

/// \brief Returns sum of input and bias tensor. Refer to Python API @ref mindspore.ops.BiasAdd for more details.
class MIND_API BiasAdd : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(BiasAdd);
  /// \brief Constructor.
  BiasAdd() : BaseOperator(kNameBiasAdd) { InitIOName({"x", "b"}, {"output"}); }
  /// \brief Init. Refer to the parameters of Python API @ref mindspore.ops.BiasAdd for the inputs.
  void Init(const Format &format = NCHW);
  /// \brief Set format.
  void set_format(const Format &format);
  /// \brief Get format.
  ///
  /// \return format.
  Format get_format() const;
  std::string get_str_format() const;
};
abstract::AbstractBasePtr BiasAddInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                       const std::vector<abstract::AbstractBasePtr> &input_args);
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_BIAS_ADD_H_
