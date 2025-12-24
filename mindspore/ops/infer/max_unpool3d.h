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

#ifndef MINDSPORE_CORE_OPS_MAXUNPOOL3D_H_
#define MINDSPORE_CORE_OPS_MAXUNPOOL3D_H_
#include <memory>
#include <string>
#include <vector>

#include "mindapi/base/types.h"
#include "ops/base_operator.h"

namespace mindspore {
namespace ops {
constexpr auto kNameMaxUnpool3D = "MaxUnpool3D";
class OPS_API MaxUnpool3D : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(MaxUnpool3D);
  MaxUnpool3D() : BaseOperator(kNameMaxUnpool3D) { InitIOName({"x", "argmax"}, {"y"}); }
  std::string get_format() const;
};

OPS_API abstract::AbstractBasePtr MaxUnpool3DInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                                   const std::vector<abstract::AbstractBasePtr> &input_args);
using PrimMaxUnpool3DPtr = std::shared_ptr<MaxUnpool3D>;
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_MAXUNPOOL3D_H_
