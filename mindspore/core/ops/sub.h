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

#ifndef MINDSPORE_CORE_OPS_SUB_H_
#define MINDSPORE_CORE_OPS_SUB_H_
#include <string>
#include <vector>
#include <memory>

#include "ops/primitive_c.h"
#include "abstract/abstract_value.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace ops {
constexpr auto kNameSub = "Sub";
class Sub : public PrimitiveC {
 public:
  Sub() : PrimitiveC(kNameSub) { InitIOName({"x", "y"}, {"output"}); }
  explicit Sub(const std::string k_name) : PrimitiveC(k_name) { InitIOName({"x", "y"}, {"output"}); }
  ~Sub() = default;
  MS_DECLARE_PARENT(Sub, PrimitiveC);
  void Init() {}
};

AbstractBasePtr SubInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                         const std::vector<AbstractBasePtr> &input_args);
using PrimSubPtr = std::shared_ptr<Sub>;
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_SUB_H_
