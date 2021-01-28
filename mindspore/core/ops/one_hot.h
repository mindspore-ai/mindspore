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

#ifndef MINDSPORE_CORE_OPS_ONE_HOT_H_
#define MINDSPORE_CORE_OPS_ONE_HOT_H_
#include <vector>
#include <memory>

#include "ops/primitive_c.h"
#include "abstract/abstract_value.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace ops {
constexpr auto kNameOneHot = "OneHot";
class OneHot : public PrimitiveC {
 public:
  OneHot() : PrimitiveC(kNameOneHot) { InitIOName({"indices", "depth", "on_value", "off_value"}, {"output"}); }
  ~OneHot() = default;
  MS_DECLARE_PARENT(OneHot, PrimitiveC);
  void Init(const int64_t axis);
  void set_axis(const int64_t axis);
  int64_t get_axis() const;
};

AbstractBasePtr OneHotInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                            const std::vector<AbstractBasePtr> &input_args);
using PrimOneHotPtr = std::shared_ptr<OneHot>;
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_ONE_HOT_H_
