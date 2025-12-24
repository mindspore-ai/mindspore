/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CORE_OPS_APPLY_RMS_PROP_H_
#define MINDSPORE_CORE_OPS_APPLY_RMS_PROP_H_
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "mindapi/base/types.h"
#include "ops/base_operator.h"

namespace mindspore {
namespace ops {
constexpr auto kNameApplyRMSProp = "ApplyRMSProp";
class OPS_API ApplyRMSProp : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(ApplyRMSProp);
  float get_attr(const char *attr) const;
  ApplyRMSProp() : BaseOperator(kNameApplyRMSProp) {
    InitIOName({"var", "mean_square", "moment", "learning_rate", "grad", "decay", "momentum", "epsilon"},
               {"var", "mean_square", "moment"});
  }
};
using kPrimApplyRMSPropPtr = std::shared_ptr<ApplyRMSProp>;
}  // namespace ops
}  // namespace mindspore
#endif  // MINDSPORE_CORE_OPS_APPLY_RMS_PROP_H_
