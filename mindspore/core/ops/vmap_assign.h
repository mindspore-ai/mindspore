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

#ifndef MINDSPORE_CORE_OPS_VMAP_ASSIGN_H_
#define MINDSPORE_CORE_OPS_VMAP_ASSIGN_H_

#include <map>
#include <set>
#include <vector>
#include <string>

#include "ops/base_operator.h"
#include "mindapi/base/types.h"

namespace mindspore {
namespace ops {
constexpr auto kNameVmapStackAssign = "VmapStackAssign";
constexpr auto kNameVmapUnstackAssign = "VmapUnstackAssign";
constexpr int64_t kInputLowerLimit = 4;
constexpr size_t kNumber2 = 2;

// Assign value from a batch of parameters to a stacked parameter in the model ensembling scenario of vmap.
class MIND_API VmapStackAssign : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(VmapStackAssign);
  /// \brief Constructor.
  VmapStackAssign() : BaseOperator(kNameVmapStackAssign) {}
};

// Assign value from a stacked parameter to a batch of parameters in the model ensembling scenario of vmap.
class MIND_API VmapUnstackAssign : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(VmapUnstackAssign);
  /// \brief Constructor.
  VmapUnstackAssign() : BaseOperator(kNameVmapUnstackAssign) {}
};

MIND_API abstract::AbstractBasePtr VmapAssignInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                                   const std::vector<abstract::AbstractBasePtr> &input_args);
}  // namespace ops
}  // namespace mindspore
#endif  // MINDSPORE_CORE_OPS_VMAP_ASSIGN_H_
