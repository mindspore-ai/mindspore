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

#ifndef MINDSPORE_CORE_OPS_L2_NORMALIZE_FUSION_H_
#define MINDSPORE_CORE_OPS_L2_NORMALIZE_FUSION_H_
#include <vector>

#include "ops/l2_normalize.h"
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace ops {
constexpr auto kNameL2NormalizeFusion = "L2NormalizeFusion";
class L2NormalizeFusion : public L2Normalize {
 public:
  L2NormalizeFusion() : L2Normalize(kNameL2NormalizeFusion) {}
  ~L2NormalizeFusion() = default;
  MS_DECLARE_PARENT(L2NormalizeFusion, L2Normalize);
  void Init(const std::vector<int64_t> &axis, const float epsilon = 1e-4,
            const ActivationType &activation_type = NO_ACTIVATION);
  void set_activation_type(const ActivationType &activation_type);
  ActivationType get_activation_type() const;
};
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_L2_NORMALIZE_FUSION_H_
