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

#ifndef MINDSPORE_CORE_OPS_TOPK_FUSION_H_
#define MINDSPORE_CORE_OPS_TOPK_FUSION_H_
#include <vector>

#include "ops/topk.h"
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace ops {
constexpr auto kNameTopKFusion = "TopKFusion";
class TopKFusion : public TopK {
 public:
  TopKFusion() : TopK(kNameTopKFusion) {}
  ~TopKFusion() = default;
  MS_DECLARE_PARENT(TopKFusion, TopK);
  void Init(const bool sorted, const int64_t axis, const int64_t largest);
  void set_axis(const int64_t axis);
  void set_largest(const int64_t largest);
  int64_t get_axis() const;
  int64_t get_largest() const;
};
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_TOPK_FUSION_H_
