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

#ifndef MINDSPORE_CORE_OPS_ADDER_FUSION_H_
#define MINDSPORE_CORE_OPS_ADDER_FUSION_H_

#include <map>
#include <vector>
#include <string>
#include <memory>
#include "ops/adder.h"
#include "abstract/abstract_value.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace ops {
constexpr auto kNameAdderFusion = "AdderFusion";
class AdderFusion : public Adder {
 public:
  AdderFusion() : Adder(kNameAdderFusion) {}
  ~AdderFusion() = default;
  MS_DECLARE_PARENT(AdderFusion, Adder);
  void Init(const int64_t in_channel, const int64_t out_channel, const std::vector<int64_t> &kernel_size,
            const PadMode &pad_mode, const std::vector<int64_t> &stride, const std::vector<int64_t> &pad_list,
            const std::vector<int64_t> &dilation, const int64_t group, const Format &format,
            const ActivationType activation_type);
  void set_activation_type(const ActivationType activation_type);

  ActivationType get_activation_type() const;
};
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_ADDER_FUSION_H_
