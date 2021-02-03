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

#ifndef MINDSPORE_CORE_OPS_AVG_POOL_FUSION_H_
#define MINDSPORE_CORE_OPS_AVG_POOL_FUSION_H_
#include <vector>
#include <memory>

#include "ops/avg_pool.h"
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace ops {
constexpr auto kNameAvgPoolFusion = "AvgPoolFusion";
class AvgPoolFusion : public AvgPool {
 public:
  AvgPoolFusion() : AvgPool(kNameAvgPoolFusion) { InitIOName({"x"}, {"output"}); }
  ~AvgPoolFusion() = default;
  MS_DECLARE_PARENT(AvgPoolFusion, AvgPool);
  void Init(const std::vector<int64_t> &kernel_size = {1}, const std::vector<int64_t> &stride = {1},
            const PadMode &pad_mode = VALID, const Format &format = NCHW,
            const std::vector<int64_t> &pad = {0, 0, 0, 0}, const RoundMode &round_mode = FLOOR,
            const bool global = false, const ActivationType activation_type = NO_ACTIVATION);
  void set_global(const bool global);
  void set_activation_type(const ActivationType activation_type);
  bool get_global() const;
  ActivationType get_activation_type() const;
};

AbstractBasePtr AvgPoolFusionInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                   const std::vector<AbstractBasePtr> &input_args);
using PrimAvgPoolFusionPtr = std::shared_ptr<AvgPoolFusion>;
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_AVG_POOL_FUSION_H_
