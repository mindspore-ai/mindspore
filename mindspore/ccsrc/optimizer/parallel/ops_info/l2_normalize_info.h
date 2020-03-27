/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_OPTIMIZER_PARALLEL_OPS_INFO_L2_NORMALIZE_INFO_H_
#define MINDSPORE_CCSRC_OPTIMIZER_PARALLEL_OPS_INFO_L2_NORMALIZE_INFO_H_

#include <string>
#include <list>
#include <unordered_map>
#include <vector>
#include <memory>

#include "ir/value.h"
#include "optimizer/parallel/ops_info/activation_info.h"
#include "optimizer/parallel/strategy.h"
#include "optimizer/parallel/auto_parallel/operator_costmodel.h"

namespace mindspore {
namespace parallel {
class L2NormalizeInfo : public Activation {
 public:
  L2NormalizeInfo(const std::string& name, const Shapes& inputs_shape, const Shapes& outputs_shape,
                  const PrimitiveAttrs& attrs)
      : Activation(name, inputs_shape, outputs_shape, attrs) {
    l2normalizecost_ptr_ = std::make_shared<L2NormalizeCost>();
  }
  ~L2NormalizeInfo() override = default;
  Status GenerateStrategies(int32_t stage_id) override;
  OperatorCostPtr GetOperatorCost() const override { return l2normalizecost_ptr_; }

 protected:
  Status GetAttrs() override;
  Status InferMirrorOps() override;
  Status CheckStrategy(const StrategyPtr& strategy) override;

 private:
  int32_t axis_ = 0;  // Default value = 0
  L2NormalizeCostPtr l2normalizecost_ptr_;
};
}  // namespace parallel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_OPTIMIZER_PARALLEL_OPS_INFO_L2_NORMALIZE_INFO_H_
