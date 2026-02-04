/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_BACKEND_OPTIMIZER_MOE_INIT_ROUTING_DYNAMIC_QUANTV2_H_
#define MINDSPORE_CCSRC_BACKEND_OPTIMIZER_MOE_INIT_ROUTING_DYNAMIC_QUANTV2_H_
#include <vector>
#include <string>
#include <memory>
#include "include/backend/common/pass_manager/optimizer.h"

auto constexpr kExpandRowIdx = 1;
auto constexpr kCumsumOutIdx = 2;
auto constexpr kCapacityOutIdx = 3;
auto constexpr kFusedScaleOutIdx = 4;
auto constexpr kScaleOutIdx = 1;

namespace mindspore {
namespace opt {
class MoeInitRoutingDynQuantV2Fusion : public PatternProcessPass {
 public:
  explicit MoeInitRoutingDynQuantV2Fusion(bool multigraph = true)
      : PatternProcessPass("moe_init_routing_dyn_quantv2_fusion", multigraph) {
    x_ = std::make_shared<Var>();
    expert_idx_ = std::make_shared<Var>();
    active_num_ = std::make_shared<Var>();
    expert_capacity_ = std::make_shared<Var>();
    expert_num_ = std::make_shared<Var>();
    drop_pad_mode_ = std::make_shared<Var>();
    expert_tokens_count_or_cumsum_flag_ = std::make_shared<Var>();
    expert_tokens_before_capacity_flag_ = std::make_shared<Var>();
    smooth_scale_ = std::make_shared<Var>();
  }
  ~MoeInitRoutingDynQuantV2Fusion() override = default;
  const BaseRef DefinePattern() const override;
  const AnfNodePtr Process(const FuncGraphPtr &, const AnfNodePtr &, const EquivPtr &) const override;

 private:
  CNodePtr CreateMoeInitRoutingDynQuantV2Node(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                              const EquivPtr &equiv) const;
  std::vector<std::string> MustExistPrimitiveName() const override;
  bool IsSupport(const AnfNodePtr &node, const EquivPtr &equiv) const;
  VarPtr x_;
  VarPtr expert_idx_;
  VarPtr active_num_;
  VarPtr expert_capacity_;
  VarPtr expert_num_;
  VarPtr drop_pad_mode_;
  VarPtr expert_tokens_count_or_cumsum_flag_;
  VarPtr expert_tokens_before_capacity_flag_;
  VarPtr smooth_scale_;
};
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_OPTIMIZER_MOE_INIT_ROUTING_DYNAMIC_QUANTV2_H_
