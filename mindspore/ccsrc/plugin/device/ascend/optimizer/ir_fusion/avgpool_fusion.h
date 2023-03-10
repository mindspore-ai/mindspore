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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_OPTIMIZER_IR_FUSION_AVGPOOL_FUSION_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_OPTIMIZER_IR_FUSION_AVGPOOL_FUSION_H_
#include <vector>
#include <memory>
#include <string>
#include "include/backend/optimizer/optimizer.h"
#include "include/common/utils/utils.h"

namespace mindspore {
namespace opt {
class AvgPoolFusion : public PatternProcessPass {
 public:
  explicit AvgPoolFusion(bool multigraph = true) : PatternProcessPass("avg_pool_fusion", multigraph) {}
  ~AvgPoolFusion() override = default;
  const BaseRef DefinePattern() const override;
  const AnfNodePtr Process(const FuncGraphPtr &, const AnfNodePtr &, const EquivPtr &) const override;

 private:
  ValueNodePtr ConstructFilterValueNode(const KernelGraphPtr &graph, float factor,
                                        const std::vector<int64_t> &assist_shape) const;
  ValueNodePtr ConstructFilterValueNodeDynamic(const KernelGraphPtr &graph, float factor,
                                               const std::vector<int64_t> &assist_shape,
                                               const std::vector<int64_t> &host_shape) const;
  ValueNodePtr ConstructCoffeValueNode(const KernelGraphPtr &graph, const std::string &format,
                                       const std::vector<int64_t> &avg_in_shape,
                                       const std::vector<int64_t> &avg_out_shape, const std::vector<int64_t> &window,
                                       const std::vector<int64_t> &stride) const;
  AnfNodePtr AddMul(const KernelGraphPtr &graph, const CNodePtr &avgpool, const AnfNodePtr &coffe) const;
};
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_OPTIMIZER_IR_FUSION_AVGPOOL_FUSION_H_
