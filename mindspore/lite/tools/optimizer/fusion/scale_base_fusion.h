/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_LITE_TOOLS_OPTIMIZER_FUSION_SCALE_BASE_FUSION_H_
#define MINDSPORE_LITE_TOOLS_OPTIMIZER_FUSION_SCALE_BASE_FUSION_H_

#include <vector>
#include <string>
#include <unordered_map>
#include "tools/optimizer/common/pattern_process_pass_extends.h"
#include "ops/fusion/scale_fusion.h"
#include "tools/optimizer/common/gllo_utils.h"

namespace mindspore::opt {
class ScaleBaseFusion : public LitePatternProcessPass {
 public:
  explicit ScaleBaseFusion(std::string name, bool multigraph = true) : LitePatternProcessPass(name, multigraph) {}
  ~ScaleBaseFusion() override = default;
  const AnfNodePtr Process(const FuncGraphPtr &, const AnfNodePtr &, const EquivPtr &) const override;

 protected:
  bool CheckCurrCnodeProper(const CNodePtr &curr_cnode) const;
  virtual bool CheckPrevCnodeProper(const CNodePtr &prev_cnode) const = 0;
  virtual PrimitiveCPtr BuildNewPrimitive(const CNodePtr &curr_cnode, const CNodePtr &prev_cnode) const = 0;
  std::vector<AnfNodePtr> GetNewCnodeInputs(const FuncGraphPtr &func_graph, const CNodePtr &curr_cnode,
                                            const CNodePtr &prev_cnode) const;
  int CalNewCnodeBias(const FuncGraphPtr &func_graph, const CNodePtr &curr_cnode,
                      const std::vector<AnfNodePtr> &fusion_cnode_inputs) const;
  int CalNewCnodeScale(const CNodePtr &curr_cnode, const std::vector<AnfNodePtr> &fusion_cnode_inputs) const;
  virtual int CalNewBiasImpl(float *curr_weight_data, float *curr_bias_data, std::vector<int64_t> prev_bias_shape,
                             float *prev_bias_data) const = 0;
  virtual int CalNewScaleImpl(float *curr_weight_data, std::vector<int64_t> prev_weight_shape, float *prev_weight_data,
                              const AnfNodePtr &prim) const = 0;
};
}  // namespace mindspore::opt
#endif  // MINDSPORE_LITE_SRC_PASS_FUSION_SCALE_BASE_FUSION_H_
