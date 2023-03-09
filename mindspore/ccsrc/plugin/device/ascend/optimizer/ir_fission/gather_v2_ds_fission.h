/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_BACKEND_OPTIMIZER_ASCEND_IR_FISSION_GATHER_V2_DS_FISSION_H_
#define MINDSPORE_CCSRC_BACKEND_OPTIMIZER_ASCEND_IR_FISSION_GATHER_V2_DS_FISSION_H_

#include <vector>
#include <memory>
#include "include/backend/optimizer/optimizer.h"
#include "include/backend/optimizer/helper.h"
#include "plugin/device/ascend/optimizer/ascend_helper.h"

namespace mindspore {
namespace opt {
class GatherV2DsFission : public PatternProcessPass {
 public:
  explicit GatherV2DsFission(bool multigraph = true) : PatternProcessPass("gather_v2_ds_fission", multigraph) {}
  ~GatherV2DsFission() override = default;
  const BaseRef DefinePattern() const override;
  const AnfNodePtr Process(const FuncGraphPtr &graph, const AnfNodePtr &node, const EquivPtr &) const override;

 private:
  CNodePtr CreatePad(const FuncGraphPtr &graph, const CNodePtr &origin_node, const size_t &pad_dim_size) const;
  CNodePtr CreateGatherV2Ds(const FuncGraphPtr &graph, const CNodePtr &origin_node, const CNodePtr &pad,
                            const size_t &pad_dim_size) const;
  CNodePtr CreateSlice(const FuncGraphPtr &graph, const CNodePtr &gather_v2, const CNodePtr &gather_v2_padding_8) const;
};
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_OPTIMIZER_ASCEND_IR_FISSION_GATHER_V2_DS_FISSION_H_
