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
#ifndef MINDSPORE_CCSRC_PRE_ACTIVATE_ASCEND_ENHANCER_SPLIT_INPUTS_FOR_REDUCE_SCATTER_H_
#define MINDSPORE_CCSRC_PRE_ACTIVATE_ASCEND_ENHANCER_SPLIT_INPUTS_FOR_REDUCE_SCATTER_H_

#include <memory>
#include <vector>
#include "include/backend/optimizer/optimizer.h"
#include "plugin/device/ascend/optimizer/ascend_helper.h"

namespace mindspore {
namespace opt {
class SplitInputsForReduceScatter : public PatternProcessPass {
 public:
  explicit SplitInputsForReduceScatter(bool multigraph = true)
      : PatternProcessPass("split_inputs_for_reduce_scatter", multigraph),
        kernel_select_(std::make_shared<KernelSelect>()) {}
  ~SplitInputsForReduceScatter() override = default;
  const BaseRef DefinePattern() const override;
  const AnfNodePtr Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node, const EquivPtr &) const override;

 private:
  AnfNodePtr RearrangeInputsForReduceScatter(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                             const std::vector<AnfNodePtr> &inputs, int64_t rank_size) const;
  std::vector<AnfNodePtr> InsertSplitForInput(const FuncGraphPtr &func_graph, const CNodePtr &node,
                                              int64_t rank_size) const;
  KernelSelectPtr kernel_select_;
};
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PRE_ACTIVATE_ASCEND_ENHANCER_SPLIT_INPUTS_FOR_REDUCE_SCATTER_H_
