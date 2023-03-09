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
#ifndef MINDSPORE_CCSRC_PRE_ACTIVATE_ASCEND_ENHANCER_CONCAT_OUTPUTS_FOR_ALLGATHER_H_
#define MINDSPORE_CCSRC_PRE_ACTIVATE_ASCEND_ENHANCER_CONCAT_OUTPUTS_FOR_ALLGATHER_H_

#include <memory>
#include <vector>
#include <string>
#include <tuple>
#include "include/backend/optimizer/optimizer.h"
#include "plugin/device/ascend/optimizer/ascend_helper.h"

namespace mindspore {
namespace opt {
constexpr int64_t kConcatInputDivisor = 63;

using OutputInfo =
  std::tuple<std::vector<TypeId>, std::vector<ShapeVector>, std::vector<std::string>, std::vector<TypeId>>;

class ConcatOutputsForAllGather : public PatternProcessPass {
 public:
  explicit ConcatOutputsForAllGather(bool multigraph = true)
      : PatternProcessPass("concat_outputs_for_all_gather", multigraph),
        inputs_divisor_(kConcatInputDivisor),
        kernel_select_(std::make_shared<KernelSelect>()) {}
  ~ConcatOutputsForAllGather() override = default;
  const BaseRef DefinePattern() const override;
  const AnfNodePtr Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node, const EquivPtr &) const override;

 private:
  AnfNodePtr InsertConcatForOutput(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                   const OutputInfo &output_info, const std::vector<AnfNodePtr> &new_tuple_getitems,
                                   int64_t rank_size) const;
  CNodePtr CreateNewConcat(const FuncGraphPtr &func_graph, const std::vector<AnfNodePtr> &concat_input_nodes,
                           const OutputInfo &concat_input_info, size_t begin_index, int64_t offset) const;
  int64_t inputs_divisor_;
  KernelSelectPtr kernel_select_;
};
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PRE_ACTIVATE_ASCEND_ENHANCER_CONCAT_OUTPUTS_FOR_ALLGATHER_H_
