/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_BACKEND_OPTIMIZER_ASCEND_IR_FISSION_UNSORTED_SEGMENT_SUM_D_FISSION_H_
#define MINDSPORE_CCSRC_BACKEND_OPTIMIZER_ASCEND_IR_FISSION_UNSORTED_SEGMENT_SUM_D_FISSION_H_

#include <vector>
#include <memory>
#include "include/backend/optimizer/optimizer.h"
#include "include/backend/optimizer/helper.h"
#include "plugin/device/ascend/optimizer/ascend_helper.h"

namespace mindspore {
namespace opt {
class UnsortedSegmentSumDFission : public PatternProcessPass {
 public:
  explicit UnsortedSegmentSumDFission(bool multigraph = true)
      : PatternProcessPass("unsorted_segment_sum_d_fission", multigraph) {}
  ~UnsortedSegmentSumDFission() override = default;
  const BaseRef DefinePattern() const override;
  const AnfNodePtr Process(const FuncGraphPtr &graph, const AnfNodePtr &node, const EquivPtr &) const override;

 private:
  CNodePtr CreatePadding(const FuncGraphPtr &graph, const CNodePtr &origin_node, const size_t &pad_dim_size) const;
  CNodePtr CreateUnsortedSegmentSum(const FuncGraphPtr &graph, const CNodePtr &origin_node, const CNodePtr &padding,
                                    const size_t &pad_dim_size) const;
  CNodePtr CreateSlice(const FuncGraphPtr &graph, const CNodePtr &unsort_segment_sum,
                       const CNodePtr &unsorted_segment_sum8) const;
};
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_OPTIMIZER_ASCEND_IR_FISSION_UNSORTED_SEGMENT_SUM_D_FISSION_H_
