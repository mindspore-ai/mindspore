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
#ifndef MINDSPORE_CCSRC_BACKEND_OPTIMIZER_ASCEND_IR_FISSION_UNSORTED_SEGMENT_SUM_REPLACE_FISSION_H_
#define MINDSPORE_CCSRC_BACKEND_OPTIMIZER_ASCEND_IR_FISSION_UNSORTED_SEGMENT_SUM_REPLACE_FISSION_H_

#include <vector>
#include <memory>
#include "backend/common/optimizer/optimizer.h"
#include "backend/common/optimizer/helper.h"
#include "plugin/device/ascend/optimizer/ascend_helper.h"

namespace mindspore {
namespace opt {
class UnsortedSegmentSumReplaceFission : public PatternProcessPass {
 public:
  explicit UnsortedSegmentSumReplaceFission(bool multigraph = true)
      : PatternProcessPass("unsorted_segment_sum_replace_fission", multigraph) {}
  ~UnsortedSegmentSumReplaceFission() override = default;
  const BaseRef DefinePattern() const override;
  const AnfNodePtr Process(const FuncGraphPtr &graph, const AnfNodePtr &node, const EquivPtr &) const override;

 private:
  bool CheckInputs(const AnfNodePtr &node) const;
  bool IsNeedReplaced(const AnfNodePtr &node) const;
  ValueNodePtr CreateNumSegmentsValueNode(int64_t num_segments) const;
  const AnfNodePtr ReplaceByUnsortedSegmentSumD(const FuncGraphPtr &graph, const AnfNodePtr &node) const;
};
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_OPTIMIZER_ASCEND_IR_FISSION_UNSORTED_SEGMENT_REPLACE_SUM_FISSION_H_
