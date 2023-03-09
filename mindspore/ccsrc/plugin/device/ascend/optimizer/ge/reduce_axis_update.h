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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_OPTIMIZER_GE_REDUCE_AXIS_UPDATE_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_OPTIMIZER_GE_REDUCE_AXIS_UPDATE_H_

#include "backend/common/optimizer/optimizer.h"
#include "backend/common/optimizer/pattern_to_pattern.h"

namespace mindspore {
namespace opt {
class ReduceAxisUpdate : public PatternToPatternPass {
 public:
  ReduceAxisUpdate() : PatternToPatternPass("reduce_axis_update") {}
  ~ReduceAxisUpdate() override = default;

  void DefineSrcPattern(SrcPattern *src_pattern) override;
  void DefineDstPattern(DstPattern *dst_pattern) override;
  bool CheckMatchedDAG(const PatternMap &, const FuncGraphPtr &, const AnfNodePtr &) const override;

 private:
  bool IsAxisEmpty(const ValueNodePtr &axis_node) const;
  bool IsInputScalar(const AnfNodePtr &x_node) const;
  static bool IsReduce(const BaseRef &ref);
};
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_OPTIMIZER_GE_REDUCE_AXIS_UPDATE_H_
