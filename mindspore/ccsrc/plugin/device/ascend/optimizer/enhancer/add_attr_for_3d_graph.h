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

#ifndef MINDSPORE_CCSRC_PRE_ACTIVATE_ASCEND_ENHANCER_ADD_ATTR_FOR_3D_GRAPH_H
#define MINDSPORE_CCSRC_PRE_ACTIVATE_ASCEND_ENHANCER_ADD_ATTR_FOR_3D_GRAPH_H
#include <string>
#include <memory>
#include "ir/anf.h"
#include "include/backend/optimizer/optimizer.h"
#include "plugin/device/ascend/optimizer/ascend_helper.h"
#include "include/backend/optimizer/pattern_to_pattern.h"

namespace mindspore {
namespace opt {
class AddIoFormatAttrFor3DGraph : public PatternToPatternPass {
 public:
  AddIoFormatAttrFor3DGraph() : PatternToPatternPass("add_attr_for_3d_graph") {}
  ~AddIoFormatAttrFor3DGraph() override = default;
  void DefineSrcPattern(SrcPattern *src_pattern) override;
  void DefineDstPattern(DstPattern *dst_pattern) override;
  bool CheckMatchedDAG(const PatternMap &, const FuncGraphPtr &, const AnfNodePtr &) const override;
};
}  // namespace opt
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PRE_ACTIVATE_ASCEND_ENHANCER_ADD_ATTR_FOR_3D_GRAPH_H
