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

#include "plugin/device/ascend/optimizer/format_type/reselect_call_inline_format.h"
#include "plugin/device/ascend/optimizer/ascend_helper.h"

namespace mindspore {
namespace opt {
namespace {
constexpr auto kXs = "Xs";
constexpr auto call_inline = "call_inline";
constexpr auto new_call_inline = "new_call_inline";
}  // namespace
bool ReselectCallInlineFormat::CheckMatchedDAG(const PatternMap &, const FuncGraphPtr &graph,
                                               const AnfNodePtr &node) const {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(node);
  return true;
}

AnfNodePtr BuildCallInline(const PatternMap &m, const AnfNodePtr &) {
  auto anf = m.Get(call_inline);
  MS_EXCEPTION_IF_NULL(anf);
  auto cnode = anf->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  SelectCallInlineKernelInfo(cnode);
  return cnode;
}

void ReselectCallInlineFormat::DefineSrcPattern(SrcPattern *src_pattern) {
  (*src_pattern).AddSeqVar(kXs).AddCNode(call_inline, {prim::kPrimCallInline, kXs});
}

void ReselectCallInlineFormat::DefineDstPattern(DstPattern *dst_pattern) {
  (*dst_pattern).AddCNode(new_call_inline, {prim::kPrimCallInline, kXs}, BuildCallInline);
}
}  // namespace opt
}  // namespace mindspore
