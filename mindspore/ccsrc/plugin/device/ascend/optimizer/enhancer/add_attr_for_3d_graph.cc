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

#include "plugin/device/ascend/optimizer/enhancer/add_attr_for_3d_graph.h"

namespace mindspore {
namespace opt {
namespace {
constexpr auto kM3d = "m_3d";
constexpr auto kV = "V";
constexpr auto kXs = "Xs";
constexpr auto kR3d = "r_3d";
}  // namespace

bool AddIoFormatAttrFor3DGraph::CheckMatchedDAG(const PatternMap &m, const FuncGraphPtr &graph,
                                                const AnfNodePtr &node) const {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(node);
  if (AnfUtils::IsRealKernel(node)) {
    return true;
  }
  return false;
}

AnfNodePtr AddAttr(const PatternMap &m, const AnfNodePtr & /* default_cnode */) {
  auto node = m.Get(kM3d);
  common::AnfAlgo::SetNodeAttr(kAttrVisited, MakeValue(true), node);
  auto formats = AnfAlgo::GetAllOutputFormats(node);
  if (std::any_of(formats.begin(), formats.end(), [](const std::string &format) { return IsOneOf3DFormat(format); })) {
    common::AnfAlgo::SetNodeAttr(kAttrFormat, MakeValue(kOpFormat_NCDHW), node);
  }
  return node;
}
void AddIoFormatAttrFor3DGraph::DefineSrcPattern(SrcPattern *src_pattern) {
  (void)(*src_pattern).AddVar(kV, UnVisited).AddSeqVar(kXs).AddCNode(kM3d, {kV, kXs});
}
void AddIoFormatAttrFor3DGraph::DefineDstPattern(DstPattern *dst_pattern) {
  (void)(*dst_pattern).AddCNode(kR3d, {kV, kXs}, AddAttr);
}
}  // namespace opt
}  // namespace mindspore
