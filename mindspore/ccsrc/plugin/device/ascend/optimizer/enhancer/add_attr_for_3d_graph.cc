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
constexpr auto m_3d = "m_3d";
constexpr auto V = "V";
constexpr auto Xs = "Xs";
constexpr auto r_3d = "r_3d";
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
  auto node = m.Get(m_3d);
  common::AnfAlgo::SetNodeAttr(kAttrVisited, MakeValue(true), node);
  auto formats = AnfAlgo::GetAllOutputFormats(node);
  if (std::any_of(formats.begin(), formats.end(), [](const std::string &format) { return IsOneOf3DFormat(format); })) {
    common::AnfAlgo::SetNodeAttr(kAttrFormat, MakeValue(kOpFormat_NCDHW), node);
  }
  return node;
}
void AddIoFormatAttrFor3DGraph::DefineSrcPattern(SrcPattern *src_pattern) {
  (*src_pattern).AddVar(V, UnVisited).AddSeqVar(Xs).AddCNode(m_3d, {V, Xs});
}
void AddIoFormatAttrFor3DGraph::DefineDstPattern(DstPattern *dst_pattern) {
  (*dst_pattern).AddCNode(r_3d, {V, Xs}, AddAttr);
}
}  // namespace opt
}  // namespace mindspore
