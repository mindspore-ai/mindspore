/**
 * Copyright 2022-2023 Huawei Technologies Co., Ltd
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
#include "plugin/device/ascend/optimizer/mindir/fse_decode_adjust.h"

#include <vector>
#include <memory>
#include <string>

#include "include/common/utils/utils.h"
#include "utils/ms_context.h"
#include "include/backend/optimizer/helper.h"
#include "include/common/utils/anfalgo.h"
#include "utils/trace_base.h"

namespace mindspore {
namespace opt {
std::vector<std::string> FSEDecodeAdjust::MustExistPrimitiveName() const {
  std::vector<std::string> ret;
  ret.emplace_back(std::make_shared<Primitive>(kFSEDecodeOpName)->name());
  return ret;
}

const BaseRef FSEDecodeAdjust::DefinePattern() const {
  VarPtr Xs = std::make_shared<SeqVar>();
  auto prim = std::make_shared<Primitive>(kFSEDecodeOpName);
  return VectorRef({prim, Xs});
}

const AnfNodePtr FSEDecodeAdjust::Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                          const EquivPtr &) const {
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(func_graph);
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  auto primitive = common::AnfAlgo::GetCNodePrimitive(cnode);
  MS_EXCEPTION_IF_NULL(primitive);
  primitive->DelAttr("format");
  primitive->DelAttr("infer_done");
  MS_LOG(INFO) << cnode->fullname_with_scope() << " run FSEDecodeAdjust pass.";
  return node;
}
}  // namespace opt
}  // namespace mindspore
