/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "backend/optimizer/ascend/format_type/trans_op_format_refine.h"
#include <memory>
#include <string>
#include <unordered_map>
#include "backend/session/anf_runtime_algorithm.h"
#include "backend/optimizer/common/helper.h"

namespace mindspore {
namespace opt {
const BaseRef TransOpFormatRefine::DefinePattern() const {
  std::shared_ptr<Var> V = std::make_shared<CondVar>(UnVisited);
  std::shared_ptr<Var> Vs = std::make_shared<SeqVar>();
  return VectorRef({V, Vs});
}

const AnfNodePtr TransOpFormatRefine::Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                              const EquivPtr &) const {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(node);
  AnfAlgo::SetNodeAttr(kAttrVisited, MakeValue(true), node);
  auto op_name = AnfAlgo::GetCNodeName(node);
  if (op_name == kTransDataOpName) {
    auto in_format = AnfAlgo::GetInputFormat(node, 0);
    auto out_format = AnfAlgo::GetOutputFormat(node, 0);
    auto builder =
      std::make_shared<kernel::KernelBuildInfo::KernelBuildInfoBuilder>(AnfAlgo::GetSelectKernelBuildInfo(node));
    if (in_format == kOpFormat_DEFAULT && k3DFormatSet.find(out_format) != k3DFormatSet.end()) {
      builder->SetInputsFormat({kOpFormat_NCDHW});
      builder->SetOutputsFormat({out_format});
      AnfAlgo::SetSelectKernelBuildInfo(builder->Build(), node.get());
      AnfAlgo::SetNodeAttr(kAttrSrcFormat, MakeValue(kOpFormat_NCDHW), node);
    }
    if (out_format == kOpFormat_DEFAULT && k3DFormatSet.find(in_format) != k3DFormatSet.end()) {
      builder->SetInputsFormat({in_format});
      builder->SetOutputsFormat({kOpFormat_NCDHW});
      AnfAlgo::SetSelectKernelBuildInfo(builder->Build(), node.get());
      AnfAlgo::SetNodeAttr(kAttrDstFormat, MakeValue(kOpFormat_NCDHW), node);
    }
  }
  return node;
}
}  // namespace opt
}  // namespace mindspore
