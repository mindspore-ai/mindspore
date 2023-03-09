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
#include "plugin/device/ascend/optimizer/ir_fusion/unsorted_segment_sum_replace.h"

#include <string>
#include <vector>
#include <memory>
#include <set>
#include "utils/hash_set.h"
#include "backend/common/pass/const_input_to_attr.h"
#include "kernel/kernel_build_info.h"
#include "include/common/utils/utils.h"
#include "include/backend/kernel_graph.h"
#include "include/backend/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "include/backend/kernel_info.h"
#include "utils/ms_context.h"
#include "plugin/device/ascend/optimizer/optimizer_factory.h"

namespace mindspore::opt {
namespace {
constexpr auto kNumSegments = "num_segments";
}  // namespace

const BaseRef UnsortedSegmentSumReplace::DefinePattern() const {
  VarPtr Xs = std::make_shared<SeqVar>();
  auto prim = std::make_shared<Primitive>(kUnsortedSegmentSumDOpName);
  return VectorRef({prim, Xs});
}

const AnfNodePtr UnsortedSegmentSumReplace::Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                                    const EquivPtr &) const {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(node);
  auto kernel_graph = func_graph->cast<KernelGraphPtr>();
  MS_EXCEPTION_IF_NULL(kernel_graph);
  auto cnode = CheckAnfNodeIfCNodeAndInputSize(node, kUnsortedSegmentSumDInputTensorNum);
  if (!common::AnfAlgo::HasNodeAttr(kNumSegments, cnode)) {
    MS_LOG(INFO) << "Has no num_segments attr.";
    return nullptr;
  }

  // Convert attr num_segments to the tensor input
  auto num_segments = common::AnfAlgo::GetNodeAttr<int64_t>(node, kNumSegments);
  const auto num_segments_type = kInt32;
  auto value_node =
    kernel_graph->NewValueNode(std::make_shared<tensor::Tensor>(static_cast<int32_t>(num_segments), num_segments_type));
  MS_EXCEPTION_IF_NULL(value_node);
  // create UnsortedSegmentSum
  std::vector<AnfNodePtr> new_inputs{NewValueNode(std::make_shared<Primitive>(kUnsortedSegmentSumOpName))};
  (void)new_inputs.insert(new_inputs.cend(), cnode->inputs().cbegin() + 1, cnode->inputs().cend());
  new_inputs.push_back(value_node);
  CNodePtr new_cnode = NewCNode(new_inputs, func_graph);
  MS_EXCEPTION_IF_NULL(new_cnode);
  new_cnode->set_abstract(cnode->abstract());
  new_cnode->set_scope(cnode->scope());

  if (!CheckAICoreSupportedAny(new_cnode)) {
    MS_LOG(INFO) << "Replace unsorted_segment_sum_d op to unsorted_segment_sum op failed.";
    return nullptr;
  }

  if (common::AnfAlgo::HasNodeAttr(kAttrCustAicpu, cnode)) {
    common::AnfAlgo::SetNodeAttr(kAttrCustAicpu, MakeValue(kUnsortedSegmentSumOpName), new_cnode);
  }
  new_cnode->set_primal_attrs(cnode->primal_attrs());
  new_cnode->set_attrs(cnode->attrs());
  SetInputOutputNames({"x", "segment_ids", "num_segments"}, {"y"}, new_cnode);
  MS_LOG(INFO) << "Replace unsorted_segment_sum_d op to unsorted_segment_sum op success. use tbe aicore.";
  return new_cnode;
}
}  // namespace mindspore::opt
