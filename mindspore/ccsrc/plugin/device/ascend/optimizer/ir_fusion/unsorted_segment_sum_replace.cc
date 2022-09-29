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
#include "backend/common/optimizer/const_input_to_attr.h"
#include "kernel/kernel_build_info.h"
#include "include/common/utils/utils.h"
#include "backend/common/session/kernel_graph.h"
#include "backend/common/session/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "runtime/device/kernel_info.h"
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
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  auto primitive = common::AnfAlgo::GetCNodePrimitive(cnode);
  MS_EXCEPTION_IF_NULL(primitive);
  common::AnfAlgo::SetNodeAttr(kAttrVisited, MakeValue(true), node);
  if (cnode->inputs().size() == 0) {
    return nullptr;
  }
  if (!common::AnfAlgo::HasNodeAttr(kNumSegments, cnode)) {
    MS_LOG(INFO) << "Has no num_segments attr.";
    return nullptr;
  }

  // Copy a new node to check supported.
  std::vector<AnfNodePtr> new_inputs{NewValueNode(std::make_shared<Primitive>(kUnsortedSegmentSumOpName))};
  (void)new_inputs.insert(new_inputs.cend(), cnode->inputs().cbegin() + 1, cnode->inputs().cend());
  CNodePtr new_cnode = NewCNode(new_inputs, func_graph);
  MS_EXCEPTION_IF_NULL(new_cnode);
  new_cnode->set_abstract(cnode->abstract());
  new_cnode->set_scope(cnode->scope());
  CheckCNodeInputSize(new_cnode, kUnsortedSegmentSumInputTensorNum);
  // Convert attr num_segments to the tensor input
  auto value = primitive->GetAttr(kNumSegments);
  if (value == nullptr) {
    MS_LOG(INFO) << "Can not get attr[" << kNumSegments << "] num_segments.";
    return nullptr;
  }
  tensor::TensorPtr tensor_ptr = nullptr;
  if (value->isa<tensor::Tensor>()) {
    tensor_ptr = value->cast<tensor::TensorPtr>();
  } else if (value->isa<Scalar>()) {
    tensor_ptr = ScalarToTensor(value->cast<ScalarPtr>());
  } else if (value->isa<ValueTuple>()) {
    tensor_ptr = opt::CreateTupleTensor(value->cast<ValueTuplePtr>());
  } else {
    MS_LOG(INFO) << "The value of attr[" << kNumSegments << "] should be a tensor or scalar or value tuple.";
    return nullptr;
  }
  if (tensor_ptr == nullptr) {
    MS_LOG(INFO) << "Convert attr[" << kNumSegments << "] to tensor value failed.";
    return nullptr;
  }
  auto value_node = kernel_graph->NewValueNode(tensor_ptr);
  MS_EXCEPTION_IF_NULL(value_node);
  new_inputs.push_back(value_node);
  new_cnode->set_inputs(new_inputs);
  if (!CheckAICoreSupportedAny(new_cnode)) {
    MS_LOG(INFO) << "Replace unsorted_segment_sum_d op to unsorted_segment_sum op failed.";
    return nullptr;
  }

  MS_LOG(INFO) << "Replace unsorted_segment_sum_d op to unsorted_segment_sum op success. use tbe aicore.";
  return new_cnode;
}
}  // namespace mindspore::opt
