/**
 * Copyright 2020-2023 Huawei Technologies Co., Ltd
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
#include "plugin/device/ascend/optimizer/ir_fission/unsorted_segment_sum_replace_fission.h"
#include <memory>
#include <vector>
#include <algorithm>
#include "backend/common/session/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "ir/primitive.h"
#include "include/common/utils/utils.h"

namespace mindspore {
namespace opt {
namespace {
// hard-code shape for replacement
// example: fp32[78848, 16], int32[78848, 16], int64[1] 3776224 -> [3776224]
const std::vector<std::vector<std::vector<int64_t>>> UnsortedSegmentSumReplaceShapes = {
  {{78848, 16}, {78848, 16}, {3776224}, {3776224}}, {{262144, 16}, {262144, 16}, {4194304}, {4194304}}};
constexpr auto kInput0Index = 0;
constexpr auto kInput1Index = 1;
constexpr auto kInput2Index = 2;
constexpr auto kOutput0Index = 3;
}  // namespace

ValueNodePtr UnsortedSegmentSumReplaceFission::CreateNumSegmentsValueNode(int64_t num_segments) const {
  // 1. create int64 scalar tensor
  std::vector<int64_t> scalar_shape = {1};
  TensorTypePtr tensor_type = std::make_shared<TensorType>(kInt64);
  MS_EXCEPTION_IF_NULL(tensor_type);
  tensor::DeviceInfo device_info{kOpFormat_DEFAULT, tensor_type};
  tensor::TensorPtr scalar_tensor = std::make_shared<tensor::Tensor>(kInt64->type_id(), scalar_shape);
  MS_EXCEPTION_IF_NULL(scalar_tensor);
  scalar_tensor->set_device_info(device_info);

  // 2. set data of tensor
  auto data_ptr = scalar_tensor->data_c();
  MS_EXCEPTION_IF_NULL(data_ptr);
  auto ptr = static_cast<int64_t *>(data_ptr);
  *ptr = num_segments;

  // 3. create ValueNode based on scalar tensor
  auto value_node = std::make_shared<ValueNode>(scalar_tensor);
  MS_EXCEPTION_IF_NULL(value_node);
  auto abstract = scalar_tensor->ToAbstract();
  value_node->set_abstract(abstract);
  auto value_node_kernel_info = std::make_shared<device::KernelInfo>();
  MS_EXCEPTION_IF_NULL(value_node_kernel_info);
  value_node->set_kernel_info(value_node_kernel_info);
  kernel::KernelBuildInfo::KernelBuildInfoBuilder builder;
  builder.SetOutputsFormat({kOpFormat_DEFAULT});
  builder.SetOutputsDeviceType({kNumberTypeInt64});
  AnfAlgo::SetSelectKernelBuildInfo(builder.Build(), value_node.get());
  return value_node;
}

bool UnsortedSegmentSumReplaceFission::CheckInputs(const AnfNodePtr &node) const {
  MS_EXCEPTION_IF_NULL(node);
  auto origin_node = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(origin_node);
  if (common::AnfAlgo::GetInputTensorNum(origin_node) != kUnsortedSegmentSumInputTensorNum) {
    MS_LOG(DEBUG) << "UnsortedSegmentSum has wrong inputs num, which is "
                  << common::AnfAlgo::GetInputTensorNum(origin_node) << ", not equal "
                  << kUnsortedSegmentSumInputTensorNum << ". CNode= " << origin_node->DebugString();
    return false;
  }
  auto x_shape = common::AnfAlgo::GetPrevNodeOutputInferShape(origin_node, 0);
  auto y_shape = common::AnfAlgo::GetPrevNodeOutputInferShape(origin_node, 1);
  if (x_shape.empty()) {
    MS_LOG(DEBUG) << "UnsortedSegmentSum's first input shape is empty"
                  << ". CNode= " << origin_node->DebugString();
  }
  if (y_shape.empty()) {
    MS_LOG(DEBUG) << "UnsortedSegmentSum's second input shape is empty"
                  << ". CNode= " << origin_node->DebugString();
    return false;
  }
  if (x_shape.size() < y_shape.size()) {
    MS_LOG(DEBUG) << "UnsortedSegmentSum's first input size " << x_shape.size()
                  << "is less equal than its second input size " << y_shape.size()
                  << ". CNode= " << origin_node->DebugString();
    return false;
  }
  return true;
}

bool UnsortedSegmentSumReplaceFission::IsNeedReplaced(const AnfNodePtr &node) const {
  // replace condition:
  MS_EXCEPTION_IF_NULL(node);
  auto origin_node = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(origin_node);
  auto x_shape = common::AnfAlgo::GetPrevNodeOutputInferShape(origin_node, 0);
  auto y_shape = common::AnfAlgo::GetPrevNodeOutputInferShape(origin_node, 1);
  auto out_shape = common::AnfAlgo::GetOutputInferShape(origin_node, 0);
  auto num_segments = common::AnfAlgo::GetNodeAttr<int64_t>(origin_node, kAttrNumSegments);

  if (std::any_of(UnsortedSegmentSumReplaceShapes.begin(), UnsortedSegmentSumReplaceShapes.end(),
                  [&](std::vector<std::vector<int64_t>> replace_shapes) {
                    return (replace_shapes[kInput0Index] == x_shape && replace_shapes[kInput1Index] == y_shape &&
                            replace_shapes[kInput2Index][0] == num_segments &&
                            replace_shapes[kOutput0Index] == out_shape);
                  })) {
    return true;
  }
  return false;
}

const BaseRef UnsortedSegmentSumReplaceFission::DefinePattern() const {
  VarPtr Xs = std::make_shared<SeqVar>();
  VectorRef pattern({prim::kPrimUnsortedSegmentSum, Xs});
  return pattern;
}

const AnfNodePtr UnsortedSegmentSumReplaceFission::ReplaceByUnsortedSegmentSumD(const FuncGraphPtr &graph,
                                                                                const AnfNodePtr &node) const {
  MS_EXCEPTION_IF_NULL(node);
  auto origin_node = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(origin_node);
  std::vector<AnfNodePtr> unsorted_segment_sum_d_inputs = {
    NewValueNode(std::make_shared<Primitive>(kUnsortedSegmentSumDOpName))};
  unsorted_segment_sum_d_inputs.push_back(origin_node->input(kIndex1));
  unsorted_segment_sum_d_inputs.push_back(origin_node->input(kIndex2));

  // num_segments is an attr for UnsortedSegmentSum, however, it is a input for UnsortedSegmentSumD
  // so convert attr to input after replacement
  auto num_segments = common::AnfAlgo::GetNodeAttr<int64_t>(origin_node, kAttrNumSegments);
  auto value_node = CreateNumSegmentsValueNode(num_segments);
  auto kernel_graph = graph->cast<KernelGraphPtr>();
  MS_EXCEPTION_IF_NULL(kernel_graph);
  kernel_graph->AddValueNodeToGraph(value_node);
  unsorted_segment_sum_d_inputs.push_back(value_node);

  // create new UnsortedSegmentSumD CNode
  auto unsorted_segment_sum_d = NewCNode(unsorted_segment_sum_d_inputs, graph);
  MS_EXCEPTION_IF_NULL(unsorted_segment_sum_d);
  unsorted_segment_sum_d->set_scope(origin_node->scope());
  unsorted_segment_sum_d->set_abstract(origin_node->abstract());
  return unsorted_segment_sum_d;
}

const AnfNodePtr UnsortedSegmentSumReplaceFission::Process(const FuncGraphPtr &graph, const AnfNodePtr &node,
                                                           const EquivPtr &) const {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(node);
  if (!CheckInputs(node)) {
    return nullptr;
  }
  if (IsNeedReplaced(node)) {
    return ReplaceByUnsortedSegmentSumD(graph, node);
  }
  return nullptr;
}
}  // namespace opt
}  // namespace mindspore
