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
#include "backend/optimizer/ascend/ir_fission/unsorted_segment_sum_fission.h"
#include <memory>
#include <vector>
#include "backend/session/anf_runtime_algorithm.h"
#include "ir/primitive.h"
#include "utils/utils.h"

namespace mindspore {
namespace opt {
namespace {
CNodePtr CreatePadding(const FuncGraphPtr &graph, const CNodePtr &origin_node, const size_t &pad_dim_size) {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(origin_node);
  std::vector<AnfNodePtr> padding_inputs = {NewValueNode(std::make_shared<Primitive>(kPaddingOpName)),
                                            origin_node->input(1)};
  auto padding = graph->NewCNode(padding_inputs);
  MS_EXCEPTION_IF_NULL(padding);
  padding->set_scope(origin_node->scope());
  auto shape = AnfAlgo::GetPrevNodeOutputInferShape(origin_node, 0);
  shape[shape.size() - 1] = pad_dim_size;
  AnfAlgo::SetOutputInferTypeAndShape({AnfAlgo::GetPrevNodeOutputInferDataType(origin_node, 0)}, {shape},
                                      padding.get());
  AnfAlgo::SetNodeAttr(kAttrPadDimSize, MakeValue(SizeToLong(pad_dim_size)), padding);
  return padding;
}

CNodePtr CreateUnsortedSegmentSum(const FuncGraphPtr &graph, const CNodePtr &origin_node, const CNodePtr &padding,
                                  const size_t &pad_dim_size) {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(origin_node);
  MS_EXCEPTION_IF_NULL(padding);
  std::vector<AnfNodePtr> unsorted_segment_sum8_inputs = {
    NewValueNode(std::make_shared<Primitive>(prim::kPrimUnsortedSegmentSum->name())), padding, origin_node->input(2)};
  auto unsorted_segment_sum = graph->NewCNode(unsorted_segment_sum8_inputs);
  MS_EXCEPTION_IF_NULL(unsorted_segment_sum);
  unsorted_segment_sum->set_scope(origin_node->scope());
  auto shape = AnfAlgo::GetOutputInferShape(origin_node, 0);
  shape[shape.size() - 1] = pad_dim_size;
  AnfAlgo::SetOutputInferTypeAndShape({AnfAlgo::GetOutputInferDataType(origin_node, 0)}, {shape},
                                      unsorted_segment_sum.get());
  AnfAlgo::SetNodeAttr(kAttrNumSegments, MakeValue(SizeToLong(shape[0])), unsorted_segment_sum);
  return unsorted_segment_sum;
}

CNodePtr CreateSlice(const FuncGraphPtr &graph, const CNodePtr &unsort_segment_sum,
                     const CNodePtr &unsorted_segment_sum8) {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(unsort_segment_sum);
  MS_EXCEPTION_IF_NULL(unsorted_segment_sum8);
  std::vector<AnfNodePtr> slice_inputs = {NewValueNode(std::make_shared<Primitive>(kSliceOpName)),
                                          unsorted_segment_sum8};
  auto slice = graph->NewCNode(slice_inputs);
  MS_EXCEPTION_IF_NULL(slice);
  slice->set_scope(unsort_segment_sum->scope());
  slice->set_abstract(unsort_segment_sum->abstract());
  auto unsort_segment_sum_shape = AnfAlgo::GetOutputInferShape(unsort_segment_sum, 0);
  std::vector<size_t> offsets(unsort_segment_sum_shape.size(), 0);
  AnfAlgo::SetNodeAttr(kAttrBegin, MakeValue(Convert2Long(offsets)), slice);
  AnfAlgo::SetNodeAttr(kAttrSize, MakeValue(Convert2Long(unsort_segment_sum_shape)), slice);
  return slice;
}

bool CheckInputs(const CNodePtr &origin_node) {
  MS_EXCEPTION_IF_NULL(origin_node);
  if (AnfAlgo::GetInputTensorNum(origin_node) != kUnsortedSegmentSumInputTensorNum) {
    MS_LOG(DEBUG) << "UnsortedSegmentSum has wrong inputs num, not equal " << kUnsortedSegmentSumInputTensorNum
                  << ". CNode= " << origin_node->DebugString();
    return false;
  }
  auto x_shape = AnfAlgo::GetPrevNodeOutputInferShape(origin_node, 0);
  auto y_shape = AnfAlgo::GetPrevNodeOutputInferShape(origin_node, 1);
  if (x_shape.empty() || y_shape.empty()) {
    return false;
  }
  if (x_shape[x_shape.size() - 1] != 1) {
    MS_LOG(DEBUG) << "UnsortedSegmentSum is not need fission. The last value of input0's shape is "
                  << x_shape[x_shape.size() - 1];
    return false;
  }
  return x_shape.size() > y_shape.size();
}
}  // namespace

const BaseRef UnsortSegmentSumFission::DefinePattern() const {
  VarPtr Xs = std::make_shared<SeqVar>();
  VectorRef pattern({prim::kPrimUnsortedSegmentSum, Xs});
  return pattern;
}

const AnfNodePtr UnsortSegmentSumFission::Process(const FuncGraphPtr &graph, const AnfNodePtr &node,
                                                  const EquivPtr &) const {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(node);
  auto origin_node = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(origin_node);
  if (!CheckInputs(origin_node)) {
    return nullptr;
  }
  size_t pad_dim_size;
  auto input_dtype = AnfAlgo::GetPrevNodeOutputInferDataType(origin_node, 0);
  if (input_dtype == kNumberTypeFloat32) {
    pad_dim_size = 8;
  } else if (input_dtype == kNumberTypeFloat16) {
    pad_dim_size = 16;
  } else {
    MS_LOG(DEBUG) << "UnsortedSegmentSum data type not in (float32, float16), no need change";
    return nullptr;
  }

  auto padding = CreatePadding(graph, origin_node, pad_dim_size);
  auto unsorted_segment_sum8 = CreateUnsortedSegmentSum(graph, origin_node, padding, pad_dim_size);
  return CreateSlice(graph, origin_node, unsorted_segment_sum8);
}
}  // namespace opt
}  // namespace mindspore
