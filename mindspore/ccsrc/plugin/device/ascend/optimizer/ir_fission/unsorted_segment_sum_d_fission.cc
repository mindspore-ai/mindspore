/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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
#include "plugin/device/ascend/optimizer/ir_fission/unsorted_segment_sum_d_fission.h"
#include <memory>
#include <vector>
#include <algorithm>
#include "include/backend/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "ir/primitive.h"
#include "include/common/utils/utils.h"

namespace mindspore {
namespace opt {
namespace {
bool CheckInputs(const CNodePtr &origin_node) {
  MS_EXCEPTION_IF_NULL(origin_node);
  if (common::AnfAlgo::GetInputTensorNum(origin_node) != kUnsortedSegmentSumDInputTensorNum) {
    MS_LOG(DEBUG) << "UnsortedSegmentSumD has wrong inputs num, not equal " << kUnsortedSegmentSumDInputTensorNum
                  << ". CNode= " << origin_node->DebugString();
    return false;
  }
  auto x_shape = common::AnfAlgo::GetPrevNodeOutputInferShape(origin_node, 0);
  auto y_shape = common::AnfAlgo::GetPrevNodeOutputInferShape(origin_node, 1);
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

CNodePtr UnsortedSegmentSumDFission::CreatePadding(const FuncGraphPtr &graph, const CNodePtr &origin_node,
                                                   const size_t &pad_dim_size) const {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(origin_node);
  std::vector<AnfNodePtr> padding_inputs = {NewValueNode(std::make_shared<Primitive>(kPaddingOpName)),
                                            origin_node->input(kIndex1)};
  auto padding = NewCNode(padding_inputs, graph);
  MS_EXCEPTION_IF_NULL(padding);
  padding->set_scope(origin_node->scope());
  auto shape = common::AnfAlgo::GetPrevNodeOutputInferShape(origin_node, 0);
  shape[shape.size() - 1] = SizeToLong(pad_dim_size);
  common::AnfAlgo::SetOutputInferTypeAndShape({common::AnfAlgo::GetPrevNodeOutputInferDataType(origin_node, 0)},
                                              {shape}, padding.get());
  common::AnfAlgo::SetNodeAttr(kAttrPadDimSize, MakeValue(SizeToLong(pad_dim_size)), padding);
  return padding;
}

CNodePtr UnsortedSegmentSumDFission::CreateUnsortedSegmentSum(const FuncGraphPtr &graph, const CNodePtr &origin_node,
                                                              const CNodePtr &padding,
                                                              const size_t &pad_dim_size) const {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(origin_node);
  MS_EXCEPTION_IF_NULL(padding);
  std::vector<AnfNodePtr> unsorted_segment_sum8_inputs = {
    NewValueNode(std::make_shared<Primitive>(prim::kPrimUnsortedSegmentSumD->name())), padding,
    origin_node->input(kIndex2)};
  auto unsorted_segment_sum = NewCNode(unsorted_segment_sum8_inputs, graph);
  MS_EXCEPTION_IF_NULL(unsorted_segment_sum);
  unsorted_segment_sum->set_scope(origin_node->scope());
  auto shape = common::AnfAlgo::GetOutputInferShape(origin_node, 0);
  shape[shape.size() - 1] = SizeToLong(pad_dim_size);
  common::AnfAlgo::SetOutputInferTypeAndShape({common::AnfAlgo::GetOutputInferDataType(origin_node, 0)}, {shape},
                                              unsorted_segment_sum.get());

  common::AnfAlgo::SetNodeAttr(kAttrNumSegments, MakeValue(shape[0]), unsorted_segment_sum);
  if (common::AnfAlgo::HasNodeAttr(kAttrCustAicpu, origin_node)) {
    common::AnfAlgo::SetNodeAttr(kAttrCustAicpu, MakeValue(kUnsortedSegmentSumOpName), unsorted_segment_sum);
  }
  return unsorted_segment_sum;
}

CNodePtr UnsortedSegmentSumDFission::CreateSlice(const FuncGraphPtr &graph, const CNodePtr &unsort_segment_sum,
                                                 const CNodePtr &unsorted_segment_sum8) const {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(unsort_segment_sum);
  MS_EXCEPTION_IF_NULL(unsorted_segment_sum8);
  auto orig_sum_shape = common::AnfAlgo::GetOutputInferShape(unsort_segment_sum, 0);
  std::vector<int64_t> offsets(orig_sum_shape.size(), 0);
  auto offsets_input = CreateShapeValueNode(graph, offsets, true);
  auto size_input = CreateShapeValueNode(graph, orig_sum_shape, true);
  std::vector<AnfNodePtr> slice_inputs = {NewValueNode(std::make_shared<Primitive>(kSliceOpName)),
                                          unsorted_segment_sum8, offsets_input, size_input};
  auto slice = NewCNode(slice_inputs, graph);
  MS_EXCEPTION_IF_NULL(slice);
  slice->set_scope(unsort_segment_sum->scope());
  slice->set_abstract(unsort_segment_sum->abstract());
  return slice;
}

const BaseRef UnsortedSegmentSumDFission::DefinePattern() const {
  VarPtr Xs = std::make_shared<SeqVar>();
  VectorRef pattern({prim::kPrimUnsortedSegmentSumD, Xs});
  return pattern;
}

const AnfNodePtr UnsortedSegmentSumDFission::Process(const FuncGraphPtr &graph, const AnfNodePtr &node,
                                                     const EquivPtr &) const {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(node);
  auto origin_node = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(origin_node);
  if (!CheckInputs(origin_node)) {
    return nullptr;
  }
  size_t pad_dim_size;
  auto input_dtype = common::AnfAlgo::GetPrevNodeOutputInferDataType(origin_node, 0);
  constexpr auto PADSIZE32 = 8;
  constexpr auto PADSIZE16 = 16;
  if (input_dtype == kNumberTypeFloat32) {
    pad_dim_size = PADSIZE32;
  } else if (input_dtype == kNumberTypeFloat16) {
    pad_dim_size = PADSIZE16;
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
