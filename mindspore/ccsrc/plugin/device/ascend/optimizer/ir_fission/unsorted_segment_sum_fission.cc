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
#include "plugin/device/ascend/optimizer/ir_fission/unsorted_segment_sum_fission.h"
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
constexpr size_t kUnsortedSegmentSumInputNum = 3;
}  // namespace

CNodePtr UnsortedSegmentSumFission::CreateConcatD(const FuncGraphPtr &graph, const CNodePtr &sum,
                                                  const size_t &pad_dim_size) const {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(sum);
  std::vector<AnfNodePtr> concat_inputs = {NewValueNode(std::make_shared<Primitive>(kConcatDOpName))};
  auto x_input = sum->input(kIndex1);
  for (size_t i = 0; i < pad_dim_size; ++i) {
    concat_inputs.push_back(x_input);
  }
  auto concat = NewCNode(concat_inputs, graph);
  MS_EXCEPTION_IF_NULL(concat);
  concat->set_scope(sum->scope());
  auto shape = common::AnfAlgo::GetPrevNodeOutputInferShape(sum, 0);
  shape[shape.size() - 1] = SizeToLong(pad_dim_size);
  common::AnfAlgo::SetOutputInferTypeAndShape({common::AnfAlgo::GetPrevNodeOutputInferDataType(sum, 0)}, {shape},
                                              concat.get());
  common::AnfAlgo::SetNodeAttr(kAttrAxis, MakeValue(SizeToLong(shape.size() - 1)), concat);
  common::AnfAlgo::SetNodeAttr(kAttrDynInputSizes, MakeValue(std::vector<int64_t>{SizeToLong(pad_dim_size)}), concat);
  return concat;
}

CNodePtr UnsortedSegmentSumFission::CreateUnsortedSegmentSum(const FuncGraphPtr &graph, const CNodePtr &orig_sum,
                                                             const CNodePtr &concat, const size_t &pad_dim_size) const {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(orig_sum);
  MS_EXCEPTION_IF_NULL(concat);
  std::vector<AnfNodePtr> new_sum_inputs = {
    NewValueNode(std::make_shared<Primitive>(prim::kPrimUnsortedSegmentSum->name())), concat, orig_sum->input(kIndex2),
    orig_sum->input(kIndex3)};
  auto new_sum = NewCNode(new_sum_inputs, graph);
  MS_EXCEPTION_IF_NULL(new_sum);
  new_sum->set_scope(orig_sum->scope());
  auto shape = common::AnfAlgo::GetOutputInferShape(orig_sum, 0);
  shape[shape.size() - 1] = SizeToLong(pad_dim_size);
  common::AnfAlgo::SetOutputInferTypeAndShape({common::AnfAlgo::GetOutputInferDataType(orig_sum, 0)}, {shape},
                                              new_sum.get());
  if (common::AnfAlgo::HasNodeAttr(kAttrCustAicpu, orig_sum)) {
    common::AnfAlgo::SetNodeAttr(kAttrCustAicpu, MakeValue(kUnsortedSegmentSumOpName), new_sum);
  }
  SetInputOutputNames({"x", "segment_ids", "num_segments"}, {"y"}, new_sum);
  return new_sum;
}

CNodePtr UnsortedSegmentSumFission::CreateSlice(const FuncGraphPtr &graph, const CNodePtr &orig_sum,
                                                const CNodePtr &new_sum) const {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(orig_sum);
  MS_EXCEPTION_IF_NULL(new_sum);
  auto orig_sum_shape = common::AnfAlgo::GetOutputInferShape(orig_sum, 0);
  std::vector<int64_t> offsets(orig_sum_shape.size(), 0);
  auto offsets_input = CreateShapeValueNode(graph, offsets, true);
  auto size_input = CreateShapeValueNode(graph, orig_sum_shape, true);
  std::vector<AnfNodePtr> slice_inputs = {NewValueNode(std::make_shared<Primitive>(kSliceOpName)), new_sum,
                                          offsets_input, size_input};
  auto slice = NewCNode(slice_inputs, graph);
  MS_EXCEPTION_IF_NULL(slice);
  slice->set_scope(orig_sum->scope());
  slice->set_abstract(orig_sum->abstract());
  SetInputOutputNames({"x", "offsets", "size"}, {"y"}, slice);
  return slice;
}

const BaseRef UnsortedSegmentSumFission::DefinePattern() const {
  VarPtr Xs = std::make_shared<SeqVar>();
  VectorRef pattern({prim::kPrimUnsortedSegmentSum, Xs});
  return pattern;
}

const AnfNodePtr UnsortedSegmentSumFission::Process(const FuncGraphPtr &graph, const AnfNodePtr &node,
                                                    const EquivPtr &) const {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(node);
  auto sum = CheckAnfNodeIfCNodeAndInputSize(node, kUnsortedSegmentSumInputNum);
  auto x_shape = common::AnfAlgo::GetPrevNodeOutputInferShape(sum, 0);
  auto y_shape = common::AnfAlgo::GetPrevNodeOutputInferShape(sum, 1);
  if (x_shape.size() <= 1 || x_shape.back() != 1 || x_shape.size() <= y_shape.size()) {
    return nullptr;
  }
  size_t pad_dim_size;
  auto input_dtype = common::AnfAlgo::GetPrevNodeOutputInferDataType(sum, 0);
  constexpr auto PADSIZE32 = 8;
  constexpr auto PADSIZE16 = 16;
  if (input_dtype == kNumberTypeFloat32) {
    pad_dim_size = PADSIZE32;
  } else if (input_dtype == kNumberTypeFloat16) {
    pad_dim_size = PADSIZE16;
  } else {
    MS_LOG(DEBUG) << "UnsortedSegmentSum data type not in (float32, float16), no need change.";
    return nullptr;
  }

  auto concat = CreateConcatD(graph, sum, pad_dim_size);
  auto new_sum = CreateUnsortedSegmentSum(graph, sum, concat, pad_dim_size);
  return CreateSlice(graph, sum, new_sum);
}
}  // namespace opt
}  // namespace mindspore
