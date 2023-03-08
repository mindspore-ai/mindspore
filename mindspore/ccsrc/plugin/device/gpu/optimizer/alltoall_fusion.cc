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

#include "plugin/device/gpu/optimizer/alltoall_fusion.h"

#include <vector>
#include <string>
#include <algorithm>

#include "include/backend/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "ir/primitive.h"
#include "include/common/utils/utils.h"
#include "plugin/device/gpu/hal/device/kernel_info_setter.h"
#include "distributed/init.h"
#ifndef _WIN32
#include "plugin/device/gpu/kernel/nccl/nccl_gpu_kernel.h"
#endif

namespace mindspore {
namespace opt {
namespace {
constexpr size_t kCNodePrimitiveIdx = 0;
constexpr size_t kAllToAllInputIdx = 1;

typedef std::vector<int> (*GetGroupRanks)(const std::string &);

inline int64_t NormalizeDim(const ShapeVector &shape, int64_t dim) {
  return dim < 0 ? SizeToLong(shape.size()) + dim : dim;
}

CNodePtr CreateSplitNode(const FuncGraphPtr &graph, const CNodePtr &all_to_all) {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(all_to_all);
  int64_t split_count = common::AnfAlgo::GetNodeAttr<int64_t>(all_to_all, kAttrSplitCount);
  int64_t split_dim = common::AnfAlgo::GetNodeAttr<int64_t>(all_to_all, kAttrSplitDim);
  if (all_to_all->size() <= kAllToAllInputIdx) {
    MS_LOG(EXCEPTION) << "Invalid cnode " << all_to_all->DebugString() << " input size " << all_to_all->size();
  }

  // Make a split CNode.
  auto all_to_all_input = all_to_all->input(kAllToAllInputIdx);
  std::vector<AnfNodePtr> split_input = {NewValueNode(std::make_shared<Primitive>(prim::kPrimSplit->name())),
                                         all_to_all_input};
  auto split = graph->NewCNode(split_input);
  MS_EXCEPTION_IF_NULL(split);

  // Judge validity of split_dim and shape
  auto dtype = common::AnfAlgo::GetOutputInferDataType(all_to_all_input, 0);
  auto shape = common::AnfAlgo::GetOutputInferShape(all_to_all_input, 0);
  split_dim = NormalizeDim(shape, split_dim);
  if (SizeToLong(shape.size()) <= split_dim) {
    MS_LOG(EXCEPTION) << "Invalid split dim " << split_dim << " is over the shape size " << shape.size();
  }
  if (split_count == 0 || shape[LongToSize(split_dim)] % split_count != 0) {
    MS_LOG(EXCEPTION) << "Invalid split count " << split_count << " cannot be divisible by shape[" << split_dim
                      << "] = " << shape[LongToSize(split_dim)];
  }
  shape[LongToSize(split_dim)] /= split_count;

  // Set Split CNode outputs type and shape, and CNode attributes.
  std::vector<TypeId> dtypes(split_count, dtype);
  std::vector<ShapeVector> shapes(split_count, shape);
  common::AnfAlgo::SetOutputInferTypeAndShape(dtypes, shapes, split.get());

  common::AnfAlgo::SetNodeAttr(kAttrAxis, MakeValue<int64_t>(split_dim), split);
  common::AnfAlgo::SetNodeAttr(kAttrOutputNum, MakeValue<int64_t>(split_count), split);
  return split;
}

CNodePtr CreateAllToAllvNode(const FuncGraphPtr &graph, const CNodePtr &all_to_all, const CNodePtr &split) {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(all_to_all);
  MS_EXCEPTION_IF_NULL(split);
  int64_t split_count = common::AnfAlgo::GetNodeAttr<int64_t>(all_to_all, kAttrSplitCount);
  std::string group = common::AnfAlgo::GetNodeAttr<std::string>(all_to_all, kAttrGroup);
  std::vector<AnfNodePtr> split_outputs;
  CreateMultipleOutputsOfAnfNode(graph, split, split_count, &split_outputs);
  if (split_outputs.empty()) {
    MS_LOG(EXCEPTION) << "The node " << split->DebugString() << " should have at least one output, but got 0.";
  }

  // Make a all_to_all_v CNode.
  std::vector<AnfNodePtr> all_to_all_v_input = {NewValueNode(std::make_shared<Primitive>(kAllToAllvOpName))};
  all_to_all_v_input.insert(all_to_all_v_input.end(), split_outputs.begin(), split_outputs.end());
  auto all_to_all_v = graph->NewCNode(all_to_all_v_input);
  MS_EXCEPTION_IF_NULL(all_to_all_v);

  // Prepare dtypes, shapes and ranks vectors.
  auto single_shape = AnfAlgo::GetOutputDetailShape(split_outputs[0], 0);
  auto single_type = common::AnfAlgo::GetOutputInferDataType(split_outputs[0], 0);
  std::vector<TypeId> dtypes(split_count, single_type);
  std::vector<BaseShapePtr> shapes(split_count, single_shape);
  common::AnfAlgo::SetOutputTypeAndDetailShape(dtypes, shapes, all_to_all_v.get());
  uint32_t rank_size = distributed::collective::CollectiveManager::instance()->GetGroupSize(group);
  std::vector<int64_t> rank_ids(rank_size, 0);
  for (uint32_t i = 0; i < rank_size; ++i) {
    rank_ids[i] = static_cast<int64_t>(i);
  }

  // Set AllToAllv CNode outputs and attributes.
  common::AnfAlgo::SetNodeAttr(kAttrSendRankIds, MakeValue<std::vector<int64_t>>(rank_ids), all_to_all_v);
  common::AnfAlgo::SetNodeAttr(kAttrRecvRankIds, MakeValue<std::vector<int64_t>>(rank_ids), all_to_all_v);
  common::AnfAlgo::SetNodeAttr(kAttrGroup, MakeValue<std::string>(group), all_to_all_v);
  MS_LOG(INFO) << "Create AllToAllv success, split count " << split_count << ", rank size " << rank_size;
  return all_to_all_v;
}

CNodePtr CreateConcatNode(const FuncGraphPtr &graph, const CNodePtr &all_to_all, const CNodePtr &all_to_all_v) {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(all_to_all);
  MS_EXCEPTION_IF_NULL(all_to_all_v);
  int64_t split_count = common::AnfAlgo::GetNodeAttr<int64_t>(all_to_all, kAttrSplitCount);
  int64_t concat_dim = common::AnfAlgo::GetNodeAttr<int64_t>(all_to_all, kAttrConcatDim);
  std::vector<AnfNodePtr> all_to_all_v_outputs;
  CreateMultipleOutputsOfAnfNode(graph, all_to_all_v, split_count, &all_to_all_v_outputs);
  if (all_to_all_v_outputs.empty()) {
    MS_LOG(EXCEPTION) << "The node " << all_to_all_v->DebugString() << " should have at least one output, but got 0.";
  }

  // Make a Concat CNode.
  std::vector<AnfNodePtr> concat_input = {NewValueNode(std::make_shared<Primitive>(kConcatOpName))};
  concat_input.insert(concat_input.end(), all_to_all_v_outputs.begin(), all_to_all_v_outputs.end());
  auto concat = graph->NewCNode(concat_input);
  MS_EXCEPTION_IF_NULL(concat);

  // Judge validity of concat_dim.
  auto single_shape = common::AnfAlgo::GetOutputInferShape(all_to_all_v_outputs[0], 0);
  concat_dim = NormalizeDim(single_shape, concat_dim);
  if (LongToSize(concat_dim) >= single_shape.size()) {
    MS_LOG(EXCEPTION) << "Invalid concat dim " << concat_dim << " is greater than shape size " << single_shape.size();
  }

  // Set Concat CNode outputs and  attributes.
  single_shape[LongToSize(concat_dim)] *= split_count;
  common::AnfAlgo::SetOutputInferTypeAndShape({common::AnfAlgo::GetOutputInferDataType(all_to_all_v_outputs[0], 0)},
                                              {single_shape}, concat.get());

  common::AnfAlgo::SetNodeAttr(kAttrAxis, MakeValue<int64_t>(concat_dim), concat);
  common::AnfAlgo::SetNodeAttr(kAttrInputNums, MakeValue(split_count), concat);
  std::vector<int64_t> dyn_input_size{split_count};
  common::AnfAlgo::SetNodeAttr(kAttrDynInputSizes, MakeValue(dyn_input_size), concat);
  return concat;
}
}  // namespace

const BaseRef AllToAllFusion::DefinePattern() const {
  return VectorRef({prim::kPrimAllToAll, std::make_shared<SeqVar>()});
}

const AnfNodePtr AllToAllFusion::Process(const FuncGraphPtr &graph, const AnfNodePtr &node, const EquivPtr &) const {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(node);
  auto all_to_all = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(all_to_all);

  // Step1: Split the AllToAll input Tensor into n_ranks parts along the AllToAll split_dim.
  auto split = CreateSplitNode(graph, all_to_all);
  // Step2: AllToAllv send and recv data to and from different rank.
  auto all_to_all_v = CreateAllToAllvNode(graph, all_to_all, split);
  // Step3: Concat all parts into one Tensor.
  auto concat = CreateConcatNode(graph, all_to_all, all_to_all_v);
  return concat;
}
}  // namespace opt
}  // namespace mindspore
