/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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

#include "plugin/device/ascend/optimizer/mindir/neighbor_exchange_v2_unify_mindir.h"
#include <algorithm>
#include <vector>
#include <string>
#include <utility>
#include "backend/common/session/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "include/common/utils/utils.h"
#include "plugin/device/ascend/hal/hccl_adapter/hccl_adapter.h"
#include "backend/common/optimizer/helper.h"
#include "utils/trace_base.h"
#include "frontend/parallel/ops_info/ops_utils.h"

namespace mindspore {
namespace opt {
namespace {
constexpr size_t kCNodePrimitiveIdx = 0;
constexpr size_t kNeighborExchangeV2InputIdx = 1;
constexpr size_t kLenTopIdx = 0;
constexpr size_t kLenBottomIdx = 1;
constexpr size_t kLenLeftIdx = 2;
constexpr size_t kLenRightIdx = 3;
constexpr size_t kHDim = 2;  // dim of H in NCHW
constexpr size_t kWDim = 3;  // dim of W in NCHW
constexpr int64_t kShapeSize = 4;
constexpr int64_t kRankIdZero = 0;
constexpr int64_t kRankIdOne = 1;
constexpr int64_t kRankIdTwo = 2;
constexpr int64_t kRankIdThree = 3;
constexpr int64_t kRankIdFour = 4;
constexpr int64_t kRankIdFive = 5;
constexpr int64_t kRankIdSix = 6;
constexpr int64_t kRankIdSeven = 7;
constexpr size_t kSizeFour = 4;
constexpr size_t kSizeEight = 8;
constexpr int64_t kInvalidId = -1;
constexpr size_t kMinSplitOutputSize = 2;

bool IsTop(const std::vector<int64_t> &send_rank_ids) {
  return send_rank_ids[kRankIdZero] != kInvalidId || send_rank_ids[kRankIdOne] != kInvalidId ||
         send_rank_ids[kRankIdSeven] != kInvalidId;
}

bool IsBottom(const std::vector<int64_t> &send_rank_ids) {
  return send_rank_ids[kRankIdThree] != kInvalidId || send_rank_ids[kRankIdFour] != kInvalidId ||
         send_rank_ids[kRankIdFive] != kInvalidId;
}

// cal split attrs size_splits, shapes and num_split
int64_t CalSplitAttrs(const ShapeVector &base_shape, const bool is_first, const bool is_last, const size_t split_dim,
                      const std::vector<int64_t> &send_lens, std::vector<int64_t> *size_splits,
                      std::vector<ShapeVector> *shapes) {
  MS_EXCEPTION_IF_NULL(size_splits);
  MS_EXCEPTION_IF_NULL(shapes);
  if (SizeToLong(base_shape.size()) != kShapeSize) {
    MS_LOG(EXCEPTION) << "Wrong base_shape size: " << base_shape.size() << ", it should be equal to 4.";
  }
  if (split_dim >= kShapeSize) {
    MS_LOG(EXCEPTION) << "Wrong split_dim: " << split_dim << ", it should less than 4.";
  }
  int64_t num_split = 0;
  int64_t split_middle_size = base_shape[split_dim];
  ShapeVector shape_tmp(base_shape);
  // [top, bottom, left, right]
  int64_t first_size = split_dim == kWDim ? send_lens[kDim2] : send_lens[0];
  int64_t last_size = split_dim == kWDim ? send_lens[kDim3] : send_lens[1];

  if (is_first) {
    // first
    ++num_split;
    size_splits->push_back(first_size);
    split_middle_size -= first_size;
    shape_tmp[split_dim] = first_size;
    shapes->push_back(shape_tmp);
  }
  if (is_last) {
    // middle
    split_middle_size -= last_size;
    if (split_middle_size > 0) {
      ++num_split;
      size_splits->push_back(split_middle_size);
      shape_tmp[split_dim] = split_middle_size;
      shapes->push_back(shape_tmp);
    }
    // last
    ++num_split;
    size_splits->push_back(last_size);
    shape_tmp[split_dim] = last_size;
    shapes->push_back(shape_tmp);
  } else if (split_middle_size > 0) {
    ++num_split;
    size_splits->push_back(split_middle_size);
    shape_tmp[split_dim] = split_middle_size;
    shapes->push_back(shape_tmp);
  }
  return num_split;
}

CNodePtr CreateSplitNode(const FuncGraphPtr &graph, const std::vector<AnfNodePtr> &split_input,
                         const ShapeVector &base_shape, bool is_first, bool is_last, size_t split_dim,
                         const std::vector<int64_t> &send_lens, TypeId input_dtype, int64_t *num_split,
                         const PatternProcessPass &pass) {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(num_split);
  if (split_input.empty()) {
    MS_LOG(EXCEPTION) << "The input is empty, can not create splitv node.";
  }
  auto split_v = pass.NewCNode(split_input, graph);
  MS_EXCEPTION_IF_NULL(split_v);
  std::vector<int64_t> size_splits = {};
  std::vector<ShapeVector> shapes = {};
  *num_split = CalSplitAttrs(base_shape, is_first, is_last, split_dim, send_lens, &size_splits, &shapes);

  std::vector<TypeId> dtypes(*num_split, input_dtype);
  common::AnfAlgo::SetOutputInferTypeAndShape(dtypes, shapes, split_v.get());
  common::AnfAlgo::SetNodeAttr(kAttrSplitDim, MakeValue<int64_t>(SizeToLong(split_dim)), split_v);
  common::AnfAlgo::SetNodeAttr(kAttrNumSplit, MakeValue<int64_t>(*num_split), split_v);
  common::AnfAlgo::SetNodeAttr(kAttrSizeSplits, MakeValue<std::vector<int64_t>>(size_splits), split_v);
  common::AnfAlgo::SetNodeAttr("is_backend_insert", MakeValue(true), split_v);
  return split_v;
}

std::vector<ShapeVector> CalAllToAllvOutputShape(const ShapeVector &base_shape, const std::vector<int64_t> &recv_lens,
                                                 const std::vector<int64_t> &recv_rank_ids) {
  if (SizeToLong(base_shape.size()) != kShapeSize) {
    MS_LOG(EXCEPTION) << "Wrong base_shape size: " << base_shape.size() << ", it should be equal to 4.";
  }
  std::vector<ShapeVector> shapes = {};
  std::vector<ShapeVector> ori_shapes = {
    {base_shape[0], base_shape[1], recv_lens[kLenTopIdx], base_shape[kWDim]},
    {base_shape[0], base_shape[1], recv_lens[kLenTopIdx], recv_lens[kLenRightIdx]},
    {base_shape[0], base_shape[1], base_shape[kHDim], recv_lens[kLenRightIdx]},
    {base_shape[0], base_shape[1], recv_lens[kLenBottomIdx], recv_lens[kLenRightIdx]},
    {base_shape[0], base_shape[1], recv_lens[kLenBottomIdx], base_shape[kWDim]},
    {base_shape[0], base_shape[1], recv_lens[kLenBottomIdx], recv_lens[kLenLeftIdx]},
    {base_shape[0], base_shape[1], base_shape[kHDim], recv_lens[kLenLeftIdx]},
    {base_shape[0], base_shape[1], recv_lens[kLenTopIdx], recv_lens[kLenLeftIdx]}};

  for (size_t idx = 0; idx < recv_rank_ids.size(); ++idx) {
    if (recv_rank_ids[idx] != kInvalidId) {
      shapes.push_back(ori_shapes[idx]);
    }
  }

  return shapes;
}

std::vector<AnfNodePtr> CreateAllToAllvInput(const std::vector<std::vector<AnfNodePtr>> &split_outputs,
                                             const std::vector<int64_t> &send_rank_ids) {
  std::vector<AnfNodePtr> all_to_all_v_input = {NewValueNode(std::make_shared<Primitive>(kAllToAllvOpName))};
  std::vector<size_t> split_idx = {0, 5, 3, 7, 1, 6, 2, 4};
  std::vector<bool> is_begin = {true, false, false, false, false, true, true, true};
  for (size_t idx = 0; idx < send_rank_ids.size(); ++idx) {
    if (send_rank_ids[idx] != kInvalidId) {
      if (is_begin[idx]) {
        (void)all_to_all_v_input.insert(all_to_all_v_input.end(), split_outputs[split_idx[idx]].begin(),
                                        split_outputs[split_idx[idx]].begin() + 1);
      } else {
        (void)all_to_all_v_input.insert(all_to_all_v_input.end(), split_outputs[split_idx[idx]].end() - 1,
                                        split_outputs[split_idx[idx]].end());
      }
    }
  }

  return all_to_all_v_input;
}

// get center of input for grad
AnfNodePtr GetCenter(const FuncGraphPtr &graph, const CNodePtr &neighbor_exchange_v2_grad,
                     const std::vector<CNodePtr> &split_nodes, const std::vector<int64_t> &split_num,
                     const std::vector<int64_t> &send_rank_ids) {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(neighbor_exchange_v2_grad);
  std::vector<AnfNodePtr> output;
  if (split_nodes[kRankIdTwo] == nullptr) {
    if (split_nodes[0] != nullptr) {
      CreateMultipleOutputsOfAnfNode(graph, split_nodes[0], static_cast<size_t>(split_num[0]), &output);
      if (output.size() < kMinSplitOutputSize) {
        MS_LOG(EXCEPTION) << "Wrong split output size: " << output.size() << ", except size >= 2.";
      }
      if (send_rank_ids[kRankIdZero] == kInvalidId) {
        return output[0];
      }
      return output[1];
    } else {
      return neighbor_exchange_v2_grad->input(kNeighborExchangeV2InputIdx);
    }
  } else {
    CreateMultipleOutputsOfAnfNode(graph, split_nodes[kDim2], static_cast<size_t>(split_num[kDim2]), &output);
    if (output.size() < kMinSplitOutputSize) {
      MS_LOG(EXCEPTION) << "Wrong split output size: " << output.size() << ", except size >= 2.";
    }
    if (send_rank_ids[kRankIdSix] == kInvalidId) {
      return output[0];
    }
    return output[1];
  }
}

std::vector<AnfNodePtr> CreateAllToAllvInputForGrad(const std::vector<int64_t> &send_rank_ids,
                                                    const std::vector<std::vector<AnfNodePtr>> &split_outputs,
                                                    const std::vector<CNodePtr> &split_nodes) {
  if (send_rank_ids.size() != kSizeEight) {
    MS_LOG(EXCEPTION) << "Wrong send_rank_ids size: " << send_rank_ids.size() << ", expect size: 8.";
  }
  if (split_outputs.size() != kSizeFour) {
    MS_LOG(EXCEPTION) << "Wrong split_outputs size: " << split_outputs.size() << ", expect size: 4.";
  }
  if (split_nodes.size() != kSizeFour) {
    MS_LOG(EXCEPTION) << "Wrong split_nodes size: " << split_nodes.size() << ", expect size: 4.";
  }
  std::vector<AnfNodePtr> all_to_all_v_input = {NewValueNode(std::make_shared<Primitive>(kAllToAllvOpName))};
  // only have top-bottom split
  std::vector<size_t> side_idx = {1, 2, 3, 5, 6, 7};
  bool no_send_side = std::all_of(side_idx.begin(), side_idx.end(),
                                  [&send_rank_ids](size_t idx) { return send_rank_ids[idx] == kInvalidId; });
  if (no_send_side) {
    if (send_rank_ids[kRankIdZero] != kInvalidId) {
      (void)all_to_all_v_input.insert(all_to_all_v_input.end(), split_outputs[0].begin(), split_outputs[0].begin() + 1);
    }
    if (send_rank_ids[kRankIdFour] != kInvalidId) {
      (void)all_to_all_v_input.insert(all_to_all_v_input.end(), split_outputs[0].end() - 1, split_outputs[0].end());
    }
    return all_to_all_v_input;
  }
  // 0, 1
  if (split_nodes[1] != nullptr) {
    if (send_rank_ids[kRankIdSeven] != kInvalidId) {
      (void)all_to_all_v_input.insert(all_to_all_v_input.end(), split_outputs[1].begin() + 1, split_outputs[1].end());
    } else {
      (void)all_to_all_v_input.insert(all_to_all_v_input.end(), split_outputs[1].begin(), split_outputs[1].end());
    }
  }
  // 2
  if (split_nodes[kIndex2] != nullptr && send_rank_ids[kRankIdTwo] != kInvalidId) {
    (void)all_to_all_v_input.insert(all_to_all_v_input.end(), split_outputs[kIndex2].end() - 1,
                                    split_outputs[kIndex2].end());
  }
  // 3, 4, 5
  if (split_nodes[kIndex3] != nullptr) {
    (void)all_to_all_v_input.insert(all_to_all_v_input.end(), split_outputs[kIndex3].rbegin(),
                                    split_outputs[kIndex3].rend());
  }
  // 6
  if (split_nodes[kIndex2] != nullptr && send_rank_ids[kRankIdSix] != kInvalidId) {
    (void)all_to_all_v_input.insert(all_to_all_v_input.end(), split_outputs[kIndex2].begin(),
                                    split_outputs[kIndex2].begin() + 1);
  }
  // 7
  if (split_nodes[1] != nullptr && send_rank_ids[kRankIdSeven] != kInvalidId) {
    (void)all_to_all_v_input.insert(all_to_all_v_input.end(), split_outputs[1].begin(), split_outputs[1].begin() + 1);
  }

  return all_to_all_v_input;
}

// alltoallv for forward & grad
CNodePtr CreateAllToAllvNode(const FuncGraphPtr &graph, const CNodePtr &neighbor_exchange_v2_or_grad,
                             const std::vector<CNodePtr> &split_nodes, const std::vector<int64_t> &split_num,
                             bool is_grad, const PatternProcessPass &pass) {
  MS_LOG(DEBUG) << "Start to create alltoallv node.";
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(neighbor_exchange_v2_or_grad);
  std::vector<int64_t> send_rank_ids =
    common::AnfAlgo::GetNodeAttr<std::vector<int64_t>>(neighbor_exchange_v2_or_grad, kAttrSendRankIds);
  std::vector<int64_t> recv_rank_ids =
    common::AnfAlgo::GetNodeAttr<std::vector<int64_t>>(neighbor_exchange_v2_or_grad, kAttrRecvRankIds);
  std::vector<int64_t> recv_lens =
    common::AnfAlgo::GetNodeAttr<std::vector<int64_t>>(neighbor_exchange_v2_or_grad, kAttrRecvLens);
  std::string group = common::AnfAlgo::GetNodeAttr<std::string>(neighbor_exchange_v2_or_grad, kAttrGroup);
  std::vector<uint32_t> group_rank_ids =
    common::AnfAlgo::HasNodeAttr(kAttrGroupRankIds, neighbor_exchange_v2_or_grad)
      ? common::AnfAlgo::GetNodeAttr<std::vector<uint32_t>>(neighbor_exchange_v2_or_grad, kAttrGroupRankIds)
      : std::vector<uint32_t>();
  // get split nodes output, split_outputs: [top_bottom, left_right, top_corner, bottom_corner]
  std::vector<std::vector<AnfNodePtr>> split_outputs;
  for (size_t i = 0; i < split_nodes.size(); ++i) {
    std::vector<AnfNodePtr> output;
    if (split_nodes[i] != nullptr) {
      CreateMultipleOutputsOfAnfNode(graph, split_nodes[i], static_cast<size_t>(split_num[i]), &output);
      if (output.empty()) {
        MS_LOG(EXCEPTION) << "The node " << split_nodes[i]->DebugString()
                          << " should have at least one output, but got 0." << trace::DumpSourceLines(split_nodes[i]);
      }
    }
    (void)split_outputs.emplace_back(output);
  }

  // all_to_all_v input
  std::vector<AnfNodePtr> all_to_all_v_input;
  AnfNodePtr base_node = nullptr;
  if (is_grad) {
    all_to_all_v_input = CreateAllToAllvInputForGrad(send_rank_ids, split_outputs, split_nodes);
    base_node = GetCenter(graph, neighbor_exchange_v2_or_grad, split_nodes, split_num, send_rank_ids);
  } else {
    all_to_all_v_input = CreateAllToAllvInput(split_outputs, send_rank_ids);
    base_node = neighbor_exchange_v2_or_grad->input(kNeighborExchangeV2InputIdx);
  }

  // for send empty depend
  int64_t all_to_all_input_num =
    std::count_if(send_rank_ids.begin(), send_rank_ids.end(), [](int64_t ids) { return ids != kInvalidId; });
  bool need_drop_input = false;
  if (all_to_all_input_num == 0) {
    all_to_all_v_input.emplace_back(neighbor_exchange_v2_or_grad->input(kNeighborExchangeV2InputIdx));
    need_drop_input = true;
  }

  // output shapes and dtypes
  auto base_dtype = common::AnfAlgo::GetOutputInferDataType(base_node, 0UL);
  auto base_shape = common::AnfAlgo::GetOutputInferShape(base_node, 0UL);
  if (SizeToLong(base_shape.size()) != kShapeSize) {
    MS_LOG(EXCEPTION) << "Invalid shape size " << base_shape.size() << ", only support NCHW input now!";
  }
  std::vector<ShapeVector> shapes = CalAllToAllvOutputShape(base_shape, recv_lens, recv_rank_ids);

  // erase -1 in send_rank_ids
  std::vector<int64_t> real_send_rank_ids(send_rank_ids.size());
  std::vector<int64_t> real_recv_rank_ids(recv_rank_ids.size());
  auto iter1 = std::copy_if(send_rank_ids.begin(), send_rank_ids.end(), real_send_rank_ids.begin(),
                            [](const int64_t item) { return item != kInvalidId; });
  auto iter2 = std::copy_if(recv_rank_ids.begin(), recv_rank_ids.end(), real_recv_rank_ids.begin(),
                            [](const int64_t item) { return item != kInvalidId; });
  real_send_rank_ids.resize(LongToSize(std::distance(real_send_rank_ids.begin(), iter1)));
  real_recv_rank_ids.resize(LongToSize(std::distance(real_recv_rank_ids.begin(), iter2)));

  std::vector<TypeId> dtypes(real_recv_rank_ids.size(), base_dtype);

  // create alltoallv node
  auto all_to_all_v = pass.NewCNode(all_to_all_v_input, graph);
  MS_EXCEPTION_IF_NULL(all_to_all_v);
  common::AnfAlgo::SetOutputInferTypeAndShape(dtypes, shapes, all_to_all_v.get());

  common::AnfAlgo::SetNodeAttr(kAttrSendRankIds, MakeValue<std::vector<int64_t>>(real_send_rank_ids), all_to_all_v);
  common::AnfAlgo::SetNodeAttr(kAttrRecvRankIds, MakeValue<std::vector<int64_t>>(real_recv_rank_ids), all_to_all_v);
  common::AnfAlgo::SetNodeAttr(kAttrRecvType, TypeIdToType(base_dtype), all_to_all_v);
  common::AnfAlgo::SetNodeAttr(kAttrGroup, MakeValue<std::string>(group), all_to_all_v);
  common::AnfAlgo::SetNodeAttr(kAttrGroupRankIds, MakeValue<std::vector<uint32_t>>(group_rank_ids), all_to_all_v);

  auto neighbor_exchange_v2_or_grad_prim = GetCNodePrimitive(neighbor_exchange_v2_or_grad);
  MS_EXCEPTION_IF_NULL(neighbor_exchange_v2_or_grad_prim);
  if (neighbor_exchange_v2_or_grad_prim->HasAttr(parallel::COMM_REUSE) &&
      GetValue<bool>(neighbor_exchange_v2_or_grad_prim->GetAttr(parallel::COMM_REUSE))) {
    auto all_to_all_v_prim = GetCNodePrimitive(all_to_all_v);
    MS_EXCEPTION_IF_NULL(all_to_all_v_prim);
    (void)all_to_all_v_prim->AddAttr(parallel::COMM_REUSE, MakeValue(true));
  }

  // add depend for input & alltoallv in send_empty condition
  common::AnfAlgo::SetNodeAttr(kAttrNeedDropInput, MakeValue<bool>(need_drop_input), all_to_all_v);
  if (all_to_all_input_num == 0) {
    auto input = neighbor_exchange_v2_or_grad->input(kNeighborExchangeV2InputIdx);
    std::vector<AnfNodePtr> depend_input = {NewValueNode(std::make_shared<Primitive>(prim::kPrimDepend->name())),
                                            all_to_all_v, input};
    auto depend = graph->NewCNode(depend_input);
    MS_EXCEPTION_IF_NULL(depend);
    depend->set_abstract(all_to_all_v->abstract());
    return depend;
  }
  MS_LOG(INFO) << "Create AllToAllv success, send rank size " << send_rank_ids.size() << ", recv rank size "
               << recv_rank_ids.size();
  return all_to_all_v;
}

int64_t AllToAllRealIds(size_t ids, const std::vector<int64_t> &recv_rank_ids) {
  int64_t real_ids = 0;
  for (size_t i = 0; i < ids; ++i) {
    if (recv_rank_ids[i] != kInvalidId) {
      ++real_ids;
    }
  }
  return real_ids;
}
}  // namespace

// return splits in 8 directions
std::vector<CNodePtr> NeighborExchangeV2UnifyMindIR::CreateSplitNodes(const FuncGraphPtr &graph,
                                                                      const CNodePtr &neighbor_exchange_v2,
                                                                      std::vector<int64_t> *split_num) const {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(neighbor_exchange_v2);
  MS_EXCEPTION_IF_NULL(split_num);
  std::vector<int64_t> send_rank_ids =
    common::AnfAlgo::GetNodeAttr<std::vector<int64_t>>(neighbor_exchange_v2, kAttrSendRankIds);
  std::vector<int64_t> send_lens =
    common::AnfAlgo::GetNodeAttr<std::vector<int64_t>>(neighbor_exchange_v2, kAttrSendLens);

  if (neighbor_exchange_v2->size() <= kNeighborExchangeV2InputIdx) {
    MS_LOG(EXCEPTION) << "Invalid cnode " << neighbor_exchange_v2->DebugString() << " input size "
                      << neighbor_exchange_v2->size() << ", should be " << kNeighborExchangeV2InputIdx
                      << trace::DumpSourceLines(neighbor_exchange_v2);
  }
  std::vector<CNodePtr> split_nodes = {};

  auto neighbor_exchange_v2_input = neighbor_exchange_v2->input(kNeighborExchangeV2InputIdx);

  auto dtype = common::AnfAlgo::GetOutputInferDataType(neighbor_exchange_v2_input, 0UL);
  auto shape = common::AnfAlgo::GetOutputInferShape(neighbor_exchange_v2_input, 0UL);
  if (SizeToLong(shape.size()) != kShapeSize) {  // only support NCHW now
    MS_LOG(EXCEPTION) << "Invalid shape size " << shape.size() << ", only support NCHW input now!"
                      << trace::DumpSourceLines(neighbor_exchange_v2);
  }

  // splitv for 0, 4, 6, 2
  bool is_top = IsTop(send_rank_ids);
  bool is_bottom = IsBottom(send_rank_ids);
  bool is_left = (send_rank_ids[kRankIdSix] != kInvalidId);
  bool is_right = (send_rank_ids[kRankIdTwo] != kInvalidId);
  std::vector<bool> splitvs_is_first = {true, false, true, false};  // is left or top
  std::vector<bool> splitvs_is_exist = {is_top, is_bottom, is_left, is_right};
  std::vector<size_t> splitvs_dim = {kHDim, kHDim, kWDim, kWDim};
  for (size_t i = 0; i < splitvs_is_first.size(); ++i) {
    int64_t num_split = 0;
    CNodePtr split_v = nullptr;
    if (splitvs_is_exist[i]) {
      std::vector<AnfNodePtr> split_input = {NewValueNode(std::make_shared<Primitive>(prim::kPrimSplitVD->name())),
                                             neighbor_exchange_v2_input};

      split_v = CreateSplitNode(graph, split_input, shape, splitvs_is_first[i], !splitvs_is_first[i], splitvs_dim[i],
                                send_lens, dtype, &num_split, *this);
    }
    (void)split_nodes.emplace_back(split_v);
    split_num->push_back(num_split);
  }

  // splitv for 7, 1, 5, 3
  std::vector<bool> corner_splitvs_is_first = {true, false, true, false};
  std::vector<bool> corner_splitvs_is_exist = {
    send_rank_ids[kRankIdSeven] != kInvalidId, send_rank_ids[kRankIdOne] != kInvalidId,
    send_rank_ids[kRankIdFive] != kInvalidId, send_rank_ids[kRankIdThree] != kInvalidId};
  std::vector<bool> corner_splitvs_is_input_top = {true, true, false, false};
  std::vector<AnfNodePtr> split_outputs_top;
  std::vector<AnfNodePtr> split_outputs_bottom;
  if (split_nodes[0] != nullptr) {
    CreateMultipleOutputsOfAnfNode(graph, split_nodes[0], static_cast<size_t>((*split_num)[0]), &split_outputs_top);
    if (split_outputs_top.empty()) {
      MS_LOG(EXCEPTION) << "The node " << split_nodes[0]->DebugString() << " should have at least one output, but got 0"
                        << trace::DumpSourceLines(split_nodes[0]);
    }
  }
  if (split_nodes[1] != nullptr) {
    CreateMultipleOutputsOfAnfNode(graph, split_nodes[1], static_cast<size_t>((*split_num)[1]), &split_outputs_bottom);
    if (split_outputs_bottom.empty()) {
      MS_LOG(EXCEPTION) << "The node " << split_nodes[1]->DebugString() << " should have at least one output, but got 0"
                        << trace::DumpSourceLines(split_nodes[1]);
    }
  }
  for (size_t i = 0; i < corner_splitvs_is_first.size(); ++i) {
    int64_t num_split = 0;
    CNodePtr split_v = nullptr;
    if (corner_splitvs_is_exist[i]) {
      auto shape_tmp = shape;
      std::vector<AnfNodePtr> split_input = {NewValueNode(std::make_shared<Primitive>(prim::kPrimSplitVD->name()))};
      if (corner_splitvs_is_input_top[i]) {
        (void)split_input.insert(split_input.end(), split_outputs_top.begin(), split_outputs_top.begin() + 1);
        shape_tmp[kHDim] = send_lens[0];
      } else {
        (void)split_input.insert(split_input.end(), split_outputs_bottom.end() - 1, split_outputs_bottom.end());
        shape_tmp[kHDim] = send_lens[1];
      }
      split_v = CreateSplitNode(graph, split_input, shape_tmp, corner_splitvs_is_first[i], !corner_splitvs_is_first[i],
                                kWDim, send_lens, dtype, &num_split, *this);
    }
    (void)split_nodes.emplace_back(split_v);
    split_num->push_back(num_split);
  }

  return split_nodes;
}

CNodePtr NeighborExchangeV2UnifyMindIR::CreateConcatNode(const FuncGraphPtr &graph,
                                                         const std::vector<AnfNodePtr> &concat_input, int64_t axis,
                                                         int64_t input_nums) const {
  MS_EXCEPTION_IF_NULL(graph);
  auto concat = NewCNode(concat_input, graph);
  MS_EXCEPTION_IF_NULL(concat);
  common::AnfAlgo::SetNodeAttr(kAttrAxis, MakeValue<int64_t>(axis), concat);
  common::AnfAlgo::SetNodeAttr(kAttrInputNums, MakeValue(input_nums), concat);
  std::vector<int64_t> dyn_input_size_empty{input_nums};
  common::AnfAlgo::SetNodeAttr(kAttrDynInputSizes, MakeValue(dyn_input_size_empty), concat);
  return concat;
}

CNodePtr NeighborExchangeV2UnifyMindIR::CreateLeftRightConcat(const FuncGraphPtr &graph,
                                                              const std::vector<AnfNodePtr> &all_to_all_v_outputs,
                                                              const std::vector<int64_t> &recv_rank_ids,
                                                              const std::vector<int64_t> &recv_lens,
                                                              bool is_left) const {
  MS_EXCEPTION_IF_NULL(graph);

  std::vector<AnfNodePtr> concat_input = {NewValueNode(std::make_shared<Primitive>(kConcatOpName))};
  int64_t input_num = 1;
  size_t first_ids = is_left ? kIndex7 : kIndex1;
  size_t middle_ids = is_left ? kIndex6 : kIndex2;
  size_t last_ids = is_left ? kIndex5 : kIndex3;
  auto output_idx = LongToSize(AllToAllRealIds(middle_ids, recv_rank_ids));

  auto single_shape = common::AnfAlgo::GetOutputInferShape(all_to_all_v_outputs[output_idx], 0UL);

  if (recv_rank_ids[first_ids] != kInvalidId) {
    ++input_num;
    single_shape[kDim2] += recv_lens[0];  // H in NCHW
  }
  if (recv_rank_ids[last_ids] != kInvalidId) {
    ++input_num;
    single_shape[kDim2] += recv_lens[1];  // H in NCHW
  }
  if (is_left) {
    (void)concat_input.insert(concat_input.end(), all_to_all_v_outputs.rbegin(),
                              all_to_all_v_outputs.rbegin() + input_num);
  } else {
    (void)concat_input.insert(concat_input.end(), all_to_all_v_outputs.begin() + AllToAllRealIds(1, recv_rank_ids),
                              all_to_all_v_outputs.begin() + input_num + AllToAllRealIds(1, recv_rank_ids));
  }

  std::vector<TypeId> concat_output_dtype = {common::AnfAlgo::GetOutputInferDataType(
    all_to_all_v_outputs[LongToSize(AllToAllRealIds(middle_ids, recv_rank_ids))], 0)};
  auto concat = CreateConcatNode(graph, concat_input, SizeToLong(kHDim), input_num);
  common::AnfAlgo::SetOutputInferTypeAndShape(concat_output_dtype, {single_shape}, concat.get());
  return concat;
}

CNodePtr NeighborExchangeV2UnifyMindIR::CreateMiddleConcat(
  const FuncGraphPtr &graph, const CNodePtr &neighbor_exchange_v2, const std::vector<AnfNodePtr> &all_to_all_v_outputs,
  const std::vector<int64_t> &recv_rank_ids, const std::vector<int64_t> &recv_lens, size_t concat_dim) const {
  std::vector<AnfNodePtr> concat_input_all = {NewValueNode(std::make_shared<Primitive>(kConcatOpName))};
  int64_t input_num_all = 0;
  auto neighbor_exchange_v2_input = neighbor_exchange_v2->input(kNeighborExchangeV2InputIdx);
  auto single_shape = common::AnfAlgo::GetOutputInferShape(neighbor_exchange_v2_input, 0UL);
  size_t first_idx = kIndex0;
  size_t last_idx = kIndex4;
  int64_t first_len = recv_lens[0];
  int64_t last_len = recv_lens[1];
  if (concat_dim == kWDim) {
    first_idx = kIndex6;
    last_idx = kIndex2;
    first_len = recv_lens[kDim2];
    last_len = recv_lens[kDim3];
  }

  // left
  if (recv_rank_ids[first_idx] != kInvalidId) {
    if (concat_dim == kWDim) {
      (void)concat_input_all.insert(concat_input_all.end(), all_to_all_v_outputs.end() - 1, all_to_all_v_outputs.end());
    } else {
      (void)concat_input_all.insert(concat_input_all.end(), all_to_all_v_outputs.begin(),
                                    all_to_all_v_outputs.begin() + 1);
    }

    ++input_num_all;
    single_shape[concat_dim] += first_len;
  }

  concat_input_all.push_back(neighbor_exchange_v2_input);
  ++input_num_all;
  // right
  if (recv_rank_ids[last_idx] != kInvalidId) {
    if (concat_dim == kWDim) {
      (void)concat_input_all.insert(concat_input_all.end(), all_to_all_v_outputs.begin(),
                                    all_to_all_v_outputs.begin() + 1);
    } else {
      int64_t bottom_num = AllToAllRealIds(kRankIdFour, recv_rank_ids);
      (void)concat_input_all.insert(concat_input_all.end(), all_to_all_v_outputs.begin() + bottom_num,
                                    all_to_all_v_outputs.begin() + bottom_num + 1);
    }

    ++input_num_all;
    single_shape[concat_dim] += last_len;
  }

  std::vector<TypeId> concat_output_dtype = {common::AnfAlgo::GetOutputInferDataType(all_to_all_v_outputs[0], 0UL)};
  auto concat_all = CreateConcatNode(graph, concat_input_all, SizeToLong(concat_dim), input_num_all);
  common::AnfAlgo::SetOutputInferTypeAndShape(concat_output_dtype, {single_shape}, concat_all.get());
  return concat_all;
}

CNodePtr NeighborExchangeV2UnifyMindIR::AllToAllvRecvEmpty(const FuncGraphPtr &graph,
                                                           const CNodePtr &neighbor_exchange_v2,
                                                           const CNodePtr &all_to_all_v) const {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(neighbor_exchange_v2);
  MS_EXCEPTION_IF_NULL(all_to_all_v);
  // add depend for input & alltoallv
  auto neighbor_exchange_v2_input = neighbor_exchange_v2->input(kNeighborExchangeV2InputIdx);
  std::vector<AnfNodePtr> depend_input = {NewValueNode(std::make_shared<Primitive>(prim::kPrimDepend->name())),
                                          neighbor_exchange_v2_input, all_to_all_v};
  auto depend = graph->NewCNode(depend_input);
  MS_EXCEPTION_IF_NULL(depend);
  depend->set_abstract(neighbor_exchange_v2_input->abstract());
  return depend;
}

CNodePtr NeighborExchangeV2UnifyMindIR::CreateConcatNodes(const FuncGraphPtr &graph,
                                                          const CNodePtr &neighbor_exchange_v2,
                                                          const CNodePtr &all_to_all_v) const {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(neighbor_exchange_v2);
  MS_EXCEPTION_IF_NULL(all_to_all_v);
  std::vector<int64_t> recv_rank_ids =
    common::AnfAlgo::GetNodeAttr<std::vector<int64_t>>(neighbor_exchange_v2, kAttrRecvRankIds);
  std::vector<int64_t> recv_lens =
    common::AnfAlgo::GetNodeAttr<std::vector<int64_t>>(neighbor_exchange_v2, kAttrRecvLens);

  int64_t all_to_all_output_num =
    std::count_if(recv_rank_ids.begin(), recv_rank_ids.end(), [](int64_t ids) { return ids != kInvalidId; });
  if (all_to_all_output_num == 0) {
    return AllToAllvRecvEmpty(graph, neighbor_exchange_v2, all_to_all_v);
  }

  std::vector<AnfNodePtr> all_to_all_v_outputs;
  CreateMultipleOutputsOfAnfNode(graph, all_to_all_v, static_cast<size_t>(all_to_all_output_num),
                                 &all_to_all_v_outputs);
  if (all_to_all_v_outputs.empty()) {
    MS_LOG(EXCEPTION) << "The node " << all_to_all_v->DebugString() << " should have at least one output, but got 0."
                      << trace::DumpSourceLines(all_to_all_v);
  }

  if (recv_rank_ids[kRankIdZero] == kInvalidId && recv_rank_ids[kRankIdFour] == kInvalidId) {
    return CreateMiddleConcat(graph, neighbor_exchange_v2, all_to_all_v_outputs, recv_rank_ids, recv_lens, kWDim);
  }

  // top or bottom
  // middle concat
  auto concat_middle =
    CreateMiddleConcat(graph, neighbor_exchange_v2, all_to_all_v_outputs, recv_rank_ids, recv_lens, kHDim);

  bool is_left = recv_rank_ids[kRankIdSix] != kInvalidId || recv_rank_ids[kRankIdFive] != kInvalidId ||
                 recv_rank_ids[kRankIdSeven] != kInvalidId;
  bool is_right = recv_rank_ids[kRankIdOne] != kInvalidId || recv_rank_ids[kRankIdTwo] != kInvalidId ||
                  recv_rank_ids[kRankIdThree] != kInvalidId;
  if (!is_left && !is_right) {
    return concat_middle;
  }

  std::vector<AnfNodePtr> concat_input_all = {NewValueNode(std::make_shared<Primitive>(kConcatOpName))};
  auto neighbor_exchange_v2_input = neighbor_exchange_v2->input(kNeighborExchangeV2InputIdx);
  auto shape_all = common::AnfAlgo::GetOutputInferShape(neighbor_exchange_v2_input, 0UL);
  shape_all[kDim2] = recv_rank_ids[kRankIdZero] != kInvalidId ? shape_all[kDim2] + recv_lens[0] : shape_all[kDim2];
  shape_all[kDim2] = recv_rank_ids[kRankIdFour] != kInvalidId ? shape_all[kDim2] + recv_lens[1] : shape_all[kDim2];
  int64_t input_nums_all = 0;
  // left concat
  if (is_left) {
    auto concat_left = CreateLeftRightConcat(graph, all_to_all_v_outputs, recv_rank_ids, recv_lens, true);

    // connect to concat_all
    std::vector<AnfNodePtr> concat_left_outputs;
    CreateMultipleOutputsOfAnfNode(graph, concat_left, 1UL, &concat_left_outputs);
    if (concat_left_outputs.empty()) {
      MS_LOG(EXCEPTION) << "The node " << concat_left->DebugString() << " should have at least one output, but got 0."
                        << trace::DumpSourceLines(concat_left);
    }
    (void)concat_input_all.insert(concat_input_all.end(), concat_left_outputs.begin(), concat_left_outputs.end());
    ++input_nums_all;
    shape_all[kDim3] += recv_lens[kDim2];
  }

  // middle concat connect to concat_all
  std::vector<AnfNodePtr> concat_middle_outputs;
  CreateMultipleOutputsOfAnfNode(graph, concat_middle, 1UL, &concat_middle_outputs);
  if (concat_middle_outputs.empty()) {
    MS_LOG(EXCEPTION) << "The node " << concat_middle->DebugString() << " should have at least one output, but got 0."
                      << trace::DumpSourceLines(concat_middle);
  }
  (void)concat_input_all.insert(concat_input_all.end(), concat_middle_outputs.begin(), concat_middle_outputs.end());
  ++input_nums_all;

  if (is_right) {
    auto concat_right = CreateLeftRightConcat(graph, all_to_all_v_outputs, recv_rank_ids, recv_lens, false);

    // connect to concat_all
    std::vector<AnfNodePtr> concat_right_outputs;
    CreateMultipleOutputsOfAnfNode(graph, concat_right, 1UL, &concat_right_outputs);
    if (concat_right_outputs.empty()) {
      MS_LOG(EXCEPTION) << "The node " << concat_right->DebugString() << " should have at least one output, but got 0."
                        << trace::DumpSourceLines(concat_right);
    }
    (void)concat_input_all.insert(concat_input_all.end(), concat_right_outputs.begin(), concat_right_outputs.end());
    ++input_nums_all;
    shape_all[kDim3] += recv_lens[kDim3];
  }

  std::vector<TypeId> concat_right_output_dtype = {common::AnfAlgo::GetOutputInferDataType(concat_input_all[1], 0)};
  auto concat_all = CreateConcatNode(graph, concat_input_all, static_cast<int64_t>(kWDim), input_nums_all);
  common::AnfAlgo::SetOutputInferTypeAndShape(concat_right_output_dtype, {shape_all}, concat_all.get());
  return concat_all;
}

// splits for grad, returns {top_bottom, left_right, top_corner, bottom_corner}, if no split, set it nullptr
std::vector<CNodePtr> NeighborExchangeV2GradUnifyMindIR::CreateSplitNodesForGrad(
  const FuncGraphPtr &graph, const CNodePtr &neighbor_exchange_v2_grad, std::vector<int64_t> *split_num) const {
  MS_LOG(DEBUG) << "Start create splitv nodes.";
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(neighbor_exchange_v2_grad);
  MS_EXCEPTION_IF_NULL(split_num);
  std::vector<int64_t> send_rank_ids =
    common::AnfAlgo::GetNodeAttr<std::vector<int64_t>>(neighbor_exchange_v2_grad, kAttrSendRankIds);
  std::vector<int64_t> send_lens =
    common::AnfAlgo::GetNodeAttr<std::vector<int64_t>>(neighbor_exchange_v2_grad, kAttrSendLens);

  if (neighbor_exchange_v2_grad->size() <= kNeighborExchangeV2InputIdx) {
    MS_LOG(EXCEPTION) << "Invalid cnode " << neighbor_exchange_v2_grad->DebugString() << " input size "
                      << neighbor_exchange_v2_grad->size() << ", should be " << kNeighborExchangeV2InputIdx
                      << trace::DumpSourceLines(neighbor_exchange_v2_grad);
  }

  auto neighbor_exchange_v2_grad_input = neighbor_exchange_v2_grad->input(kNeighborExchangeV2InputIdx);
  auto dtype = common::AnfAlgo::GetOutputInferDataType(neighbor_exchange_v2_grad_input, 0);
  auto shape = common::AnfAlgo::GetOutputInferShape(neighbor_exchange_v2_grad_input, 0);
  if (SizeToLong(shape.size()) != kShapeSize) {
    MS_LOG(EXCEPTION) << "Invalid shape size " << shape.size() << ", only support NCHW input now!"
                      << trace::DumpSourceLines(neighbor_exchange_v2_grad);
  }

  std::vector<CNodePtr> split_nodes = {};
  // splitv for top & bottom
  bool is_top = IsTop(send_rank_ids);
  bool is_bottom = IsBottom(send_rank_ids);
  CNodePtr split_v_top_bottom = nullptr;
  int64_t num_split_h = 0;
  if (is_top || is_bottom) {
    std::vector<AnfNodePtr> split_input = {NewValueNode(std::make_shared<Primitive>(prim::kPrimSplitVD->name())),
                                           neighbor_exchange_v2_grad_input};
    split_v_top_bottom =
      CreateSplitNode(graph, split_input, shape, is_top, is_bottom, kHDim, send_lens, dtype, &num_split_h, *this);
  }
  (void)split_nodes.emplace_back(split_v_top_bottom);
  split_num->push_back(num_split_h);

  // splitvs for left & right
  // inputs
  std::vector<AnfNodePtr> split_outputs_top_bottom;
  std::vector<int64_t> size_split_h;
  if (split_nodes[0] != nullptr) {
    CreateMultipleOutputsOfAnfNode(graph, split_nodes[0], static_cast<size_t>(num_split_h), &split_outputs_top_bottom);
    if (split_outputs_top_bottom.empty()) {
      MS_LOG(EXCEPTION) << "The node " << split_nodes[0]->DebugString()
                        << " should have at least one output, but got 0." << trace::DumpSourceLines(split_nodes[0]);
    }
    size_split_h = common::AnfAlgo::GetNodeAttr<std::vector<int64_t>>(split_nodes[0], kAttrSizeSplits);
  } else {
    // just middle
    split_outputs_top_bottom.push_back(neighbor_exchange_v2_grad_input);
    size_split_h.push_back(shape[kHDim]);
  }

  // left_right splitv nodes from top to bottom
  bool is_left = (send_rank_ids[kRankIdFive] != kInvalidId) || (send_rank_ids[kRankIdSix] != kInvalidId) ||
                 (send_rank_ids[kRankIdSeven] != kInvalidId);
  bool is_right = (send_rank_ids[kRankIdOne] != kInvalidId) || (send_rank_ids[kRankIdTwo] != kInvalidId) ||
                  (send_rank_ids[kRankIdThree] != kInvalidId);
  if (is_left || is_right) {
    if (!is_top) {
      split_nodes.emplace_back(nullptr);
      split_num->push_back(0);
    }
    for (size_t i = 0; i < split_outputs_top_bottom.size(); ++i) {
      std::vector<AnfNodePtr> split_input = {NewValueNode(std::make_shared<Primitive>(prim::kPrimSplitVD->name())),
                                             split_outputs_top_bottom[i]};

      int64_t num_split_w = 0;
      ShapeVector base_shape(shape);
      base_shape[kHDim] = size_split_h[i];
      auto split_v_left_right = CreateSplitNode(graph, split_input, base_shape, is_left, is_right, kWDim, send_lens,
                                                dtype, &num_split_w, *this);
      (void)split_nodes.emplace_back(split_v_left_right);
      split_num->push_back(num_split_w);
    }
    if (!is_bottom) {
      split_nodes.emplace_back(nullptr);
      split_num->push_back(0);
    }
  } else {
    split_nodes.emplace_back(nullptr);
    split_num->push_back(0);
    split_nodes.emplace_back(nullptr);
    split_num->push_back(0);
    split_nodes.emplace_back(nullptr);
    split_num->push_back(0);
  }
  MS_LOG(DEBUG) << "Create splitv nodes success.";
  return split_nodes;
}

CNodePtr NeighborExchangeV2GradUnifyMindIR::CreatePadNode(const FuncGraphPtr &graph, const AnfNodePtr &input,
                                                          const std::vector<int64_t> &begin,
                                                          const std::vector<int64_t> &size,
                                                          const std::pair<ShapeVector, BaseShapePtr> &shape_info,
                                                          TypeId dtype) const {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(input);
  auto shape = shape_info.first;
  auto shape_base = shape_info.second;
  MS_EXCEPTION_IF_NULL(shape_base);
  std::vector<AnfNodePtr> pad_inputs = {NewValueNode(std::make_shared<Primitive>(kPadDOpName)), input};
  auto pad = NewCNode(pad_inputs, graph);
  std::vector<std::vector<int64_t>> paddings;
  for (size_t i = 0; i < shape.size(); ++i) {
    (void)paddings.emplace_back(std::vector<int64_t>{begin[i], (shape[i] - begin[i]) - size[i]});
  }
  common::AnfAlgo::SetOutputTypeAndDetailShape({dtype}, {shape_base}, pad.get());
  common::AnfAlgo::SetNodeAttr(kAttrPaddings, MakeValue(paddings), pad);
  common::AnfAlgo::SetNodeAttr(kAttrInputNames, MakeValue(std::vector<std::string>{"x"}), pad);
  return pad;
}

CNodePtr NeighborExchangeV2GradUnifyMindIR::CreateSplitGradNodes(const FuncGraphPtr &graph,
                                                                 const CNodePtr &neighbor_exchange_v2_grad,
                                                                 const CNodePtr &all_to_all_v,
                                                                 const std::vector<CNodePtr> &split_nodes,
                                                                 const std::vector<int64_t> &split_num) const {
  MS_LOG(DEBUG) << "Start create splitvs grad nodes.";
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(neighbor_exchange_v2_grad);
  std::vector<int64_t> send_rank_ids =
    common::AnfAlgo::GetNodeAttr<std::vector<int64_t>>(neighbor_exchange_v2_grad, kAttrSendRankIds);
  std::vector<int64_t> recv_rank_ids =
    common::AnfAlgo::GetNodeAttr<std::vector<int64_t>>(neighbor_exchange_v2_grad, kAttrRecvRankIds);
  std::vector<int64_t> recv_lens =
    common::AnfAlgo::GetNodeAttr<std::vector<int64_t>>(neighbor_exchange_v2_grad, kAttrRecvLens);

  auto centerx = GetCenter(graph, neighbor_exchange_v2_grad, split_nodes, split_num, send_rank_ids);
  auto centerx_dtype = common::AnfAlgo::GetOutputInferDataType(centerx, 0UL);
  auto centerx_shape = common::AnfAlgo::GetOutputInferShape(centerx, 0UL);
  auto base_shape = common::AnfAlgo::GetOutputDetailShape(centerx, 0UL);
  // empty
  int64_t all_to_all_output_num =
    std::count_if(recv_rank_ids.begin(), recv_rank_ids.end(), [](int64_t ids) { return ids != kInvalidId; });
  if (all_to_all_output_num == 0) {
    // add depend(alltoallv, centerx)
    std::vector<AnfNodePtr> depend_input = {NewValueNode(std::make_shared<Primitive>(prim::kPrimDepend->name())),
                                            centerx, all_to_all_v};
    auto depend = graph->NewCNode(depend_input);
    MS_EXCEPTION_IF_NULL(depend);
    depend->set_abstract(centerx->abstract());
    return depend;
  }
  // get alltoallv outputs
  std::vector<AnfNodePtr> all_to_all_v_outputs;
  CreateMultipleOutputsOfAnfNode(graph, all_to_all_v, static_cast<size_t>(all_to_all_output_num),
                                 &all_to_all_v_outputs);
  if (all_to_all_v_outputs.empty()) {
    MS_LOG(EXCEPTION) << "The node " << all_to_all_v->DebugString() << " should have at least one output, but got 0."
                      << trace::DumpSourceLines(all_to_all_v);
  }
  // create pad nodes
  // slice begin & size
  std::vector<std::vector<int64_t>> begins = {
    {0, 0, 0, 0},
    {0, 0, 0, centerx_shape[kDim3] - recv_lens[kDim3]},
    {0, 0, 0, centerx_shape[kDim3] - recv_lens[kDim3]},
    {0, 0, centerx_shape[kDim2] - recv_lens[kDim1], centerx_shape[kDim3] - recv_lens[kDim3]},
    {0, 0, centerx_shape[kDim2] - recv_lens[kDim1], 0},
    {0, 0, centerx_shape[kDim2] - recv_lens[kDim1], 0},
    {0, 0, 0, 0},
    {0, 0, 0, 0}};
  std::vector<std::vector<int64_t>> sizes = {
    {centerx_shape[0], centerx_shape[1], recv_lens[0], centerx_shape[kDim3]},
    {centerx_shape[0], centerx_shape[1], recv_lens[0], recv_lens[kDim3]},
    {centerx_shape[0], centerx_shape[1], centerx_shape[kDim2], recv_lens[kDim3]},
    {centerx_shape[0], centerx_shape[1], recv_lens[1], recv_lens[kDim3]},
    {centerx_shape[0], centerx_shape[1], recv_lens[1], centerx_shape[kDim3]},
    {centerx_shape[0], centerx_shape[1], recv_lens[1], recv_lens[kDim2]},
    {centerx_shape[0], centerx_shape[1], centerx_shape[kDim2], recv_lens[kDim2]},
    {centerx_shape[0], centerx_shape[1], recv_lens[0], recv_lens[kDim2]}};
  std::vector<CNodePtr> pad_nodes;
  size_t output_index = 0;
  for (size_t i = 0; i < recv_rank_ids.size(); ++i) {
    if (recv_rank_ids[i] != kInvalidId) {
      auto shape_info = std::make_pair(centerx_shape, base_shape);
      auto pad =
        CreatePadNode(graph, all_to_all_v_outputs[output_index], begins[i], sizes[i], shape_info, centerx_dtype);
      ++output_index;
      (void)pad_nodes.emplace_back(pad);
    }
  }

  // create add node
  std::vector<AnfNodePtr> addn_inputs = {NewValueNode(std::make_shared<Primitive>(kAddNOpName)), centerx};
  int64_t pad_num = 1;
  for (auto pad : pad_nodes) {
    std::vector<AnfNodePtr> pad_outputs;
    CreateMultipleOutputsOfAnfNode(graph, pad, 1, &pad_outputs);
    if (pad_outputs.empty()) {
      MS_LOG(EXCEPTION) << "The node " << pad->DebugString() << " should have at least one output, but got 0."
                        << trace::DumpSourceLines(pad);
    }
    (void)addn_inputs.insert(addn_inputs.end(), pad_outputs.begin(), pad_outputs.end());
    ++pad_num;
  }
  auto addn = NewCNode(addn_inputs, graph);
  MS_EXCEPTION_IF_NULL(addn);
  common::AnfAlgo::SetOutputTypeAndDetailShape({centerx_dtype}, {base_shape}, addn.get());
  common::AnfAlgo::SetNodeAttr(kAttrDynInputSizes, MakeValue<std::vector<int64_t>>({pad_num}), addn);
  common::AnfAlgo::SetNodeAttr(kAttrN, MakeValue(pad_num), addn);
  MS_LOG(DEBUG) << "Create splitvs grad nodes success.";
  return addn;
}

std::vector<std::string> NeighborExchangeV2UnifyMindIR::MustExistPrimitiveName() const {
  std::vector<std::string> ret;
  ret.emplace_back(prim::kPrimNeighborExchangeV2->name());
  return ret;
}

const BaseRef NeighborExchangeV2UnifyMindIR::DefinePattern() const {
  return VectorRef({prim::kPrimNeighborExchangeV2, std::make_shared<SeqVar>()});
}

const AnfNodePtr NeighborExchangeV2UnifyMindIR::Process(const FuncGraphPtr &graph, const AnfNodePtr &node,
                                                        const EquivPtr &) const {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(node);
  auto neighbor_exchange_v2 = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(neighbor_exchange_v2);
  std::vector<int64_t> split_num;
  auto split_nodes = CreateSplitNodes(graph, neighbor_exchange_v2, &split_num);
  auto all_to_all_v = CreateAllToAllvNode(graph, neighbor_exchange_v2, split_nodes, split_num, false, *this);
  auto concat = CreateConcatNodes(graph, neighbor_exchange_v2, all_to_all_v);
  return concat;
}

std::vector<std::string> NeighborExchangeV2GradUnifyMindIR::MustExistPrimitiveName() const {
  std::vector<std::string> ret;
  ret.emplace_back(prim::kPrimNeighborExchangeV2Grad->name());
  return ret;
}

const BaseRef NeighborExchangeV2GradUnifyMindIR::DefinePattern() const {
  return VectorRef({prim::kPrimNeighborExchangeV2Grad, std::make_shared<SeqVar>()});
}

const AnfNodePtr NeighborExchangeV2GradUnifyMindIR::Process(const FuncGraphPtr &graph, const AnfNodePtr &node,
                                                            const EquivPtr &) const {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(node);
  auto neighbor_exchange_v2_grad = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(neighbor_exchange_v2_grad);
  std::vector<int64_t> split_num;
  auto split_nodes = CreateSplitNodesForGrad(graph, neighbor_exchange_v2_grad, &split_num);
  auto all_to_all_v = CreateAllToAllvNode(graph, neighbor_exchange_v2_grad, split_nodes, split_num, true, *this);
  auto add = CreateSplitGradNodes(graph, neighbor_exchange_v2_grad, all_to_all_v, split_nodes, split_num);
  return add;
}
}  // namespace opt
}  // namespace mindspore
