/**
 * Copyright 2021-2024 Huawei Technologies Co., Ltd
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

#include "frontend/parallel/ops_info/resizebilinear_v2_info.h"

#include <algorithm>
#include <memory>
#include <utility>
#include <vector>
#include <cmath>

#include "frontend/parallel/device_matrix.h"
#include "frontend/parallel/dynamic_creator.h"
#include "frontend/parallel/strategy.h"
#include "frontend/parallel/graph_util/generate_graph.h"
#include "frontend/parallel/tensor_layout/tensor_redistribution.h"
#include "pipeline/jit/ps/resource.h"
#include "ops/op_utils.h"

namespace mindspore {
namespace parallel {
// ResizeBilinearV2: support to split N/C/W
// ResizeNearestNeighbor: support to split N/C/H/W if align_corners=False, support to split N/C if align_corners=True
Status ResizeBilinearV2Info::GetAttrs() {
  auto op_def = mindspore::ops::GetOpDef(GetPrimNameFromInfoName(name_));
  if (op_def == nullptr) {
    size_ = GetTupleIntAttr(SIZE);
    align_corners_ = GetBoolAttr(ALIGN_CORNERS);
  } else {
    auto size_opt = GetArrayValueFromInputs<int64_t>(input_value_, name_, SIZE);
    if (!size_opt.has_value()) {
      MS_LOG(ERROR) << "For " << name_ << ", failed to get value for " << SIZE << ".";
    }
    size_ = size_opt.value();

    auto align_corners_opt = GetScalarValueFromInputs<bool>(input_value_, name_, ALIGN_CORNERS);
    if (!align_corners_opt.has_value()) {
      MS_LOG(ERROR) << "For " << name_ << ", failed to get value for " << ALIGN_CORNERS << ".";
    }
    align_corners_ = align_corners_opt.value();
  }
  if (size_.size() != 2) {
    MS_LOG(ERROR) << name_ << ": The size of input size must be 2, but got " << size_.size();
    return FAILED;
  }
  MS_LOG(INFO) << name_ << ": The input size is " << size_ << ", align_corners is " << align_corners_;

  return SUCCESS;
}

Status ResizeBilinearV2Info::CheckStrategy(const StrategyPtr &strategy) {
  MS_EXCEPTION_IF_NULL(strategy);
  need_exchange_overlap_ = false;
  if (CheckStrategyValue(strategy, inputs_shape_) != SUCCESS) {
    MS_LOG(ERROR) << name_ << ": Check input strategy failed";
    return FAILED;
  }

  std::vector<Dimensions> stra = strategy->GetInputDim();
  if (stra.size() != 1) {
    MS_LOG(ERROR) << name_ << ": The size of strategy must be 1, but got " << stra.size();
    return FAILED;
  }

  Dimensions input_strategy = stra[0];
  if (input_strategy.size() != 4) {
    MS_LOG(ERROR) << name_ << ": The size of input strategy must be 4, but got" << input_strategy.size();
    return FAILED;
  }

  if (input_strategy[2] != 1) {
    MS_LOG(ERROR) << name_ << ": Do not support split H dimension";
    return FAILED;
  }

  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  std::string backend = ms_context->get_param<std::string>(MS_CTX_DEVICE_TARGET);
  if (input_strategy[3] != 1) {
    if (backend == kGPUDevice) {
      MS_LOG(ERROR) << name_ << ": Do not support split W dimension in GPU platform";
      return FAILED;
    }
    need_exchange_overlap_ = true;
    MS_LOG(INFO) << name_ << ": Split the w dimension";
  }

  // check output strategy
  if (CheckStrategyValue(strategy, outputs_shape_) != SUCCESS) {
    MS_LOG(ERROR) << name_ << ": Check output strategy failed";
    return FAILED;
  }

  return SUCCESS;
}

Status ResizeBilinearV2Info::CheckStrategyForDynamicShape(const StrategyPtr &) {
  if (inputs_shape_[0][2] == -1 || inputs_shape_[0][3] == -1 || outputs_shape_[0][2] == -1 ||
      outputs_shape_[0][3] == -1) {
    MS_LOG(ERROR) << name_ << ": it does not support H or W dimension dynamic shape now, the input shape is "
                  << ShapeToString(inputs_shape_[0]) << ", the output shape is " << ShapeToString(outputs_shape_[0]);
    return FAILED;
  }
  return SUCCESS;
}

Status ResizeBilinearV2Info::InferDevMatrixShape() {
  // the strategy is (n, c, h, w)
  // the dev matrix is (n, c, h, w)
  MS_EXCEPTION_IF_NULL(strategy_);
  std::vector<Dimensions> stra = strategy_->GetInputDim();
  if (stra.empty()) {
    MS_LOG(ERROR) << name_ << ": The strategy is empty";
    return FAILED;
  }

  if (stra[0].size() != 4) {
    MS_LOG(ERROR) << name_ << ": The size of strategy must be 4, but got " << stra[0].size();
    return FAILED;
  }

  dev_matrix_shape_ = stra[0];
  slice_size_ = size_;
  MS_EXCEPTION_IF_ZERO("dev_matrix_shape_[2]", dev_matrix_shape_[2]);
  MS_EXCEPTION_IF_ZERO("dev_matrix_shape_[3]", dev_matrix_shape_[3]);
  slice_size_[0] = slice_size_[0] / dev_matrix_shape_[2];
  slice_size_[1] = slice_size_[1] / dev_matrix_shape_[3];
  w_dimension_shard_num_ = dev_matrix_shape_[3];
  return SUCCESS;
}

Status ResizeBilinearV2Info::InferMirrorOps() {
  if (OperatorInfo::InferMirrorOps() != SUCCESS) {
    return FAILED;
  }

  if (mirror_ops_.empty()) {
    return SUCCESS;
  }

  OperatorVector op_for_size;
  OperatorVector op_for_align_corners;

  if (mindspore::ops::GetOpDef(GetPrimNameFromInfoName(name_))) {
    (void)mirror_ops_.emplace_back(std::move(op_for_size));
    (void)mirror_ops_.emplace_back(std::move(op_for_align_corners));
  }

  return SUCCESS;
}

Status ResizeBilinearV2Info::InferTensorMap() {
  // input_strategy: (n, c, h, w)
  // output_strategy: (n, c, h, w)
  // dev_matrix: (n, c, h, w)
  TensorMap input_tensor_map = {3, 2, 1, 0};
  TensorMap output_tensor_map = {3, 2, 1, 0};

  (void)inputs_tensor_map_.emplace_back(std::move(input_tensor_map));
  (void)outputs_tensor_map_.emplace_back(std::move(output_tensor_map));
  return SUCCESS;
}

Status ResizeBilinearV2Info::SetCostUnderStrategy(const StrategyPtr &strategy) {
  return SetCostUnderStrategyBase(strategy);
}

std::vector<StrategyPtr> ResizeBilinearV2Info::GenerateOpStrategies(int64_t stage_id) {
  Shape input0_split(inputs_shape_[0].size(), 0);
  input0_split[0] = 1;
  Shapes splittable_inputs = {input0_split};

  std::vector<StrategyPtr> sp_vector;
  if (GenerateStrategiesForIndependentInputs(stage_id, inputs_shape_, splittable_inputs, &sp_vector) != SUCCESS) {
    MS_LOG(EXCEPTION) << name_ << " : Generate strategies for independent inputs() failed.";
  }
  return sp_vector;
}

void ResizeBilinearV2Info::ReplaceNodeInputOrAttrs() {
  // if need exchange overlap, use replace_graph()
  if (need_exchange_overlap_) {
    return;
  }
  for (auto &cnode : cnodes_) {
    auto prim = GetValueNode<PrimitivePtr>(cnode->input(0));
    auto op_def = mindspore::ops::GetOpDef(prim->name());
    if (op_def != nullptr) {
      auto [is_input, node_input_idx] = CheckAndGetValidIdxByOpDef(op_def, prim->name(), SIZE, cnode->size());
      if (is_input) {
        cnode->set_input(node_input_idx, NewValueNode(MakeValue(slice_size_)));
        continue;
      }
    }
    prim->set_attr(SIZE, MakeValue(slice_size_));
  }
}

Status ResizeBilinearV2Info::InferRankBias() {
  // the origin dev_matrix is [n, c, h, w]
  // if repeated calculation
  //     1) repeated num in the left of dev matrix, the dev_matrix is [repeated_num, n, c, h, w]
  //     2) repeated num in the right of dev matrix, the dev_matrix is [n, c, h, w, repeated_num]
  uint64_t w_index_in_dev_matrix = 3;
  if (repeated_calc_num_ > 1 && !repeated_num_in_dev_matrix_right_) {
    w_index_in_dev_matrix += 1;
  }

  CheckGlobalDeviceManager();
  int64_t rank = g_device_manager->global_rank();
  DeviceMatrix dev_matrix(rank, stage_device_list_, dev_matrix_shape_);
  RankList group_devices;
  if (dev_matrix.GetDevicesAlongDim(w_index_in_dev_matrix, &group_devices) != SUCCESS) {
    return FAILED;
  }

  if (group_devices.size() <= 1) {
    MS_LOG(INFO) << name_ << ": The devices' size of w dimension is " << group_devices.size()
                 << ", no need to infer rank bias";
    return SUCCESS;
  }

  if (group_devices.size() != LongToSize(w_dimension_shard_num_)) {
    MS_LOG(ERROR) << name_ << ": The devices' size of w dimension is " << group_devices.size()
                  << ", but the shard num of w dimension is " << w_dimension_shard_num_;
    return FAILED;
  }

  std::vector<int64_t>::iterator it = std::find(group_devices.begin(), group_devices.end(), rank);
  if (it == group_devices.end()) {
    MS_LOG(ERROR) << name_ << ": Can not find the current rank in device list of w dimension, the current rank is "
                  << rank << ", the device list is " << group_devices;
    return FAILED;
  }

  rank_bias_ = std::distance(group_devices.begin(), it);
  if (it == group_devices.begin()) {
    // the current rank is on the left boundary
    left_rank_bias_ = -1;
    right_rank_bias_ = rank_bias_ + 1;

    left_rank_id_ = -1;
    right_rank_id_ = *(it + 1);
  } else if (it == group_devices.end() - 1) {
    // the current rank is on the right boundary
    left_rank_bias_ = rank_bias_ - 1;
    right_rank_bias_ = -1;

    left_rank_id_ = *(it - 1);
    right_rank_id_ = -1;
  } else {
    // the current rank is middle rank
    left_rank_bias_ = rank_bias_ - 1;
    right_rank_bias_ = rank_bias_ + 1;

    left_rank_id_ = *(it - 1);
    right_rank_id_ = *(it + 1);
  }

  all_to_all_group_ = g_device_manager->world_group();  // use world group temporarily
  MS_LOG(INFO) << name_ << ": The current rank is " << rank << ", the device list of w dimension is " << group_devices
               << ", the rank bias is " << rank_bias_ << ", the left rank bias is " << left_rank_bias_
               << ", the right rank bias is " << right_rank_bias_ << ", the left rank id is " << left_rank_id_
               << ", the right rank id is " << right_rank_id_ << ", the all to all group is " << all_to_all_group_;
  return SUCCESS;
}

void ResizeBilinearV2Info::InferScale() {
  origin_in_w_shape_ = inputs_shape_[0][3];
  origin_out_w_shape_ = outputs_shape_[0][3];

  if (origin_out_w_shape_ == 1) {
    MS_LOG(EXCEPTION) << name_ << ": Do not support that the w dimension of output shape is 1";
  }

  if (align_corners_) {
    MS_EXCEPTION_IF_ZERO("origin_out_w_shape_ - 1", origin_out_w_shape_ - 1);
    w_scale_ = LongToDouble(origin_in_w_shape_ - 1) / LongToDouble(origin_out_w_shape_ - 1);
  } else {
    MS_EXCEPTION_IF_ZERO("origin_out_w_shape_", origin_out_w_shape_);
    w_scale_ = LongToDouble(origin_in_w_shape_) / LongToDouble(origin_out_w_shape_);
  }

  MS_LOG(INFO) << name_ << ": The scale is " << w_scale_;
}

int64_t ResizeBilinearV2Info::InferOverlapLeftSizeByRankBias(int64_t rank_bias) {
  // left_overlap_size = (rank * ori_in_w / w_shard) - floor(scale * rank * slice_w)
  int64_t map_left_boundary = DoubleToLong(std::floor(w_scale_ * rank_bias * slice_size_[1]));
  MS_EXCEPTION_IF_ZERO("w_dimension_shard_num_", w_dimension_shard_num_);
  int64_t local_left_boundary = rank_bias * origin_in_w_shape_ / w_dimension_shard_num_;

  if (map_left_boundary > local_left_boundary) {
    MS_LOG(EXCEPTION) << name_ << ": Invalid left overlap, the rank bias is " << rank_bias << ", the map boundary is "
                      << map_left_boundary << ", the local boundary is " << local_left_boundary;
  }
  return local_left_boundary - map_left_boundary;
}

int64_t ResizeBilinearV2Info::InferOverlapRightSizeByRankBias(int64_t rank_bias) {
  // right_overlap_size = ceil(scale * (rank + 1) * slice_w - 1) - ((rank + 1) * ori_in_w / w_shard - 1)
  int64_t map_right_boundary = DoubleToLong(std::ceil(w_scale_ * ((rank_bias + 1) * slice_size_[1] - 1)));
  MS_EXCEPTION_IF_ZERO("w_dimension_shard_num_", w_dimension_shard_num_);
  int64_t local_right_boundary = (rank_bias + 1) * origin_in_w_shape_ / w_dimension_shard_num_ - 1;

  // need to handle this special condition
  if (map_right_boundary > origin_in_w_shape_ - 1) {
    map_right_boundary = origin_in_w_shape_ - 1;
  }

  if (map_right_boundary < local_right_boundary) {
    MS_LOG(EXCEPTION) << name_ << ": Invalid right overlap, the rank bias is " << rank_bias << ", the map boundary is "
                      << map_right_boundary << ", the local boundary is " << local_right_boundary;
  }

  return map_right_boundary - local_right_boundary;
}

void ResizeBilinearV2Info::InferOverlapSize() {
  overlap_left_size_ = InferOverlapLeftSizeByRankBias(rank_bias_);
  overlap_right_size_ = InferOverlapRightSizeByRankBias(rank_bias_);

  if (rank_bias_ == 0) {
    // it has not left rank
    left_rank_overlap_right_size_ = 0;
    right_rank_overlap_left_size_ = InferOverlapLeftSizeByRankBias(right_rank_bias_);
  } else if (rank_bias_ == w_dimension_shard_num_ - 1) {
    // it has not right rank
    left_rank_overlap_right_size_ = InferOverlapRightSizeByRankBias(left_rank_bias_);
    right_rank_overlap_left_size_ = 0;
  } else {
    // it has left rank and right rank
    left_rank_overlap_right_size_ = InferOverlapRightSizeByRankBias(left_rank_bias_);
    right_rank_overlap_left_size_ = InferOverlapLeftSizeByRankBias(right_rank_bias_);
  }

  MS_LOG(INFO) << name_ << ": the left overlap size of current rank is " << overlap_left_size_
               << ", the right overlap size of current rank is " << overlap_right_size_
               << ", the right overlap size of left rank is " << left_rank_overlap_right_size_
               << ", the left overlap size of right rank is " << right_rank_overlap_left_size_;
}

void ResizeBilinearV2Info::InferCommunicationAttrs() {
  // send rank ids: [-1, -1, send_right_rank, -1, -1, -1, send_left_rank, -1]
  // recv rank ids: [-1, -1, recv_right_rank, -1, -1, -1, recv_left_rank, -1]
  // send lens: [0, 0, send_left_len, send_right_len]
  // recv lens: [0, 0, recv_left_len, recv_right_len]
  int64_t send_right_rank = -1;
  int64_t send_left_rank = -1;
  int64_t recv_right_rank = -1;
  int64_t recv_left_rank = -1;
  int64_t send_left_len = 0;
  int64_t send_right_len = 0;
  int64_t recv_left_len = 0;
  int64_t recv_right_len = 0;

  if (rank_bias_ == 0) {
    // the first rank
    send_right_len = right_rank_overlap_left_size_;
    send_right_rank = send_right_len > 0 ? right_rank_id_ : -1;

    recv_right_len = overlap_right_size_;
    recv_right_rank = recv_right_len > 0 ? right_rank_id_ : -1;
  } else if (rank_bias_ == w_dimension_shard_num_ - 1) {
    // the last rank
    send_left_len = left_rank_overlap_right_size_;
    send_left_rank = send_left_len > 0 ? left_rank_id_ : -1;

    recv_left_len = overlap_left_size_;
    recv_left_rank = recv_left_len > 0 ? left_rank_id_ : -1;
  } else {
    // the middle rank
    send_right_len = right_rank_overlap_left_size_;
    send_right_rank = send_right_len > 0 ? right_rank_id_ : -1;

    recv_right_len = overlap_right_size_;
    recv_right_rank = recv_right_len > 0 ? right_rank_id_ : -1;
    send_left_len = left_rank_overlap_right_size_;
    send_left_rank = send_left_len > 0 ? left_rank_id_ : -1;

    recv_left_len = overlap_left_size_;
    recv_left_rank = recv_left_len > 0 ? left_rank_id_ : -1;
  }

  send_rank_ids_ = {-1, -1, send_right_rank, -1, -1, -1, send_left_rank, -1};
  recv_rank_ids_ = {-1, -1, recv_right_rank, -1, -1, -1, recv_left_rank, -1};
  send_lens_ = {0, 0, send_left_len, send_right_len};
  recv_lens_ = {0, 0, recv_left_len, recv_right_len};
  MS_LOG(INFO) << name_ << ": The send rank ids is " << send_rank_ids_ << ", the send lens is " << send_lens_
               << ", the recv rank ids is " << recv_rank_ids_ << ", the recv lens is " << recv_lens_;
}

void ResizeBilinearV2Info::InferResizeBilinearV2Attrs() {
  origin_image_size_ = {inputs_shape_[0][2], inputs_shape_[0][3]};
  src_start_w_ = DoubleToLong(std::floor(w_scale_ * rank_bias_ * slice_size_[1]));
  dst_start_w_ = rank_bias_ * slice_size_[1];

  MS_LOG(INFO) << name_ << ": The origin image size is " << origin_image_size_ << ", src start index is "
               << src_start_w_ << ", dst start index is " << dst_start_w_;
}

void ResizeBilinearV2Info::InferNewOperatorAttrs() {
  InferCommunicationAttrs();
  InferResizeBilinearV2Attrs();
}

OperatorAttrs ResizeBilinearV2Info::CreateNeighborExchangeV2Attrs() {
  // the type of send_rank_ids, recv_rank_ids, send_lens, recv_lens is list, is not tuple, can not use MakeValue
  // the MakeValue(vector) return a tuple
  Attr send_rank_ids = {SEND_RANK_IDS, MakeListValue(send_rank_ids_)};
  Attr send_lens = {SEND_LENS, MakeListValue(send_lens_)};
  Attr recv_rank_ids = {RECV_RANK_IDS, MakeListValue(recv_rank_ids_)};
  Attr recv_lens = {RECV_LENS, MakeListValue(recv_lens_)};
  Attr data_format = {DATA_FORMAT, MakeValue(NCHW)};
  Attr group = {GROUP, MakeValue(all_to_all_group_)};

  OperatorAttrs attrs = {send_rank_ids, send_lens, recv_rank_ids, recv_lens, data_format, group};
  return attrs;
}

OperatorAttrs ResizeBilinearV2Info::CreateParallelResizeBilinearAttrs() {
  Attr ori_image_size = {ORI_IMAGE_SIZE, MakeValue(origin_image_size_)};
  Attr split_size = {SPLIT_SIZE, MakeValue(slice_size_)};
  Attr src_start_w = {SRC_START_W, MakeValue(src_start_w_)};
  Attr dst_start_w = {DST_START_W, MakeValue(dst_start_w_)};
  Attr align_corners = {ALIGN_CORNERS, MakeValue(align_corners_)};

  OperatorAttrs attrs = {ori_image_size, split_size, src_start_w, dst_start_w, align_corners};
  return attrs;
}

void ResizeBilinearV2Info::InferReplaceGraph(const CNodePtr &cnode) {
  auto graph = cnode->func_graph();
  MS_EXCEPTION_IF_NULL(graph);

  GenerateGraph gen_g = GenerateGraph(attrs_);
  if (gen_g.Init(cnode) != SUCCESS) {
    MS_LOG(EXCEPTION) << name_ << ": Init generator graph failed";
  }

  auto neighbor_exchange_v2_attrs = CreateNeighborExchangeV2Attrs();
  auto neighbor_exchange_v2_node =
    gen_g.PushBack({gen_g.NewOpInst(NEIGHBOREXCHANGEV2, neighbor_exchange_v2_attrs), gen_g.virtual_input_node()});

  auto size = CreateValueTupleAnfNodePtr(size_);
  auto parallel_resize_bilinear_attrs = CreateParallelResizeBilinearAttrs();
  auto parallel_resize_bilinear_node = gen_g.PushBack(
    {gen_g.NewOpInst(PARALLEL_RESIZE_BILINEAR, parallel_resize_bilinear_attrs), neighbor_exchange_v2_node, size});

  std::vector<std::pair<AnfNodePtr, int64_t>> input_nodes = {std::make_pair(neighbor_exchange_v2_node, 1)};
  replace_graph_ = std::make_shared<std::pair<std::vector<std::pair<AnfNodePtr, int64_t>>, AnfNodePtr>>(
    std::make_pair(input_nodes, parallel_resize_bilinear_node));
}

ReplaceGraphPtr ResizeBilinearV2Info::replace_graph(const CNodePtr &cnode) {
  if (!need_exchange_overlap_) {
    return nullptr;
  }

  if (InferRankBias() != SUCCESS) {
    MS_LOG(EXCEPTION) << name_ << ": infer rank bias failed";
  }

  InferScale();

  InferOverlapSize();

  InferNewOperatorAttrs();

  InferReplaceGraph(cnode);

  return replace_graph_;
}

Status ResizeNearestNeighborInfo::CheckStrategy(const StrategyPtr &strategy) {
  MS_EXCEPTION_IF_NULL(strategy);

  if (align_corners_) {
    std::vector<Dimensions> stra = strategy->GetInputDim();
    if (stra.size() != 1) {
      MS_LOG(ERROR) << name_ << ": The size of strategy must be 1, but got " << stra.size();
      return FAILED;
    }

    Dimensions input_strategy = stra[0];
    if (input_strategy.size() != 4) {
      MS_LOG(ERROR) << name_ << ": The size of input strategy must be 4, but got" << input_strategy.size();
      return FAILED;
    }

    if (input_strategy[2] != 1 || input_strategy[3] != 1) {
      MS_LOG(ERROR) << name_ << ": The align_corners is True, do not support split from H or W";
      return FAILED;
    }
  }

  // check input strategy
  if (CheckStrategyValue(strategy, inputs_shape_) != SUCCESS) {
    MS_LOG(ERROR) << name_ << ": Check input strategy failed";
    return FAILED;
  }

  // check output strategy
  if (CheckStrategyValue(strategy, outputs_shape_) != SUCCESS) {
    MS_LOG(ERROR) << name_ << ": Check output strategy failed";
    return FAILED;
  }

  return SUCCESS;
}

std::vector<StrategyPtr> ResizeNearestNeighborInfo::GenerateOpStrategies(int64_t stage_id) {
  Shape multiples_split(inputs_shape_[0].size(), 1);
  if (align_corners_) {
    multiples_split[2] = 0;
    multiples_split[3] = 0;
  }
  Shapes splittable_inputs = {multiples_split};

  std::vector<StrategyPtr> sp_vector;
  if (GenerateStrategiesForIndependentInputs(stage_id, inputs_shape_, splittable_inputs, &sp_vector) != SUCCESS) {
    MS_LOG(EXCEPTION) << name_ << ": generate strategies failed";
  }

  return sp_vector;
}

REGISTER(ResizeBilinearV2Info);
REGISTER(ResizeNearestNeighborInfo);
}  // namespace parallel
}  // namespace mindspore
