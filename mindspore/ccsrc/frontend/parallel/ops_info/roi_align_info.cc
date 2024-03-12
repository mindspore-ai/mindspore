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

#include "frontend/parallel/ops_info/roi_align_info.h"

#include <utility>
#include <functional>

#include "frontend/parallel/dynamic_creator.h"
#include "frontend/parallel/graph_util/generate_graph.h"
#include "include/common/utils/parallel_context.h"

namespace mindspore {
namespace parallel {
Status ROIAlignInfo::GetAttrs() {
  std::vector<std::string> attr_key_list = {POOLED_HEIGHT, POOLED_WIDTH, SPATIAL_SCALE, SAMPLE_NUM, ROI_END_MODE};
  for (const auto &attr_key : attr_key_list) {
    if (attrs_.find(attr_key) == attrs_.end()) {
      MS_LOG(ERROR) << name_ << ": Get primitive attr \"" << attr_key << "\" failed.";
      return FAILED;
    }
    (void)roi_align_attrs.emplace_back(std::make_pair(attr_key, attrs_[attr_key]));
  }
  return SUCCESS;
}

Status ROIAlignInfo::CheckStrategy(const StrategyPtr &strategy) {
  if (CheckStrategyValue(strategy, inputs_shape_) != SUCCESS) {
    return FAILED;
  }

  Strategies strategies = strategy->GetInputDim();
  auto features_strategy = strategies.at(0);
  auto rois_strategy = strategies.at(1);
  if (features_strategy[2] != 1 || features_strategy[3] != 1) {
    MS_LOG(ERROR) << name_
                  << ": Invalid strategy, the value of strategy[0][2] and strategy[0][3] must be 1, but got strategy "
                  << StrategyToString(strategies);
    return FAILED;
  }
  if (rois_strategy[1] != 1) {
    MS_LOG(ERROR) << name_ << ": Invalid strategy, the value of strategy[1][1] must be 1, but got strategy "
                  << StrategyToString(strategies);
    return FAILED;
  }
  return SUCCESS;
}

Status ROIAlignInfo::CheckStrategyForDynamicShape(const StrategyPtr &strategy) {
  auto strategies = strategy->GetInputDim();
  auto features_strategy = strategies[0];
  if (features_strategy[0] != 1) {
    MS_LOG(ERROR) << name_ << ": the dim-0 of first input can not be split if it's dynamic shape, the strategy is "
                  << ShapesToString(strategies) << ", the inputs' shape: " << ShapesToString(inputs_shape_);
    return FAILED;
  }
  return SUCCESS;
}

Status ROIAlignInfo::InferDevMatrixShape() {
  dev_matrix_shape_.clear();

  auto strategies = strategy_->GetInputDim();
  auto features_strategy = strategies.at(0);
  auto rois_strategy = strategies.at(1);
  dev_matrix_shape_ = {features_strategy[0], features_strategy[1], rois_strategy[0]};
  return SUCCESS;
}

Status ROIAlignInfo::InferTensorMap() {
  inputs_tensor_map_.clear();
  outputs_tensor_map_.clear();

  Shape features_map = {2, 1, -1, -1};
  Shape rois_map = {0, -1};
  Shape outputs_map = {0, 1, -1, -1};
  (void)inputs_tensor_map_.emplace_back(std::move(features_map));
  (void)inputs_tensor_map_.emplace_back(std::move(rois_map));
  (void)outputs_tensor_map_.emplace_back(std::move(outputs_map));
  return SUCCESS;
}

Status ROIAlignInfo::InferBias() {
  MS_EXCEPTION_IF_NULL(strategy_);
  CheckGlobalDeviceManager();

  int64_t rank = g_device_manager->rank_index_in_stage();
  auto strategies = strategy_->GetInputDim();
  auto features_strategy = strategies.at(0);
  auto features_shape = inputs_shape_.at(0);
  auto rois_strategy = strategies.at(1);
  auto rois_shape = inputs_shape_.at(1);

  MS_EXCEPTION_IF_ZERO("features_strategy[0]", features_strategy[0]);
  MS_EXCEPTION_IF_ZERO("rois_strategy[0]", rois_strategy[0]);
  if (features_shape[0] % features_strategy[0] != 0 || rois_shape[0] % rois_strategy[0] != 0) {
    return FAILED;
  }

  int64_t dev_num =
    std::accumulate(dev_matrix_shape_.begin() + 1, dev_matrix_shape_.end(), 1, std::multiplies<int64_t>());
  MS_EXCEPTION_IF_ZERO("dev_num", dev_num);
  features_slice_size_ = features_shape[0] / features_strategy[0];
  rois_slice_size_ = rois_shape[0] / rois_strategy[0];
  if (repeated_calc_num_ > 1 && !repeated_num_in_dev_matrix_right_) {
    bias_ = rank / dev_matrix_shape_[2] / dev_matrix_shape_[3] % dev_matrix_shape_[1] * features_slice_size_;
  } else {
    bias_ = rank / dev_num * features_slice_size_;
  }
  MS_LOG(INFO) << "Sharding batch, the rank is " << rank << ", features_slice size is " << features_slice_size_
               << ", rois_slice_size is " << rois_slice_size_ << ", bias is " << bias_;
  return SUCCESS;
}

Status ROIAlignInfo::InferGroup() {
  CheckGlobalDeviceManager();
  int64_t rank = g_device_manager->global_rank();
  DeviceMatrix dev_matrix(rank, stage_device_list_, dev_matrix_shape_);
  RankList group_devices;
  uint64_t dim = 0;
  if (repeated_calc_num_ > 1 && !repeated_num_in_dev_matrix_right_) {
    ++dim;
  }

  if (dev_matrix.GetDevicesAlongDim(dim, &group_devices) != SUCCESS) {
    MS_LOG(ERROR) << name_ << "Create group failed.";
    return FAILED;
  }
  if (group_devices.size() == 1) {
    MS_LOG(INFO) << name_ << ": The group is empty.";
  }

  MS_LOG(INFO) << name_ << ": The group rank is " << group_devices;
  if (g_device_manager->CreateGroup(group_devices, &group_) != SUCCESS) {
    MS_LOG(ERROR) << "The node " << cnode_->fullname_with_scope() << " create sync allreduce failed";
  }
  return SUCCESS;
}

Status ROIAlignInfo::ComputeReplaceGraph(const CNodePtr &cnode) {
  GenerateGraph gen_g = GenerateGraph(attrs_);
  if (gen_g.Init(cnode) != SUCCESS) {
    MS_LOG(ERROR) << "GenerateGraph Init failed.";
    return FAILED;
  }
  if (InferBias() != SUCCESS) {
    MS_LOG(ERROR) << "Infer bias failed.";
    return FAILED;
  }
  if (InferGroup() != SUCCESS) {
    MS_LOG(ERROR) << "Infer group failed";
    return FAILED;
  }

  CheckGlobalDeviceManager();
  MS_LOG(INFO) << name_ << ": The rank is " << g_device_manager->rank_index_in_stage() << ", the bias is " << bias_;
  auto begin = CreateValueTupleAnfNodePtr({0, 0});
  auto end = CreateValueTupleAnfNodePtr({rois_slice_size_, 0});
  auto strides = CreateValueTupleAnfNodePtr({1, 1});
  auto begin_mask = CreatInt64Imm(0);
  auto end_mask = CreatInt64Imm(0);
  auto ellipsis_mask = CreatInt64Imm(0);
  auto new_axis_mask = CreatInt64Imm(0);
  auto shrink_axis_mask = CreatInt64Imm(2);
  auto strided_slice = gen_g.PushBack({gen_g.NewOpInst(STRIDEDSLICE), gen_g.virtual_input_node(), begin, end, strides,
                                       begin_mask, end_mask, ellipsis_mask, new_axis_mask, shrink_axis_mask});
  auto dtype_rois = gen_g.PushBack({gen_g.NewOpInst(DTYPE), gen_g.virtual_input_node()});
  auto dtype_id_rois = gen_g.PushBack(
    {gen_g.NewOpInst(DTYPETOENUM), CreateStringImm("DtypeToEnum"), CreateStringImm("dtype"), dtype_rois});
  auto cast_bias = gen_g.PushBack({gen_g.NewOpInst(CAST), CreateInt32Tensor(bias_), dtype_id_rois});
  auto cast_slice_max_index =
    gen_g.PushBack({gen_g.NewOpInst(CAST), CreateInt32Tensor(features_slice_size_ - 1), dtype_id_rois});
  auto sub = gen_g.PushBack({gen_g.NewOpInst(SUB), strided_slice, cast_bias});
  auto relu = gen_g.PushBack({gen_g.NewOpInst(RELU), sub});
  auto minimum = gen_g.PushBack({gen_g.NewOpInst(MINIMUM), relu, cast_slice_max_index});
  auto stack = gen_g.PushBack(
    {gen_g.NewOpInst(STACK, {std::make_pair(AXIS, MakeValue(-1))}),
     CreateTensorTupleAnfNodePtr({std::make_shared<tensor::Tensor>(CreateRangeVector(rois_slice_size_)),
                                  std::make_shared<tensor::Tensor>(std::vector<int64_t>(rois_slice_size_, 0))})});
  auto tensor_scatter_update =
    gen_g.PushBack({gen_g.NewOpInst(TENSOR_SCATTER_UPDATE), gen_g.virtual_input_node(), stack, minimum});
  auto roi_align =
    gen_g.PushBack({gen_g.NewOpInst(ROI_ALIGN, roi_align_attrs), gen_g.virtual_input_node(), tensor_scatter_update});
  auto equal = gen_g.PushBack({gen_g.NewOpInst(EQUAL), sub, minimum});
  auto dtype_features = gen_g.PushBack({gen_g.NewOpInst(DTYPE), gen_g.virtual_input_node()});
  auto dtype_id_features = gen_g.PushBack(
    {gen_g.NewOpInst(DTYPETOENUM), CreateStringImm("DtypeToEnum"), CreateStringImm("dtype"), dtype_features});
  auto cast_equal = gen_g.PushBack({gen_g.NewOpInst(CAST), equal, dtype_id_features});
  auto expand_dims_0 = gen_g.PushBack({gen_g.NewOpInst(EXPAND_DIMS), cast_equal, CreatInt64Imm(-1)});
  auto expand_dims_1 = gen_g.PushBack({gen_g.NewOpInst(EXPAND_DIMS), expand_dims_0, CreatInt64Imm(-1)});
  auto expand_dims_2 = gen_g.PushBack({gen_g.NewOpInst(EXPAND_DIMS), expand_dims_1, CreatInt64Imm(-1)});
  auto mul = gen_g.PushBack({gen_g.NewOpInst(MUL), roi_align, expand_dims_2});
  Attr attr_reduce_op = std::make_pair(OP, MakeValue(REDUCE_OP_SUM));
  Attr attr_reduce_group = std::make_pair(GROUP, MakeValue(group_.name()));
  OperatorAttrs attrs_reduce = {attr_reduce_op, attr_reduce_group};
  AnfNodePtr reduce_op = gen_g.PushBack({gen_g.NewOpInst(ALL_REDUCE, attrs_reduce), mul});

  std::vector<std::pair<AnfNodePtr, int64_t>> inputs_nodes = {
    std::make_pair(strided_slice, 2), std::make_pair(dtype_rois, 2), std::make_pair(tensor_scatter_update, 2),
    std::make_pair(roi_align, 1), std::make_pair(dtype_features, 1)};
  replace_graph_ = std::make_shared<std::pair<std::vector<std::pair<AnfNodePtr, int64_t>>, AnfNodePtr>>(
    std::make_pair(inputs_nodes, reduce_op));
  return SUCCESS;
}

Status ROIAlignInfo::InitForCostModel(const StrategyPtr &strategy, const StrategyPtr &out_strategy) {
  if (InitForCostModelWithAutoRepeatCalc(strategy, out_strategy) != SUCCESS) {
    MS_LOG(DEBUG) << name_ << ": Init for cost model failed.";
    return FAILED;
  }
  auto strategies = strategy_->GetInputDim();
  auto features_strategy = strategies.at(0);
  auto roi_align_cost = std::dynamic_pointer_cast<ROIAlignCost>(operator_cost());
  auto pooled_height = GetValue<int64_t>(attrs_[POOLED_HEIGHT]);
  auto pooled_width = GetValue<int64_t>(attrs_[POOLED_WIDTH]);
  roi_align_cost->set_strategy(features_strategy);
  roi_align_cost->set_pooled_shape({pooled_height, pooled_width});
  MS_LOG(INFO) << name_ << ": Init for cost model success.";
  return SUCCESS;
}

ReplaceGraphPtr ROIAlignInfo::replace_graph(const CNodePtr &cnode) {
  auto strategies = strategy_->GetInputDim();
  auto features_strategy = strategies.at(0);
  if (features_strategy[0] != 1 && ComputeReplaceGraph(cnode) != SUCCESS) {
    MS_LOG(EXCEPTION) << name_ << ": ComputeReplaceGraph failed.";
  }
  return replace_graph_;
}

std::vector<int64_t> ROIAlignInfo::CreateRangeVector(int64_t upper_bound) const {
  std::vector<int64_t> range(upper_bound);
  for (int64_t i = 0; i < upper_bound; ++i) {
    range[LongToSize(i)] = i;
  }
  return range;
}

std::vector<StrategyPtr> ROIAlignInfo::GenerateOpStrategies(int64_t stage_id) {
  Shape features_split = {1, 1, 0, 0};
  Shape rois_spilt = {1, 0};
  Shapes splittable_inputs = {features_split, rois_spilt};

  std::vector<StrategyPtr> sp_vector;
  if (GenerateStrategiesForIndependentInputs(stage_id, inputs_shape_, splittable_inputs, &sp_vector) != SUCCESS) {
    MS_LOG(EXCEPTION) << name_ << ": Generate strategies for independent inputs() failed.";
  }
  return sp_vector;
}

REGISTER(ROIAlignInfo);
}  // namespace parallel
}  // namespace mindspore
