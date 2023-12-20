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

#include "frontend/parallel/ops_info/gatherd_info.h"

#include <algorithm>
#include <memory>
#include <utility>
#include <vector>

#include "frontend/parallel/device_matrix.h"
#include "frontend/parallel/dynamic_creator.h"
#include "frontend/parallel/strategy.h"
#include "frontend/parallel/graph_util/generate_graph.h"
#include "frontend/parallel/tensor_layout/tensor_redistribution.h"
#include "pipeline/jit/ps/resource.h"

namespace mindspore {
namespace parallel {
Status GatherDInfo::GetAttrs() {
  if (input_value_.size() != 3) {
    MS_LOG(ERROR) << name_ << ": Invalid input_value's size " << input_value_.size();
    return FAILED;
  }

  if (!input_value_[1]->isa<Int64Imm>()) {
    MS_LOG(ERROR) << name_ << ": The value of dim is not int";
    return FAILED;
  }

  int64_t dim = GetValue<int64_t>(input_value_[1]);
  int64_t input_dim = SizeToLong(inputs_shape_[0].size());
  if ((dim > (input_dim - 1)) || (dim < -input_dim)) {
    MS_LOG(ERROR) << name_ << ": The dim(" << dim << ") is out of range[" << (-input_dim) << ", " << (input_dim - 1)
                  << "]";
    return FAILED;
  }

  if (dim < 0) {
    dim += input_dim;
  }

  dim_ = LongToSize(dim);
  MS_LOG(INFO) << name_ << ": The dim is " << dim_;
  return SUCCESS;
}

Status GatherDInfo::CheckStrategy(const StrategyPtr &strategy) {
  MS_EXCEPTION_IF_NULL(strategy);
  if (CheckStrategyValue(strategy, inputs_shape_) != SUCCESS) {
    MS_LOG(ERROR) << name_ << ": Invalid strategy";
    return FAILED;
  }

  std::vector<Dimensions> stra = strategy->GetInputDim();
  if (stra.size() != 2) {
    MS_LOG(ERROR) << name_ << ": The size of strategy must be 2, but got " << stra.size();
    return FAILED;
  }

  const Dimensions &input_strategy = stra[0];
  const Dimensions &index_strategy = stra[1];
  if (input_strategy.size() != index_strategy.size()) {
    MS_LOG(ERROR) << name_ << ": The dimension of X and Index strategy should be same, but got "
                  << input_strategy.size() << " and " << index_strategy.size();
    return FAILED;
  }

  for (size_t i = 0; i < input_strategy.size(); i++) {
    if (i != dim_) {
      if (input_strategy[i] != index_strategy[i]) {
        MS_LOG(ERROR) << name_ << ": The dimension " << i << "of X and Index should be same, but got "
                      << input_strategy[i] << " and " << index_strategy[i];
        return FAILED;
      }
      continue;
    }

    if (index_strategy[i] != 1) {
      MS_LOG(ERROR) << name_ << ": Index dimension dim can not be splited.";
      return FAILED;
    }
  }

  axis_shard_ = false;
  if (input_strategy[dim_] != 1) {
    axis_shard_ = true;
  }

  return SUCCESS;
}

Status GatherDInfo::CheckStrategyForDynamicShape(const StrategyPtr &strategy) {
  std::vector<Dimensions> stra = strategy->GetInputDim();
  if (axis_shard_ && (inputs_shape_[0][dim_] == -1)) {
    MS_LOG(ERROR) << name_ << ": it does not support the dim-axis is split if the dim is dynamic, the strategy: "
                  << ShapesToString(stra) << ", the inputs' shape: " << ShapesToString(inputs_shape_)
                  << ", the axis: " << dim_;
    return FAILED;
  }
  return SUCCESS;
}

Status GatherDInfo::InferDevMatrixShape() {
  MS_EXCEPTION_IF_NULL(strategy_);
  std::vector<Dimensions> stra = strategy_->GetInputDim();
  if (stra.empty()) {
    MS_LOG(ERROR) << name_ << ": The strategy is empty";
    return FAILED;
  }

  dev_matrix_shape_ = stra[0];
  return SUCCESS;
}

Status GatherDInfo::InferTensorMap() {
  Shape input_tensor_map;
  size_t size = inputs_shape_[0].size();
  for (size_t i = 0; i < size; ++i) {
    input_tensor_map.push_back(SizeToLong(size - i - 1));
  }

  Shape index_tensor_map = input_tensor_map;
  index_tensor_map[dim_] = -1;

  inputs_tensor_map_.push_back(input_tensor_map);   // input
  inputs_tensor_map_.push_back(index_tensor_map);   // index
  outputs_tensor_map_.push_back(index_tensor_map);  // output
  return SUCCESS;
}

Status GatherDInfo::InferTensorInfo() {
  if (inputs_shape_.empty() || outputs_shape_.empty() || inputs_tensor_map_.empty() || outputs_tensor_map_.empty()) {
    MS_LOG(ERROR) << name_ << ": Invalid args";
    return FAILED;
  }

  for (size_t i = 0; i < inputs_tensor_map_.size(); ++i) {
    TensorLayout input_layout;
    if (input_layout.InitFromVector(dev_matrix_shape_, inputs_tensor_map_[i], inputs_shape_[i]) != SUCCESS) {
      MS_LOG(ERROR) << name_ << ": Infer input tensor layout failed, the index is " << i;
      return FAILED;
    }
    TensorInfo input_tensor_info(input_layout);
    inputs_tensor_info_.push_back(input_tensor_info);
  }

  TensorInfo dim_tensor_info;
  (void)inputs_tensor_info_.insert(inputs_tensor_info_.cbegin() + 1, dim_tensor_info);

  for (size_t i = 0; i < outputs_tensor_map_.size(); ++i) {
    TensorLayout output_layout;
    if (output_layout.InitFromVector(dev_matrix_shape_, outputs_tensor_map_[i], outputs_shape_[i]) != SUCCESS) {
      MS_LOG(ERROR) << name_ << ": Infer output tensor layout failed, the index is " << i;
      return FAILED;
    }
    TensorInfo output_tensor_info(output_layout);
    outputs_tensor_info_.push_back(output_tensor_info);
  }
  return SUCCESS;
}

Status GatherDInfo::InferMirrorOps() {
  mirror_ops_.clear();
  if (inputs_shape_.empty()) {
    MS_LOG(INFO) << name_ << ": The inputs size is empty";
    return SUCCESS;
  }

  if (inputs_tensor_map_.size() != inputs_shape_.size()) {
    MS_LOG(ERROR) << name_ << ": The size of inputs tensor map is not equal to the size of inputs shape";
    return FAILED;
  }

  std::vector<Group> group;
  if (CreateGroupByTensorMap(inputs_tensor_map_[0], &group) != SUCCESS) {
    ReportError(name_ + ": Create group failed.");
    return FAILED;
  }

  if (group.empty()) {
    MS_LOG(INFO) << name_ << ": No need to create mirror ops";
    return SUCCESS;
  }

  OperatorVector mirror_op = CreateMirrorOps(group[0].name(), group[0].GetDevNum());
  mirror_ops_.push_back(mirror_op);  // input

  OperatorVector tmp_mirror_op;  // dim
  mirror_ops_.push_back(tmp_mirror_op);

  mirror_ops_.push_back(mirror_op);  // index
  return SUCCESS;
}

void GatherDInfo::ReComputeBatchSplitFlagList() {
  if (InferAttrs() != SUCCESS) {
    MS_LOG(EXCEPTION) << name_ << ": Infer attrs failed";
  }

  if (dim_ == 0) {
    for (size_t i = 0; i < inputs_shape_.size(); ++i) {
      split_flag_list_[i] = false;
    }
    MS_LOG(INFO) << name_ << ": the dim is 0, can not split batch dim";
    return;
  }

  for (size_t i = 0; i < inputs_shape_.size(); ++i) {
    split_flag_list_[i] = true;
  }
}

Status GatherDInfo::SetCostUnderStrategy(const StrategyPtr &strategy) { return SetCostUnderStrategyBase(strategy); }

std::vector<StrategyPtr> GatherDInfo::GenerateOpStrategies(int64_t stage_id) {
  Shape input0_split(inputs_shape_[0].size(), 1);
  input0_split[dim_] = 0;
  Shapes splittable_inputs = {input0_split};
  Shapes tmp_inputs_shape = {inputs_shape_[0]};

  std::vector<StrategyPtr> sp_vector;
  if (GenerateStrategiesForIndependentInputs(stage_id, tmp_inputs_shape, splittable_inputs, &sp_vector) != SUCCESS) {
    MS_LOG(EXCEPTION) << name_ << " : Generate strategies for independent inputs() failed.";
  }

  for (auto &sp : sp_vector) {
    if ((sp == nullptr) || sp->GetInputDim().empty()) {
      MS_LOG(EXCEPTION) << name_ << ": The strategy is null or empty";
    }
    Strategies tmp_strategy;
    Dimensions first_input_strategy = sp->GetInputDim()[0];
    for (size_t i = 0; i < inputs_shape_.size(); ++i) {
      tmp_strategy.push_back(first_input_strategy);
    }
    sp->ResetInputs(tmp_strategy);
  }
  return sp_vector;
}

Status GatherDInfo::InferBias() {
  CheckGlobalDeviceManager();
  int64_t rank = g_device_manager->rank_index_in_stage();
  const auto &input_shape = inputs_shape_.at(0);
  int64_t slice_num = dev_matrix_shape_.at(dim_);
  slice_size_ = input_shape.at(dim_) / slice_num;
  bias_ = rank % slice_num * slice_size_;
  return SUCCESS;
}

Status GatherDInfo::InferGroup() {
  int64_t rank = g_device_manager->global_rank();
  DeviceMatrix dev_matrix(rank, g_device_manager->GetDeviceListInThisStage(), dev_matrix_shape_);

  RankList group_devices;
  if (dev_matrix.GetDevicesAlongDim(SizeToUlong(dim_), &group_devices) != SUCCESS) {
    MS_LOG(ERROR) << name_ << ": Create group failed.";
    return FAILED;
  }

  MS_LOG(INFO) << name_ << ": The group ranks is " << group_devices;
  if (g_device_manager->CreateGroup(group_devices, &group_) != SUCCESS) {
    MS_LOG(ERROR) << name_ << ": create reduce group failed in table row split.";
    return FAILED;
  }
  return SUCCESS;
}

ReplaceGraphPtr GatherDInfo::replace_graph(const CNodePtr &cnode) {
  if (!axis_shard_) {
    return replace_graph_;
  }

  GenerateGraph gen_g = GenerateGraph(attrs_);
  if (gen_g.Init(cnode) != SUCCESS) {
    MS_LOG(EXCEPTION) << name_ << "GenerateGraph Init failed";
  }

  if (InferBias() != SUCCESS) {
    MS_LOG(EXCEPTION) << name_ << ": Infer Bias failed.";
  }

  // Split along `dim` logic of gatherd(x, dim, index):
  // Divide the x into N segments. Each rank keep a segment of x and full index, and responsible for gatherd a segment.
  // Then aggregate all segments with AllReduce.
  MS_LOG(INFO) << name_ << ": The rank is " << g_device_manager->rank_index_in_stage() << ", the bias is " << bias_;
  auto sub = gen_g.PushBack({gen_g.NewOpInst(SUB), gen_g.virtual_input_node(), CreateInt32Tensor(bias_)});
  auto relu = gen_g.PushBack({gen_g.NewOpInst(RELU), sub});
  auto minimum = gen_g.PushBack({gen_g.NewOpInst(MINIMUM), relu, CreateInt32Tensor(slice_size_ - 1)});
  auto equal = gen_g.PushBack({gen_g.NewOpInst(EQUAL), sub, minimum});
  auto gatherd = gen_g.PushBack({gen_g.NewOpInst(GATHERD), gen_g.virtual_input_node(), CreatInt64Imm(dim_), minimum});
  auto dtype = gen_g.PushBack({gen_g.NewOpInst(DTYPE), gatherd});
  auto dtype_id =
    gen_g.PushBack({gen_g.NewOpInst(DTYPETOENUM), CreateStringImm("DtypeToEnum"), CreateStringImm("dtype"), dtype});
  auto cast = gen_g.PushBack({gen_g.NewOpInst(CAST), equal, dtype_id});
  auto mul = gen_g.PushBack({gen_g.NewOpInst(MUL), gatherd, cast});

  if (InferGroup() != SUCCESS) {
    MS_LOG(EXCEPTION) << name_ << ": Infer Group failed.";
  }
  Attr attr_op = std::make_pair(OP, MakeValue(REDUCE_OP_SUM));
  Attr attr_group = std::make_pair(GROUP, MakeValue(group_.name()));
  OperatorAttrs attrs = {attr_op, attr_group};
  AnfNodePtr reduce_op = gen_g.PushBack({gen_g.NewOpInst(ALL_REDUCE, attrs), mul});
  std::vector<std::pair<AnfNodePtr, int64_t>> input_nodes = {std::make_pair(sub, 3), std::make_pair(gatherd, 1)};
  replace_graph_ = std::make_shared<std::pair<std::vector<std::pair<AnfNodePtr, int64_t>>, AnfNodePtr>>(
    std::make_pair(input_nodes, reduce_op));

  return replace_graph_;
}

REGISTER(GatherDInfo);
}  // namespace parallel
}  // namespace mindspore
