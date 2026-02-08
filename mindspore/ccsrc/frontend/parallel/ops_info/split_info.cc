/**
 * Copyright 2020-2024 Huawei Technologies Co., Ltd
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

#include "frontend/parallel/ops_info/split_info.h"

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>
#include <algorithm>

#include "include/utils/parallel_context.h"
#include "frontend/parallel/device_matrix.h"
#include "frontend/parallel/dynamic_creator.h"
#include "frontend/parallel/ops_info/operator_info.h"
#include "frontend/parallel/status.h"
#include "frontend/parallel/strategy.h"
#include "frontend/parallel/tensor_layout/tensor_redistribution.h"
#include "frontend/parallel/graph_util/generate_graph.h"

namespace mindspore {
namespace parallel {

constexpr char kSizeSplits[] = "size_splits";
constexpr char kSplitDim[] = "split_dim";
constexpr char kNumSplit[] = "num_split";

Status SplitInfo::GetAttrs() {
  auto axis_opt = GetScalarValueFromInputs<int64_t>(input_value_, name_, AXIS);
  if (!axis_opt.has_value()) {
    MS_LOG(ERROR) << name_ << ": Cannot get axis value.";
    return FAILED;
  }
  auto axis = axis_opt.value();

  if (inputs_shape_.empty()) {
    MS_LOG(ERROR) << name_ << ": The inputs shape is empty";
    return FAILED;
  }
  int dim = SizeToInt(inputs_shape_[0].size());
  if (axis < 0) {
    axis = axis + dim;
  }
  axis_ = LongToSize(axis);

  inputs_shape_ = Shapes{inputs_shape_[0]};  // Truncation for Strategy check.

  auto prim = GetCNodePrimitive(cnode_);
  MS_EXCEPTION_IF_NULL(prim);
  if (prim->HasAttr(parallel::SKIP_REDISTRIBUTION)) {
    skip_redistribution_ = GetValue<bool>(prim->GetAttr(parallel::SKIP_REDISTRIBUTION));
  }

  return SUCCESS;
}

Status SplitVInfo::GetAttrs() {
  int64_t axis = 0;

  auto axis_iter = attrs_.find(SPLIT_DIM);
  if (axis_iter != attrs_.end()) {
    MS_EXCEPTION_IF_NULL(axis_iter->second);
    if (axis_iter->second->isa<Int64Imm>()) {
      MS_EXCEPTION_IF_NULL(axis_iter->second->cast<Int64ImmPtr>());
      axis = axis_iter->second->cast<Int64ImmPtr>()->value();
    } else {
      MS_LOG(ERROR) << name_ << ": The value of axis is not int";
      return FAILED;
    }
  } else {
    MS_LOG(ERROR) << name_ << ": Can not find the axis attr";
    return FAILED;
  }

  if (inputs_shape_.empty()) {
    MS_LOG(ERROR) << name_ << ": The inputs shape is empty";
    return FAILED;
  }
  int dim = SizeToInt(inputs_shape_[0].size());
  if (axis < 0) {
    axis = axis + dim;
  }
  axis_ = LongToSize(axis);

  auto prim = GetCNodePrimitive(cnode_);
  MS_EXCEPTION_IF_NULL(prim);
  if (prim->HasAttr(parallel::SKIP_REDISTRIBUTION)) {
    skip_redistribution_ = GetValue<bool>(prim->GetAttr(parallel::SKIP_REDISTRIBUTION));
  }

  return SUCCESS;
}

Status SplitWithSizeInfo::GetAttrs() {
  auto dim_opt = GetScalarValueFromInputs<int64_t>(input_value_, name_, DIM);
  if (!dim_opt.has_value()) {
    MS_LOG(ERROR) << name_ << ": Cannot get dim value.";
    return FAILED;
  }
  auto dim = dim_opt.value();

  if (inputs_shape_.empty()) {
    MS_LOG(ERROR) << name_ << ": The inputs shape is empty";
    return FAILED;
  }
  int dim_size = SizeToInt(inputs_shape_[0].size());
  if (dim < 0) {
    dim = dim + dim_size;
  }
  axis_ = LongToSize(dim);

  inputs_shape_ = Shapes{inputs_shape_[0]};  // Truncation for Strategy check.

  auto prim = GetCNodePrimitive(cnode_);
  MS_EXCEPTION_IF_NULL(prim);
  if (prim->HasAttr(parallel::SKIP_REDISTRIBUTION)) {
    skip_redistribution_ = GetValue<bool>(prim->GetAttr(parallel::SKIP_REDISTRIBUTION));
  }

  return SUCCESS;
}

Status SplitTensorInfo::GetAttrs() {
  auto dim_opt = GetScalarValueFromInputs<int64_t>(input_value_, name_, DIM);
  if (!dim_opt.has_value()) {
    MS_LOG(ERROR) << name_ << ": Cannot get dim value.";
    return FAILED;
  }
  auto dim = dim_opt.value();

  if (inputs_shape_.empty()) {
    MS_LOG(ERROR) << name_ << ": The inputs shape is empty";
    return FAILED;
  }
  int dim_size = SizeToInt(inputs_shape_[0].size());
  if (dim < 0) {
    dim = dim + dim_size;
  }
  axis_ = LongToSize(dim);

  inputs_shape_ = Shapes{inputs_shape_[0]};  // Truncation for Strategy check.

  auto prim = GetCNodePrimitive(cnode_);
  MS_EXCEPTION_IF_NULL(prim);
  if (prim->HasAttr(parallel::SKIP_REDISTRIBUTION)) {
    skip_redistribution_ = GetValue<bool>(prim->GetAttr(parallel::SKIP_REDISTRIBUTION));
  }

  return SUCCESS;
}

Status SplitVInfo::ComputeReplaceGraphForInterleaved(const CNodePtr &cnode) {
  GenerateGraph gen_g = GenerateGraph(attrs_);
  if (gen_g.Init(cnode) != SUCCESS) {
    MS_LOG(ERROR) << name_ << "GenerateGraph Init failed";
    return FAILED;
  }
  auto interleaved_num = ParallelContext::GetInstance()->fine_grained_micro_interleaved_size();
  Attr output_nums_attr = {"output_nums", MakeValue(interleaved_num)};
  OperatorAttrs virtual_converter_begin_attrs = {output_nums_attr};
  auto virtual_converter_begin = gen_g.PushBack(
    {gen_g.NewOpInst(VIRTUAL_CONVERTER_BEGIN, virtual_converter_begin_attrs), gen_g.virtual_input_node()});
  std::vector<AnfNodePtr> virtual_converter_end_inputs_vector;
  std::vector<std::pair<AnfNodePtr, int64_t>> input_nodes = {std::make_pair(virtual_converter_begin, 1)};

  std::pair<std::string, ValuePtr> size_splits;
  if (attrs_.find(kSizeSplits) != attrs_.end()) {
    size_splits = std::make_pair(kSizeSplits, attrs_[kSizeSplits]);
  } else {
    MS_LOG(ERROR) << name_ << ": Can not find the size_splits attr";
    return FAILED;
  }
  std::pair<std::string, ValuePtr> num_split;
  if (attrs_.find(kNumSplit) != attrs_.end()) {
    num_split = std::make_pair(kNumSplit, attrs_[kNumSplit]);
  } else {
    MS_LOG(ERROR) << name_ << ": Can not find the num_split attr";
    return FAILED;
  }
  std::pair<std::string, ValuePtr> split_dim;
  if (attrs_.find(kSplitDim) != attrs_.end()) {
    split_dim = std::make_pair(kSplitDim, attrs_[kSplitDim]);
  } else {
    MS_LOG(ERROR) << name_ << ": Can not find the split_dim attr";
    return FAILED;
  }
  OperatorAttrs splitv_attrs = {size_splits, split_dim, num_split};
  for (int64_t i = 0; i < interleaved_num; ++i) {
    auto tuple_get_item = gen_g.PushBack({gen_g.NewOpInst(TUPLE_GETITEM), virtual_converter_begin, CreatInt64Imm(i)});
    std::vector<std::string> added_attrs = {kSizeSplits, kSplitDim, kNumSplit};
    OperatorAttrs other_attrs = {};
    for (const auto &attr : attrs_) {
      if (std::find(added_attrs.begin(), added_attrs.end(), attr.first) == added_attrs.end()) {
        other_attrs.push_back(std::make_pair(attr.first, attr.second));
      }
    }
    auto splitv_cnode = gen_g.NewOpInst(prim_name_, splitv_attrs, other_attrs);
    auto split = gen_g.PushBack({splitv_cnode, tuple_get_item});
    virtual_converter_end_inputs_vector.push_back(split);
  }
  Attr input_nums_attr = {"input_nums", MakeValue(interleaved_num)};
  OperatorAttrs virtual_converter_end_attrs = {input_nums_attr};
  std::vector<AnfNodePtr> virtual_converter_end_inputs = {
    gen_g.NewOpInst(VIRTUAL_CONVERTER_END, virtual_converter_end_attrs)};
  std::copy(virtual_converter_end_inputs_vector.begin(), virtual_converter_end_inputs_vector.end(),
            std::back_inserter(virtual_converter_end_inputs));
  auto virtual_converter_end = gen_g.PushBack(virtual_converter_end_inputs);
  replace_graph_ = std::make_shared<std::pair<std::vector<std::pair<AnfNodePtr, int64_t>>, AnfNodePtr>>(
    std::make_pair(input_nodes, virtual_converter_end));
  return SUCCESS;
}

void SplitWithSizeInfo::ReplaceNodeInputOrAttrs() {
  if (!skip_redistribution_) {
    return;
  }
  if (!IsValueNode<ValueTuple>(cnode_->input(kIndex2))) {
    MS_LOG_WITH_NODE(EXCEPTION, cnode_) << name_ << ": The input[2] of SplitWithSize cnode is not ValueTuple.";
  }
  auto tuple = GetValueNode<ValuePtr>(cnode_->input(kIndex2));
  MS_EXCEPTION_IF_NULL(tuple);
  std::vector<int64_t> size_splits = GetValue<std::vector<int64_t>>(tuple);
  std::vector<int64_t> new_size_splits;

  auto stra = inputs_tensor_info_.front().InferStrategy();
  for (size_t i = 0; i < size_splits.size(); ++i) {
    new_size_splits.push_back(size_splits[i] / stra[axis_]);
  }

  auto func_graph = cnode_->func_graph();
  MS_EXCEPTION_IF_NULL(func_graph);

  ValuePtr replace_shape = MakeValue(new_size_splits);
  AnfNodePtr val = NewValueNode(replace_shape);
  cnode_->set_input(kIndex2, val);
}

Status SplitInfo::CheckStrategy(const StrategyPtr &strategy) {
  MS_EXCEPTION_IF_NULL(strategy);
  if (CheckStrategyValue(strategy, inputs_shape_) != SUCCESS) {
    MS_LOG(ERROR) << name_ << ": Invalid strategy";
    return FAILED;
  }

  std::vector<Dimensions> stra = strategy->GetInputDim();
  if (stra.empty()) {
    MS_LOG(ERROR) << name_ << ": The strategy is empty";
    return FAILED;
  }

  if (axis_ >= stra[0].size()) {
    MS_LOG(ERROR) << name_ << ": The axis is out of range, the axis is " << axis_;
    return FAILED;
  }

  if (stra[0][axis_] != 1 && !skip_redistribution_) {
    MS_LOG(ERROR) << name_ << ": The axis can not be split";
    return FAILED;
  }

  return SUCCESS;
}

Status SplitInfo::InferDevMatrixShape() {
  MS_EXCEPTION_IF_NULL(strategy_);
  std::vector<Dimensions> stra = strategy_->GetInputDim();
  if (stra.empty()) {
    MS_LOG(ERROR) << name_ << "The strategy is empty";
    return FAILED;
  }

  dev_matrix_shape_ = stra[0];
  return SUCCESS;
}

Status SplitInfo::InferMirrorOps() {
  if (OperatorInfo::InferMirrorOps() != SUCCESS) {
    return FAILED;
  }

  if (mirror_ops_.empty()) {
    return SUCCESS;
  }

  OperatorVector op_for_axis;
  (void)mirror_ops_.emplace_back(std::move(op_for_axis));
  OperatorVector op_for_output_num;
  (void)mirror_ops_.emplace_back(std::move(op_for_output_num));
  return SUCCESS;
}

Status SplitInfo::InferTensorMap() {
  TensorMap tensor_map;
  if (inputs_shape_.empty()) {
    MS_LOG(ERROR) << name_ << "The inputs shape is empty";
    return FAILED;
  }

  int32_t size = SizeToInt(inputs_shape_[0].size());
  for (int i = 0; i < size; ++i) {
    tensor_map.push_back(size - i - 1);
  }

  inputs_tensor_map_.push_back(tensor_map);

  for (size_t i = 0; i < outputs_shape_.size(); ++i) {
    outputs_tensor_map_.push_back(tensor_map);
  }

  return SUCCESS;
}

Status SplitInfo::CheckInputLayout() {
  if (inputs_tensor_info_.size() != kSizeOne) {
    if (is_in_layout_propagation_) {
      MS_LOG(INFO) << "The size of input_tensor_layout for " << name_ << " is " << inputs_tensor_info_.size()
                   << " rather than 1.";
    } else {
      MS_LOG(ERROR) << "The size of input_tensor_layout for " << name_ << " is " << inputs_tensor_info_.size()
                    << " rather than 1.";
    }
    return FAILED;
  }
  auto stra = inputs_tensor_info_.front().InferStrategy();
  if (axis_ >= stra.size()) {
    if (is_in_layout_propagation_) {
      MS_LOG(INFO) << name_ << ": The axis is out of range, the axis is " << axis_;
    } else {
      MS_LOG(ERROR) << name_ << ": The axis is out of range, the axis is " << axis_;
    }
    return FAILED;
  }
  auto input_layout = inputs_tensor_info_.front().tensor_layout();
  if (input_layout.IsInterleavedParallel()) {
    auto tensor_map_axis = input_layout.tensor_map_before()[axis_];
    if (std::find(tensor_map_axis.begin(), tensor_map_axis.end(), 0) != tensor_map_axis.end()) {
      if (is_in_layout_propagation_) {
        MS_LOG(INFO) << name_ << ": The axis can not be split by interleaved_parallel.";
      } else {
        MS_LOG(ERROR) << name_ << ": The axis can not be split by interleaved_parallel.";
      }
      return FAILED;
    }
  }
  if (stra[axis_] != 1 && !skip_redistribution_) {
    if (is_in_layout_propagation_) {
      MS_LOG(INFO) << name_ << ": The axis can not be split";
    } else {
      MS_LOG(ERROR) << name_ << ": The axis can not be split";
    }
    return FAILED;
  }
  return SUCCESS;
}

Status SplitInfo::CheckOutputLayout() {
  if (output_infer_tensor_layout_.tensor_shape_before().array().empty()) {
    if (is_in_layout_propagation_) {
      MS_LOG(INFO) << "Parameter of output tensor layout for " << name_ << " is not allowed to be set by users.";
    } else {
      MS_LOG(ERROR) << "Parameter of output tensor layout for " << name_ << " is not allowed to be set by users.";
    }
    return FAILED;
  }
  MS_LOG(INFO) << name_ << ": Using output tensor layout infer by input tensor layout.";
  UpdateOutputTensorInfoForInterleaved();
  return SUCCESS;
}

Status SplitInfo::InferOutputTensorInfo() {
  for (size_t i = 0; i < outputs_shape_.size(); ++i) {
    auto output_infer_tensor_layout = inputs_tensor_info_[kIndex0].tensor_layout();
    TensorLayout output_infer_tensor_layout_new;
    output_infer_tensor_layout_new.InitFromExtendVector(output_infer_tensor_layout.device_arrangement_origin().array(),
                                                        output_infer_tensor_layout.tensor_map_before(),
                                                        outputs_shape_[i]);
    TensorInfo output_tensor_info(output_infer_tensor_layout_new);
    outputs_tensor_info_.push_back(output_tensor_info);
    output_infer_tensor_layout_ = output_infer_tensor_layout_new;
  }
  return SUCCESS;
}

void SplitInfo::UpdateOutputTensorInfoForInterleaved() {
  if (inputs_tensor_info_[kIndex0].tensor_layout().device_arrangement_interleaved().array().empty()) {
    return;
  }
  for (size_t i = 0; i < outputs_shape_.size(); ++i) {
    if (!outputs_tensor_info_[i].tensor_layout().device_arrangement_interleaved().array().empty()) {
      continue;
    }
    auto interleaved_num = ParallelContext::GetInstance()->fine_grained_micro_interleaved_size();
    auto output_dev_matrix = outputs_tensor_info_[i].tensor_layout().device_arrangement_origin().array();
    output_dev_matrix[output_dev_matrix.size() - 1] = interleaved_num;
    Arrangement out_device_arrangement_interleaved;
    out_device_arrangement_interleaved.Init(output_dev_matrix);
    auto new_tensor_layout = outputs_tensor_info_[i].tensor_layout();
    new_tensor_layout.set_device_arrangement_interleaved(out_device_arrangement_interleaved);
    TensorInfo new_output_tensor_info(new_tensor_layout);
    outputs_tensor_info_[i] = new_output_tensor_info;
  }
}

Status SplitInfo::ComputeReplaceGraphForInterleaved(const CNodePtr &cnode) {
  GenerateGraph gen_g = GenerateGraph(attrs_);
  if (gen_g.Init(cnode) != SUCCESS) {
    MS_LOG(ERROR) << name_ << "GenerateGraph Init failed";
    return FAILED;
  }
  auto interleaved_num = ParallelContext::GetInstance()->fine_grained_micro_interleaved_size();
  Attr output_nums_attr = {"output_nums", MakeValue(interleaved_num)};
  OperatorAttrs virtual_converter_begin_attrs = {output_nums_attr};
  auto virtual_converter_begin = gen_g.PushBack(
    {gen_g.NewOpInst(VIRTUAL_CONVERTER_BEGIN, virtual_converter_begin_attrs), gen_g.virtual_input_node()});
  std::vector<AnfNodePtr> virtual_converter_end_inputs_vector;
  std::vector<std::pair<AnfNodePtr, int64_t>> input_nodes = {std::make_pair(virtual_converter_begin, 1)};
  for (int64_t i = 0; i < interleaved_num; ++i) {
    auto tuple_get_item = gen_g.PushBack({gen_g.NewOpInst(TUPLE_GETITEM), virtual_converter_begin, CreatInt64Imm(i)});
    auto axis = CreatInt64Imm(SizeToLong(axis_));
    auto output_num = CreatInt64Imm(SizeToLong(outputs_shape_.size()));
    auto split = gen_g.PushBack({gen_g.NewOpInst(prim_name_), tuple_get_item, axis, output_num});
    virtual_converter_end_inputs_vector.push_back(split);
  }
  Attr input_nums_attr = {"input_nums", MakeValue(interleaved_num)};
  OperatorAttrs virtual_converter_end_attrs = {input_nums_attr};
  std::vector<AnfNodePtr> virtual_converter_end_inputs = {
    gen_g.NewOpInst(VIRTUAL_CONVERTER_END, virtual_converter_end_attrs)};
  std::copy(virtual_converter_end_inputs_vector.begin(), virtual_converter_end_inputs_vector.end(),
            std::back_inserter(virtual_converter_end_inputs));
  auto virtual_converter_end = gen_g.PushBack(virtual_converter_end_inputs);
  replace_graph_ = std::make_shared<std::pair<std::vector<std::pair<AnfNodePtr, int64_t>>, AnfNodePtr>>(
    std::make_pair(input_nodes, virtual_converter_end));
  return SUCCESS;
}

ReplaceGraphPtr SplitInfo::replace_graph(const CNodePtr &cnode) {
  if (inputs_tensor_info_[kIndex0].tensor_layout().IsInterleavedParallel()) {
    if (ComputeReplaceGraphForInterleaved(cnode) != SUCCESS) {
      MS_LOG_WITH_NODE(EXCEPTION, cnode) << name_ << " splitting micro interleaved failed.";
    }
    return replace_graph_;
  }
  return replace_graph_;
}

Status SplitInfo::SetCostUnderStrategy(const StrategyPtr &strategy) { return SetCostUnderStrategyBase(strategy); }

std::vector<StrategyPtr> SplitInfo::GenerateOpStrategies(int64_t stage_id) {
  Shape split_flag;
  for (size_t i = 0; i < inputs_shape_[0].size(); ++i) {
    if (i == axis_) {
      split_flag.push_back(0);
    } else {
      split_flag.push_back(1);
    }
  }

  Shapes splittable_input = {split_flag};
  Shapes tmp_inputs_shape = {inputs_shape_[0]};

  std::vector<StrategyPtr> sp_vector;
  if (GenerateStrategiesForIndependentInputs(stage_id, tmp_inputs_shape, splittable_input, &sp_vector) != SUCCESS) {
    MS_LOG_WITH_NODE(EXCEPTION, cnode_) << name_ << ": Generate strategies failed";
  }
  if (sp_vector.empty()) {
    MS_LOG_WITH_NODE(EXCEPTION, cnode_) << name_ << ": No available strategy";
  }

  return sp_vector;
}

std::shared_ptr<Strategies> SplitInfo::GenerateBatchStrategies() {
  if (GetAttrs() != SUCCESS) {
    MS_LOG_WITH_NODE(EXCEPTION, cnode_) << name_ << ": Get attr failed";
  }
  Dimensions input_strategy(inputs_shape_[0].size(), 1);
  // axis can't split
  if (inputs_shape_[0].size() > 1) {
    if (axis_ != 0) {
      input_strategy[0] = stage_device_size_;
    }
  }
  Strategies strategy_v = {input_strategy};
  return std::make_shared<Strategies>(strategy_v);
}

Status SplitInfo::InferAsLossDivisor() {
  if (!ParallelContext::GetInstance()->loss_repeated_mean()) {
    as_loss_divisor_ = 1;
    return SUCCESS;
  }

  if (outputs_tensor_map_.empty()) {
    MS_LOG(ERROR) << name_ << ": The outputs tensor map is empty.";
    return FAILED;
  }

  if (outputs_tensor_map_[0].empty()) {
    as_loss_divisor_ = stage_device_size_;
    MS_LOG(INFO) << name_ << ": The output is a scalar, use the dev size " << as_loss_divisor_ << ", loss divisor.";
    return SUCCESS;
  }

  as_loss_divisor_ = ComputeRepeatDeviceNumByTensorMap(dev_matrix_shape_, outputs_tensor_map_[0]);
  MS_LOG(INFO) << name_ << ": the dev matrix shape is " << ShapeToString(dev_matrix_shape_)
               << ", the output tensor map is " << ShapeToString(outputs_tensor_map_[0]) << ", loss divisor is "
               << as_loss_divisor_;
  return SUCCESS;
}

Status SplitInfo::InferAsLossDivisorByLayout() {
  if (!ParallelContext::GetInstance()->loss_repeated_mean()) {
    as_loss_divisor_ = 1;
    return SUCCESS;
  }

  if (outputs_tensor_info_.empty()) {
    MS_LOG(ERROR) << name_ << ": The outputs tensor info is empty.";
    return FAILED;
  }

  TensorMaps outputs_tensor_map = outputs_tensor_info_[0].tensor_layout().tensor_map_before();
  if (outputs_tensor_map.empty()) {
    MS_LOG(INFO) << name_ << ": out_dev_matrix_shape is empty";
    as_loss_divisor_ = stage_device_size_;
    MS_LOG(INFO) << name_ << ": The output is a scalar, use the dev size " << as_loss_divisor_ << ", loss divisor.";
    return SUCCESS;
  }

  auto out_dev_matrix_shape = outputs_tensor_info_[0].tensor_layout().device_arrangement_origin().array();
  if (out_dev_matrix_shape.empty()) {
    out_dev_matrix_shape = dev_matrix_shape_;
  }
  Shape squashed_tensor_map;
  for (const auto &tensor_map : outputs_tensor_map) {
    std::copy(tensor_map.begin(), tensor_map.end(), std::back_inserter(squashed_tensor_map));
  }
  as_loss_divisor_ = ComputeRepeatDeviceNumByTensorMap(out_dev_matrix_shape, squashed_tensor_map);
  MS_LOG(INFO) << name_ << ": the dev matrix shape is " << ShapeToString(out_dev_matrix_shape)
               << ", the output tensor map is " << ShapeToString(squashed_tensor_map) << ", loss divisor is "
               << as_loss_divisor_;
  return SUCCESS;
}

REGISTER(SplitInfo);
REGISTER(SplitWithSizeInfo);
REGISTER(SplitTensorInfo);
REGISTER(SplitVInfo);
}  // namespace parallel
}  // namespace mindspore
