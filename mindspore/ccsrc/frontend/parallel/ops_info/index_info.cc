/**
 * Copyright 2024-2025 Huawei Technologies Co., Ltd
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
#include "frontend/parallel/ops_info/index_info.h"
#include <algorithm>
#include <memory>
#include <utility>
#include <vector>
#include <functional>
#include "frontend/parallel/device_matrix.h"
#include "frontend/parallel/dynamic_creator.h"
#include "frontend/parallel/strategy.h"
#include "frontend/parallel/tensor_layout/tensor_redistribution.h"
#include "frontend/parallel/graph_util/generate_graph.h"
#include "include/utils/parallel_context.h"
#include "frontend/jit/ps/resource.h"
#include "frontend/parallel/dynamic_shape/dynamic_shape.h"

namespace mindspore {
namespace parallel {
constexpr size_t kInput = 0;
constexpr size_t kInputIndex = 1;
constexpr size_t validSizeOfInput = 2;
constexpr size_t validNumsOfIndex = 2;

constexpr size_t validSizeOfIndex = 1;
constexpr size_t validShardSizeOfIndex = 2;
constexpr size_t validShardElementSizeOfIndex = 1;
// 1.get attr
Status IndexInfo::GetAttrs() { return SUCCESS; }

// two index must have the same size and equals to validSizeOfIndex
// The value of indices must be within a valid range. eg:[0, shape_of_input)
Status IndexInfo::CheckIndex() {
  auto index_shape = inputs_shape_new_.at(kInputIndex);
  MS_EXCEPTION_IF_NULL(index_shape);
  if (index_shape->size() != validNumsOfIndex) {
    MS_LOG(ERROR) << name_ << ": The index_shape size not valid, only support 2, now:" << index_shape->size();
    return FAILED;
  }
  if ((index_shape->GetElement(0)->GetValue().size() != index_shape->GetElement(1)->GetValue().size()) ||
      (index_shape->GetElement(0)->GetValue().size() != validSizeOfIndex) ||
      (index_shape->GetElement(1)->GetValue().size() != validSizeOfIndex)) {
    MS_LOG(ERROR) << name_ << "index_shape 0 GetValue:" << index_shape->GetElement(0)->GetValue();  // shape of index 0
    MS_LOG(ERROR) << name_ << "index_shape 1 GetValue:" << index_shape->GetElement(1)->GetValue();  // shape of index 1
    return FAILED;
  }
  return SUCCESS;
}

// 2.checkstra
Status IndexInfo::CheckStrategy(const StrategyPtr &strategy) {
  MS_EXCEPTION_IF_NULL(strategy);
  auto input_data_shard_dim = strategy->GetInputNewDim();
  // check shape of input
  auto input_shape = inputs_shape_new_.at(kInput);
  MS_EXCEPTION_IF_NULL(input_shape);
  if ((input_data_shard_dim.empty()) || (input_shape->GetAllElements()[0].size() != validSizeOfInput)) {
    MS_LOG(ERROR) << name_ << ": The strategy is empty";
    return FAILED;
  }
  auto input_data_shard = input_data_shard_dim[0]->GetAllElements();
  MS_LOG(INFO) << name_ << ":input_data_shard_dim GetElement(0): "
               << input_data_shard_dim[0]->GetAllElements();  // 0 shard shape of input
  MS_LOG(INFO) << name_ << ":input_data_shard_dim GetElement(1): "
               << input_data_shard_dim[1]->GetAllElements();  // 1 shard shape of index
  // check strategy shape of index
  if ((input_data_shard_dim[1]->GetAllElements().size() != validShardSizeOfIndex) ||
      (input_data_shard_dim[1]->GetElement(0)->GetValue().size() != validShardElementSizeOfIndex) ||
      (input_data_shard_dim[1]->GetElement(1)->GetValue().size() != validShardElementSizeOfIndex)) {
    MS_LOG(ERROR) << name_ << ":input_data_shard_dim GetAllElements support 2, now: "
                  << input_data_shard_dim[1]->GetAllElements();
    MS_LOG(ERROR) << name_ << ":input_data_shard_dim GetAllElements support 1, now: "
                  << input_data_shard_dim[1]->GetElement(0)->GetValue().size();
    MS_LOG(ERROR) << name_ << ":input_data_shard_dim GetAllElements support 1, now: "
                  << input_data_shard_dim[1]->GetElement(0)->GetValue().size();
    return FAILED;
  }
  if (CheckIndex() != SUCCESS) {
    return FAILED;
  }
  MS_LOG(INFO) << name_ << ":input_shape GetAllElements size:" << input_shape->GetAllElements()[0].size();
  MS_LOG(INFO) << name_ << ":input_shape GetAllElements size:" << input_shape->GetAllElements()[0];  // shape of input
  // dynamic shape judge
  if (IsForwardDynamicShape() == true && IsDynamicShapes(input_shape->GetAllElements()) == true) {
    set_dynamic_shape_flag(true);
  }
  // check strategy shape of input
  if (CheckStrategyByVector(input_data_shard, input_shape->GetAllElements()) != SUCCESS) {
    MS_LOG(ERROR) << name_ << ":input_shape:" << input_shape->GetAllElements();
    MS_LOG(ERROR) << name_ << ":shard policy:" << input_data_shard;
    MS_LOG(ERROR) << name_ << ":CheckStrategy failed:";
    return FAILED;
  }
  if (dynamic_shape_flag() != true) {
    MS_LOG(INFO) << name_ << ": static shape ";
    // static shape
    MS_EXCEPTION_IF_ZERO("inputs_data_shard 1", input_data_shard[0].at(0));
    shard_input_data_height_ =
      (LongToInt(input_shape->GetElement(0)->GetValue().at(0)) / LongToInt(input_data_shard[0].at(0))) - 1;
    shard_input_data_weight_ =
      (LongToInt(input_shape->GetElement(0)->GetValue().at(1)) / LongToInt(input_data_shard[0].at(1))) - 1;
  } else {
    // dynamic shape, no need to calc shard_input_data_height_, use calc graph instead
    MS_LOG(INFO) << name_ << ": dynamic shape";
  }
  return SUCCESS;
}

// 3.devmatrix
Status IndexInfo::InferDevMatrixShape() {
  // the strategy is (a, b)
  // the dev matrix is (a, b)
  MS_EXCEPTION_IF_NULL(strategy_);
  auto stra = strategy_->GetInputNewDim();
  if (stra.empty()) {
    MS_LOG(ERROR) << name_ << "::InferDevMatrixShape::The strategy_ is empty";
    return FAILED;
  }
  // only input has shard stra, indices do not shard should be (1,)
  auto input_data_shard = (stra.at(0)->GetAllElements())[0];
  Shape common_shape;
  size_t input_data_shard_size = input_data_shard.size();
  MS_LOG(INFO) << name_ << ":InferDevMatrixShape input_data_shard size: " << input_data_shard.size();
  MS_LOG(INFO) << name_ << ":InferDevMatrixShape input_data_shard 0: " << input_data_shard.at(0);  // 0 shard of input 0
  MS_LOG(INFO) << name_ << ":InferDevMatrixShape input_data_shard 1: " << input_data_shard.at(1);  // 1 shard of input 1
  /* all input */
  for (size_t i = 0; i < input_data_shard_size; i++) {
    common_shape.push_back(input_data_shard.at(i));
  }
  dev_matrix_shape_.clear();
  dev_matrix_shape_ = common_shape;
  // should not include repeated num
  origin_dev_matrix_shape_ = dev_matrix_shape_;
  // output dev_matrix
  out_dev_matrix_shape_ = dev_matrix_shape_;
  MS_LOG(INFO) << name_ << ":InferDevMatrixShape dev matrix: " << ShapeToString(dev_matrix_shape_);
  MS_LOG(INFO) << name_ << ":inputs_shape_new_[0]: " << inputs_shape_new_[0]->GetValue();
  return SUCCESS;
}

// tensor map
void IndexInfo::SetOptionalInputTensorMap(const size_t &index, size_t *valid_input_index) {
  MS_EXCEPTION_IF_NULL(valid_input_index);
  if (input_value_[index] != nullptr && !input_value_[index]->isa<None>()) {
    MS_EXCEPTION_IF_NULL(inputs_shape_new_[*valid_input_index]);
    auto input_shape = inputs_shape_new_[*valid_input_index]->GetElement(0);
    Shape nosplit_tensor_map_idx;
    if (index == kInputIndex) {
      for (size_t i = 0; i < input_shape->size(); i++) {
        nosplit_tensor_map_idx.emplace_back(-1);  // {-1}
      }
    } else {
      MS_EXCEPTION(ShapeError) << "op [" << name_
                               << " ] set infer tensor map error. Current input_value_ size: " << input_value_.size()
                               << ", inputs_shape_new_ size: " << inputs_shape_new_.size()
                               << ". Current input_value_ idx is: " << index
                               << ", inputs_shape_new_ index is: " << *valid_input_index << ". inputs_shape_new_ ["
                               << *valid_input_index << "] size is: " << input_shape->size();
    }
    std::vector<ShapeBasePtr> nosplit_tensorlist_map_idx;
    for (size_t i = 0; i < inputs_shape_new_[*valid_input_index]->size(); i++) {
      nosplit_tensorlist_map_idx.emplace_back(std::make_shared<ShapeValue>(nosplit_tensor_map_idx));
    }
    inputs_tensor_map_new_.emplace_back(std::make_shared<ShapeList>(nosplit_tensorlist_map_idx));
    (*valid_input_index)++;
  }
  MS_LOG(INFO) << name_ << ":inputs_tensor_map_new_ size now: " << inputs_tensor_map_new_.size();
  MS_LOG(INFO) << name_ << ":inputs_tensor_map_new_[0]: " << inputs_tensor_map_new_.at(0)->GetAllElements();
  MS_LOG(INFO) << name_ << ":inputs_tensor_map_new_[1]: " << inputs_tensor_map_new_.at(1)->GetAllElements();
}

// 4.TensorMap
Status IndexInfo::InferTensorMap() {
  auto size = origin_dev_matrix_shape_.size();
  Shape input_tensor_map_idx;
  Shape out_tensor_map_idx;

  for (size_t i = size - 1; i >= 1; i--) {
    input_tensor_map_idx.emplace_back(i);
  }
  input_tensor_map_idx.emplace_back(0);
  for (size_t i = 0; i < outputs_shape_new_[0]->GetElement(0)->size(); i++) {
    out_tensor_map_idx.emplace_back(-1);
  }
  MS_LOG(INFO) << name_ << ":input_tensor_map now: " << input_tensor_map_idx;
  (void)inputs_tensor_map_new_.emplace_back(std::make_shared<ShapeValue>(input_tensor_map_idx));  // input tensor map
  (void)outputs_tensor_map_new_.emplace_back(std::make_shared<ShapeValue>(out_tensor_map_idx));   // output tensor map
  MS_LOG(INFO) << name_ << ":inputs_tensor_map_new_[0]: " << inputs_tensor_map_new_.at(0)->GetAllElements();
  MS_LOG(INFO) << name_ << ":outputs_tensor_map_new_: " << outputs_tensor_map_new_.at(0)->GetAllElements();
  // any input need tensor map
  size_t valid_input_index = kInputIndex;
  SetOptionalInputTensorMap(kInputIndex, &valid_input_index);
  return SUCCESS;
}

Status IndexInfo::InferAsLossDivisor() {
  if (!ParallelContext::GetInstance()->loss_repeated_mean()) {
    as_loss_divisor_ = 1;
    return SUCCESS;
  }
  if (outputs_tensor_map_new_.empty()) {
    MS_LOG(ERROR) << name_ << ": outputs_tensor_map_new_ empty.";
    return FAILED;
  }
  if (out_dev_matrix_shape_.empty()) {
    MS_LOG(ERROR) << name_ << ": out_dev_matrix_shape_ empty.";
    out_dev_matrix_shape_ = dev_matrix_shape_;
  }
  as_loss_divisor_ =
    ComputeRepeatDeviceNumByTensorMap(out_dev_matrix_shape_, outputs_tensor_map_new_[0]->GetAllElements()[0]);
  MS_LOG(INFO) << name_ << ": the dev matrix shape is " << ShapeToString(out_dev_matrix_shape_)
               << ", the output tensor map is " << ShapeToString(outputs_tensor_map_new_[0]->GetAllElements()[0])
               << ", loss divisor is " << as_loss_divisor_;
  return SUCCESS;
}

// only support params_size = 2 and index_size = 2
Status IndexInfo::InferBias() {
  CheckGlobalDeviceManager();
  auto input_shape = inputs_shape_new_.at(kInput);
  MS_EXCEPTION_IF_NULL(input_shape);
  auto stra = strategy_->GetInputNewDim();
  auto input_data_shard = (stra.at(0)->GetAllElements())[0];
  std::vector<int64_t> inputs_shape_tmp;
  std::vector<int64_t> inputs_data_shard_tmp;
  if (input_shape->GetAllElements()[0].size() != input_data_shard.size()) {
    MS_LOG(ERROR) << name_ << ":input_shape size: " << input_shape->GetAllElements()[0].size()
                  << "not equals to input_data_shard size: " << input_data_shard.size();
    return FAILED;
  }
  for (size_t i = 0; i < input_shape->GetAllElements()[0].size(); i++) {
    bias_.push_back(1);
    slice_size_.push_back(0);
    inputs_shape_tmp.push_back(input_shape->GetAllElements()[0].at(i));
    inputs_data_shard_tmp.push_back(input_data_shard.at(i));
    MS_LOG(INFO) << name_ << ":InferBias push back: " << i;
  }
  int64_t rank = g_device_manager->rank_index_in_stage();
  if (input_shape->GetAllElements()[0].size() == validSizeOfInput) {
    if (repeated_calc_num_ > 1) {
      // bias_0 = rank / (b * r)
      bias_.at(0) = (rank / (inputs_data_shard_tmp.at(1) * repeated_calc_num_));
      // bias_1 = rank % (b * r) / r
      bias_.at(1) = ((rank % (inputs_data_shard_tmp.at(1) * repeated_calc_num_)) / repeated_calc_num_);
    } else {
      // bias_0 = rank / (b * c * d) % a = rank / (b * 1 * 1) % a
      bias_.at(0) = (rank / (inputs_data_shard_tmp.at(1))) % inputs_data_shard_tmp.at(0);
      // bias_1 = rank % (b * c * d) / (c * d) = rank % (b * 1 * 1) / (1 * 1)
      bias_.at(1) = rank % (inputs_data_shard_tmp.at(1));
    }

    if (dynamic_shape_flag() == true) {
      // A/B needs to be calculated by graph.
      return SUCCESS;
    }
    MS_EXCEPTION_IF_ZERO("inputs_data_shard 0", inputs_data_shard_tmp.at(0));
    MS_EXCEPTION_IF_ZERO("inputs_data_shard 1", inputs_data_shard_tmp.at(1));
    slice_size_.at(0) = inputs_shape_tmp.at(0) / (inputs_data_shard_tmp.at(0));  // A/a
    slice_size_.at(1) = inputs_shape_tmp.at(1) / (inputs_data_shard_tmp.at(1));  // B/b
    bias_.at(0) = bias_.at(0) * slice_size_.at(0);
    bias_.at(1) = bias_.at(1) * slice_size_.at(1);
    MS_LOG(INFO) << name_ << ":rank num: " << rank << " bias_0::" << bias_.at(0) << " bias_1::" << bias_.at(1);
    MS_LOG(INFO) << name_ << ":rank num: " << rank << " slice_size_0::" << slice_size_.at(0)
                 << " slice_size_1::" << slice_size_.at(1);
    MS_LOG(INFO) << name_ << ":rank num: " << rank << " inputs_shape_tmp_0::" << inputs_shape_tmp.at(0)
                 << " inputs_shape_tmp_1::" << inputs_shape_tmp.at(1);
    MS_LOG(INFO) << name_ << ":rank num: " << rank << " shard_tmp_0::" << inputs_data_shard_tmp.at(0)
                 << " shard_tmp_1::" << inputs_data_shard_tmp.at(1);
    return SUCCESS;
  }
  MS_LOG(ERROR) << name_ << ":Only support input_shape size 2. Now:" << input_shape->GetElement(0)->GetValue().size();
  return FAILED;
}

Status IndexInfo::SetCostUnderStrategy(const StrategyPtr &strategy) { return SetCostUnderStrategyBase(strategy); }
RankList IndexInfo::GetAllReduceRankList() {
  RankList reduce_rank_list;
  RankList reduce_rank_list_test;
  std::vector<int64_t> dims;
  if (!inputs_tensor_map_new_.at(0)->GetAllElements().empty()) {
    // strategy is set.
    MS_LOG(INFO) << "index input device_arrangement:" << dev_matrix_shape_;  // 4 1 2
    MS_LOG(INFO) << "index input tensor_map:" << inputs_tensor_map_new_.at(0)->GetAllElements();
    Shape in_tensor_map = inputs_tensor_map_new_.at(0)->GetValue();

    for (size_t i = 0; i < in_tensor_map.size(); ++i) {
      dims.push_back(SizeToLong(dev_matrix_shape_.size() - kIndex1 - in_tensor_map[i]));
    }

    auto device_matrix =
      DeviceMatrix(g_device_manager->global_rank(), g_device_manager->GetDeviceListInThisStage(), dev_matrix_shape_);
    device_matrix.GetDevicesAlongMultiDim(dims, &reduce_rank_list);
    for (const auto &element : dims) {
      MS_LOG(INFO) << "index dims ele:" << element;
    }
    for (const auto &element : reduce_rank_list) {
      MS_LOG(INFO) << "index reduce_rank_list ele (current allreduce ele):" << element;
    }
    for (const auto &element : dev_matrix_shape_) {
      MS_LOG(INFO) << "index dev_matrix_shape_ ele):" << element;
    }
    MS_LOG(INFO) << "repeated_calc_num_:" << repeated_calc_num_;
    return reduce_rank_list;
  }
  MS_LOG(EXCEPTION) << "index input tensor map empty.";
}

std::string IndexInfo::InferGroup() {
  auto reduce_rank_list = GetAllReduceRankList();
  Group comm_group;
  if (g_device_manager->CreateGroup(reduce_rank_list, &comm_group) != SUCCESS) {
    MS_LOG(EXCEPTION) << "InferGroup Create comm group failed in index";
  }
  std::string group_name = comm_group.name();
  return group_name;
}

ReplaceGraphPtr IndexInfo::ReplaceGraphDynamicShape(const CNodePtr &cnode) {
  GenerateGraph gen_g = GenerateGraph(attrs_);
  if (gen_g.Init(cnode) != SUCCESS) {
    MS_LOG_WITH_NODE(EXCEPTION, cnode) << name_ << "GenerateGraph Init failed";
  }
  auto tuple_get_item_0 =
    gen_g.PushBack({gen_g.NewOpInst(TUPLE_GETITEM), gen_g.virtual_input_node(), CreatInt64Imm(0)});
  auto tuple_get_item_1 =
    gen_g.PushBack({gen_g.NewOpInst(TUPLE_GETITEM), gen_g.virtual_input_node(), CreatInt64Imm(1)});
  auto shape_of_input = gen_g.PushBack({gen_g.NewOpInst(SHAPE_OP), gen_g.virtual_input_node()});
  auto dtype_input = gen_g.PushBack({gen_g.NewOpInst(DTYPE), tuple_get_item_0});
  auto dtype_id_input = gen_g.PushBack(
    {gen_g.NewOpInst(DTYPETOENUM), CreateStringImm("DtypeToEnum"), CreateStringImm("dtype"), dtype_input});
  auto slice_size_0 = gen_g.PushBack({gen_g.NewOpInst(TUPLE_GETITEM), shape_of_input, CreatInt64Imm(0)});
  auto slice_size_1 = gen_g.PushBack({gen_g.NewOpInst(TUPLE_GETITEM), shape_of_input, CreatInt64Imm(1)});
  auto bias_0 = gen_g.PushBack({gen_g.NewOpInst(SCALAR_MUL), slice_size_0, CreatInt64Imm(bias_.at(0))});
  auto bias_1 = gen_g.PushBack({gen_g.NewOpInst(SCALAR_MUL), slice_size_1, CreatInt64Imm(bias_.at(1))});
  auto bias_0_tensor = gen_g.PushBack({gen_g.NewOpInst(SCALAR_TO_TENSOR), bias_0, dtype_id_input});
  auto bias_1_tensor = gen_g.PushBack({gen_g.NewOpInst(SCALAR_TO_TENSOR), bias_1, dtype_id_input});

  auto slice_size_for_min_0 = gen_g.PushBack({gen_g.NewOpInst(SCALAR_SUB), slice_size_0, CreatInt64Imm(1)});
  auto slice_size_for_min_1 = gen_g.PushBack({gen_g.NewOpInst(SCALAR_SUB), slice_size_1, CreatInt64Imm(1)});
  auto slice_size_0_tensor = gen_g.PushBack({gen_g.NewOpInst(SCALAR_TO_TENSOR), slice_size_for_min_0, dtype_id_input});
  auto slice_size_1_tensor = gen_g.PushBack({gen_g.NewOpInst(SCALAR_TO_TENSOR), slice_size_for_min_1, dtype_id_input});
  // index_0 process, keepalive is required if sub 0.
  OperatorAttrs keep_alive_attr = {std::make_pair(KEEP_ALIVE, MakeValue(true))};
  auto sub_0 = gen_g.PushBack({gen_g.NewOpInst(SUB, keep_alive_attr), tuple_get_item_0, bias_0_tensor});
  auto relu_0 = gen_g.PushBack({gen_g.NewOpInst(RELU), sub_0});
  auto minimum_0 = gen_g.PushBack({gen_g.NewOpInst(MINIMUM), relu_0, slice_size_0_tensor});
  auto equal_0 = gen_g.PushBack({gen_g.NewOpInst(EQUAL), sub_0, minimum_0});
  // index_1 process, keepalive is required if sub 0.
  auto sub_1 = gen_g.PushBack({gen_g.NewOpInst(SUB, keep_alive_attr), tuple_get_item_1, bias_1_tensor});
  auto relu_1 = gen_g.PushBack({gen_g.NewOpInst(RELU), sub_1});
  auto minimum_1 = gen_g.PushBack({gen_g.NewOpInst(MINIMUM), relu_1, slice_size_1_tensor});
  auto equal_1 = gen_g.PushBack({gen_g.NewOpInst(EQUAL), sub_1, minimum_1});
  auto index_tuple = gen_g.PushBack({gen_g.NewOpInst(MAKE_TUPLE_OP), minimum_0, minimum_1});
  auto index = gen_g.PushBack({gen_g.NewOpInst(INDEX_INFO), gen_g.virtual_input_node(), index_tuple});
  // valid MASK process
  auto logicaland = gen_g.PushBack({gen_g.NewOpInst(LOGICALAND), equal_0, equal_1});
  // valid MASK process cast
  auto dtype_index = gen_g.PushBack({gen_g.NewOpInst(DTYPE), index});
  auto dtype_id_index = gen_g.PushBack(
    {gen_g.NewOpInst(DTYPETOENUM), CreateStringImm("DtypeToEnum"), CreateStringImm("dtype"), dtype_index});
  auto valid_mask_cast = gen_g.PushBack({gen_g.NewOpInst(CAST), logicaland, dtype_id_index});
  // valid MASK reshape // don't need expand dim
  auto mul = gen_g.PushBack({gen_g.NewOpInst(MUL), index, valid_mask_cast});

  std::string group_name = InferGroup();
  MS_LOG(INFO) << "Rmsnorm allreduce group: " << group_name;

  Attr attr_op = std::make_pair(OP, MakeValue(REDUCE_OP_SUM));
  Attr attr_group = std::make_pair(GROUP, MakeValue(group_name));
  OperatorAttrs attrs = {attr_op, attr_group};
  // Then aggregate all segments with AllReduce.
  auto index_output_node = gen_g.PushBack({gen_g.NewOpInst(ALL_REDUCE, attrs), mul});
  std::vector<std::pair<AnfNodePtr, int64_t>> input_nodes = {
    std::make_pair(index, 1), std::make_pair(shape_of_input, 1), std::make_pair(tuple_get_item_0, 2),
    std::make_pair(tuple_get_item_1, 2)};
  replace_graph_ = std::make_shared<std::pair<std::vector<std::pair<AnfNodePtr, int64_t>>, AnfNodePtr>>(
    std::make_pair(input_nodes, index_output_node));
  return replace_graph_;
}

ReplaceGraphPtr IndexInfo::replace_graph(const CNodePtr &cnode) {
  if (InferBias() != SUCCESS) {
    MS_LOG_WITH_NODE(EXCEPTION, cnode) << name_ << ": Infer Bias failed.";
  }
  if (dynamic_shape_flag() == true) {
    return ReplaceGraphDynamicShape(cnode);
  }
  GenerateGraph gen_g = GenerateGraph(attrs_);
  if (gen_g.Init(cnode) != SUCCESS) {
    MS_LOG_WITH_NODE(EXCEPTION, cnode) << name_ << "GenerateGraph Init failed";
  }
  auto tuple_get_item_0 =
    gen_g.PushBack({gen_g.NewOpInst(TUPLE_GETITEM), gen_g.virtual_input_node(), CreatInt64Imm(0)});
  auto tuple_get_item_1 =
    gen_g.PushBack({gen_g.NewOpInst(TUPLE_GETITEM), gen_g.virtual_input_node(), CreatInt64Imm(1)});
  // index_0 process, keepalive is required if sub 0.
  OperatorAttrs keep_alive_attr = {
    std::make_pair(KEEP_ALIVE, MakeValue(true))};  // In sub 0 scenario, keepalive is required.
  MS_LOG(INFO) << name_ << ": The rank is " << g_device_manager->rank_index_in_stage() << ", the bias is " << 0;

  auto dtype_index_input = gen_g.PushBack({gen_g.NewOpInst(DTYPE), tuple_get_item_0});
  auto dtype_id_index_input = gen_g.PushBack(
    {gen_g.NewOpInst(DTYPETOENUM), CreateStringImm("DtypeToEnum"), CreateStringImm("dtype"), dtype_index_input});
  auto bias_0 = gen_g.PushBack({gen_g.NewOpInst(CAST), CreateInt32Tensor(bias_.at(0), true), dtype_id_index_input});
  auto bias_1 = gen_g.PushBack({gen_g.NewOpInst(CAST), CreateInt32Tensor(bias_.at(1), true), dtype_id_index_input});
  auto shard_input_height =
    gen_g.PushBack({gen_g.NewOpInst(CAST), CreateInt32Tensor(shard_input_data_height_, true), dtype_id_index_input});
  auto shard_input_weight =
    gen_g.PushBack({gen_g.NewOpInst(CAST), CreateInt32Tensor(shard_input_data_weight_, true), dtype_id_index_input});

  auto sub_0 = gen_g.PushBack({gen_g.NewOpInst(SUB, keep_alive_attr), tuple_get_item_0, bias_0});
  auto relu_0 = gen_g.PushBack({gen_g.NewOpInst(RELU), sub_0});
  auto minimum_0 = gen_g.PushBack({gen_g.NewOpInst(MINIMUM), relu_0, shard_input_height});
  auto equal_0 = gen_g.PushBack({gen_g.NewOpInst(EQUAL), sub_0, minimum_0});
  // index_1 process, keepalive is required if sub 0.
  auto sub_1 = gen_g.PushBack({gen_g.NewOpInst(SUB, keep_alive_attr), tuple_get_item_1, bias_1});
  auto relu_1 = gen_g.PushBack({gen_g.NewOpInst(RELU), sub_1});
  auto minimum_1 = gen_g.PushBack({gen_g.NewOpInst(MINIMUM), relu_1, shard_input_weight});
  auto equal_1 = gen_g.PushBack({gen_g.NewOpInst(EQUAL), sub_1, minimum_1});

  auto index_tuple = gen_g.PushBack({gen_g.NewOpInst(MAKE_TUPLE_OP), minimum_0, minimum_1});
  auto index = gen_g.PushBack({gen_g.NewOpInst(INDEX_INFO), gen_g.virtual_input_node(), index_tuple});
  // valid MASK process
  auto logicaland = gen_g.PushBack({gen_g.NewOpInst(LOGICALAND), equal_0, equal_1});
  // valid MASK process cast
  auto dtype = gen_g.PushBack({gen_g.NewOpInst(DTYPE), index});
  auto dtype_id =
    gen_g.PushBack({gen_g.NewOpInst(DTYPETOENUM), CreateStringImm("DtypeToEnum"), CreateStringImm("dtype"), dtype});
  auto valid_mask_cast = gen_g.PushBack({gen_g.NewOpInst(CAST), logicaland, dtype_id});

  // valid MASK reshape // don't need expand dim
  auto mul = gen_g.PushBack({gen_g.NewOpInst(MUL), index, valid_mask_cast});

  std::string group_name = InferGroup();
  MS_LOG(INFO) << "index allreduce group: " << group_name;

  Attr attr_op = std::make_pair(OP, MakeValue(REDUCE_OP_SUM));
  Attr attr_group = std::make_pair(GROUP, MakeValue(group_name));
  OperatorAttrs attrs = {attr_op, attr_group};
  // Then aggregate all segments with AllReduce.
  auto index_output_node = gen_g.PushBack({gen_g.NewOpInst(ALL_REDUCE, attrs), mul});
  std::vector<std::pair<AnfNodePtr, int64_t>> input_nodes = {
    std::make_pair(index, 1), std::make_pair(tuple_get_item_0, 2), std::make_pair(tuple_get_item_1, 2)};
  replace_graph_ = std::make_shared<std::pair<std::vector<std::pair<AnfNodePtr, int64_t>>, AnfNodePtr>>(
    std::make_pair(input_nodes, index_output_node));
  return replace_graph_;
}

REGISTER(IndexInfo);
}  // namespace parallel
}  // namespace mindspore
