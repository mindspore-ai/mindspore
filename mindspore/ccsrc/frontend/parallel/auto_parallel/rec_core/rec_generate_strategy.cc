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

#include "frontend/parallel/auto_parallel/rec_core/rec_generate_strategy.h"

#include <algorithm>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <vector>

#include "frontend/parallel/auto_parallel/rec_core/rec_parse_graph.h"
#include "frontend/parallel/auto_parallel/rec_core/rec_partition.h"
#include "frontend/parallel/ops_info/flash_attention_score_info.h"
#include "frontend/parallel/ops_info/operator_info.h"
#include "frontend/parallel/ops_info/strided_slice_info.h"
#include "frontend/parallel/ops_info/gather_info.h"
#include "frontend/parallel/parameter_manager.h"
#include "frontend/parallel/step_parallel.h"
#include "frontend/parallel/step_parallel_utils.h"
#include "frontend/parallel/strategy.h"
#include "include/common/utils/utils.h"
#include "ir/value.h"
#include "mindspore/core/ops/ops_func_impl/flash_attention_score.h"
#include "ops/op_enum.h"

namespace mindspore {
namespace parallel {
namespace {
using PrepareStraFuncPtr = Strategies (*)(const std::shared_ptr<OperatorInfo> &, Dimensions, bool);
std::map<std::string, PrepareStraFuncPtr> g_prepare_stra_map;

std::optional<bool> GetKeepDimsFromAttrs(const std::shared_ptr<OperatorInfo> &op) {
  auto keep_dims_iter = op->attrs().find(KEEP_DIMS);
  if (keep_dims_iter == op->attrs().end()) {
    return std::nullopt;
  }
  auto keep_dims_ptr = keep_dims_iter->second;
  MS_EXCEPTION_IF_NULL(keep_dims_ptr);
  if (!keep_dims_ptr->isa<BoolImm>()) {
    MS_LOG(EXCEPTION) << op->name() << ": Keep_dims is not a bool.";
  }
  auto keepdims = keep_dims_ptr->cast<BoolImmPtr>()->value();
  return keepdims;
}

std::optional<bool> GetKeepDimsFromInputs(const std::shared_ptr<OperatorInfo> &op) {
  auto keep_dims_opt = GetScalarValueFromInputs<bool>(op->input_value(), op->name(), KEEP_DIMS);
  return keep_dims_opt;
}

bool GetKeepDims(const std::shared_ptr<OperatorInfo> &op) {
  auto keep_dims_opt = GetKeepDimsFromAttrs(op);
  if (!keep_dims_opt.has_value()) {
    keep_dims_opt = GetKeepDimsFromInputs(op);
  }
  if (!keep_dims_opt.has_value()) {
    MS_LOG(EXCEPTION) << op->name() << ": Don't have attr keep_dims.";
  }
  auto keepdims = keep_dims_opt.value();
  return keepdims;
}

Dimensions GetDimList(const std::shared_ptr<OperatorInfo> &op) {
  Dimensions dim_list;
  bool keep_dims = GetKeepDims(op);
  if (keep_dims) {
    return dim_list;
  }

  const auto &name = op->name();
  auto dim_list_opt = GetArrayValueFromInputs<int64_t>(op->input_value(), name, AXIS);
  if (!dim_list_opt.has_value()) {
    MS_LOG(EXCEPTION) << "For " << name << ", failed to get value for " << AXIS << ".";
  }

  dim_list = dim_list_opt.value();
  auto x_dim = op->inputs_shape()[0].size();
  // axis is (), reduce all dim
  if (dim_list.empty()) {
    for (size_t i = 0; i < x_dim; ++i) {
      dim_list.push_back(SizeToLong(i));
    }
  } else {
    auto AxisCorrectFunc = [x_dim](const int64_t axis) {
      if (axis < 0) {
        return axis + SizeToLong(x_dim);
      }
      return axis;
    };
    std::transform(dim_list.begin(), dim_list.end(), dim_list.begin(), AxisCorrectFunc);
  }
  return dim_list;
}
}  // namespace

size_t OpNameToId(const std::vector<std::shared_ptr<OperatorInfo>> &ops, const std::shared_ptr<OperatorInfo> &op) {
  for (size_t i = 0; i < ops.size(); ++i) {
    if (ops[i]->name() == op->name()) {
      return i;
    }
  }

  return SIZE_MAX;
}

bool IsDimensionsFlat(const Dimensions &dims) {
  return !std::any_of(dims.begin(), dims.end(), [](const int64_t &dim) { return dim != 1; });
}

bool IsDimensionsEmpty(const Dimensions &dims) { return dims.empty(); }

bool IsStrategyFlat(const StrategyPtr &str) {
  const auto &input_dims = str->GetInputDim();
  return !std::any_of(input_dims.begin(), input_dims.end(),
                      [](const Dimensions &dims) { return !IsDimensionsFlat(dims); });
}

size_t DevicesForDimensions(const Dimensions &dims) {
  return std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<size_t>());
}

bool HasStrategy(std::shared_ptr<OperatorInfo> op) {
  StrategyPtr s_strategy = op->selected_strategy();
  if (s_strategy != nullptr && !s_strategy->ToString().empty()) {
    return true;
  }
  return false;
}

size_t FindIndexOfOperatorIncoming(const std::vector<std::shared_ptr<OperatorInfo>> &ops,
                                   const std::vector<std::vector<std::string>> &input_tensor_names, size_t iter_ops) {
  size_t incoming_op_index = SIZE_MAX;
  for (size_t i = 1; i < input_tensor_names[iter_ops].size(); i++) {
    for (size_t j = 0; j < input_tensor_names.size(); j++) {
      if (input_tensor_names[iter_ops][i] == input_tensor_names[j][0]) {
        incoming_op_index = j;
        break;
      }
    }
    if (incoming_op_index != SIZE_MAX && HasStrategy(ops.at(incoming_op_index)) &&
        !IsStrategyFlat(ops.at(incoming_op_index)->selected_strategy())) {
      break;
    }
  }
  if (incoming_op_index != SIZE_MAX &&
      ops.at(incoming_op_index)->name().find(VIRTUALDATASETINFO) != std::string::npos) {
    return SIZE_MAX;
  }
  return incoming_op_index;
}

std::pair<size_t, size_t> FindIndexOfOperatorOutgoing(const std::vector<std::shared_ptr<OperatorInfo>> &ops,
                                                      const std::vector<std::vector<std::string>> &input_tensor_names,
                                                      const size_t iter_ops) {
  bool found = false;
  size_t outgoing_op_index = SIZE_MAX;
  size_t iter_op_inputs = SIZE_MAX;

  for (size_t i = 0; i < input_tensor_names.size(); i++) {
    for (size_t j = 1; j < input_tensor_names[i].size(); j++) {
      if (input_tensor_names[i][j] == input_tensor_names[iter_ops][0] &&
          ops[i]->selected_strategy()->GetInputNumber() != 0) {
        outgoing_op_index = i;
        iter_op_inputs = std::min(j - 1, ops[outgoing_op_index]->inputs_shape().size() - 1);
        found = true;
        break;
      }
    }
    if (found) {
      break;
    }
  }

  std::pair<size_t, size_t> res = std::make_pair(outgoing_op_index, iter_op_inputs);

  return res;
}

int64_t GetGatherAxis(const std::shared_ptr<OperatorInfo> &op) {
  auto axis_input = GetValue<int64_t>(op->input_value().at(2));
  if (axis_input < 0) {
    axis_input += SizeToLong(op->inputs_shape()[0].size());
  }
  if (axis_input >= SizeToLong(op->inputs_shape()[0].size())) {
    MS_LOG(EXCEPTION) << "Failure: Gather's axis out of range.";
  }
  return axis_input;
}

int64_t GetGatherBatchDims(const std::shared_ptr<OperatorInfo> &op) {
  int64_t batch_dims = -1;
  auto batch_dims_val = GetScalarValueFromInputs<int64_t>(op->input_value(), op->name(), BATCH_DIMS);
  if (batch_dims_val.has_value()) {
    batch_dims = batch_dims_val.value();
  } else {
    MS_LOG(EXCEPTION) << op->name() << ": Failed to fetch the value of batch dims";
  }
  return batch_dims;
}

void ReverseRemainingList(const std::shared_ptr<std::vector<size_t>> &no_stra_op_list) {
  MS_LOG(INFO) << "ReverseRemainingList";
  std::reverse(no_stra_op_list->begin(), no_stra_op_list->end());
}

void GenerateStrategy(const std::shared_ptr<Graph> &graph, const std::vector<std::shared_ptr<OperatorInfo>> &ops,
                      const std::shared_ptr<std::vector<std::vector<size_t>>> &eli_list,
                      const std::vector<std::vector<std::string>> &input_tensor_names,
                      const std::shared_ptr<std::vector<size_t>> &index_list, bool is_training,
                      const std::vector<std::vector<size_t>> &param_users_ops_index, const FuncGraphPtr &root) {
  RecStrategyPropagator propagator(graph, ops, eli_list, input_tensor_names, index_list, is_training,
                                   param_users_ops_index, root);

  if (g_device_manager->DeviceNum() > SIZE_THIRTY_TWO) {
    propagator.ExtraShardMatmulOnBatchDim();
  }
  if (is_training) {
    propagator.GenerateStrategyV3();
  } else {
    propagator.GenerateStrategyV1();
  }
}

void FillFlashLayoutIndexes(const std::shared_ptr<FlashAttentionScoreInfo> &flashOp, size_t *batch_split_idx,
                            size_t *n_split_idx, size_t *s_split_idx) {
  MS_EXCEPTION_IF_NULL(flashOp);
  MS_EXCEPTION_IF_NULL(batch_split_idx);
  MS_EXCEPTION_IF_NULL(n_split_idx);
  MS_EXCEPTION_IF_NULL(s_split_idx);

  size_t tmp_batch_split_idx;
  size_t tmp_n_split_idx;
  size_t tmp_s_split_idx;

  using mindspore::ops::FASInputLayoutMode;
  switch (flashOp->input_layout()) {
    case FASInputLayoutMode::BSH:
    case FASInputLayoutMode::BSND:
      tmp_batch_split_idx = kIndex0;
      tmp_s_split_idx = kIndex1;
      tmp_n_split_idx = kIndex2;
      break;
    case FASInputLayoutMode::BNSD:
      tmp_batch_split_idx = kIndex0;
      tmp_n_split_idx = kIndex1;
      tmp_s_split_idx = kIndex2;
      break;
    case FASInputLayoutMode::SBH:
      tmp_s_split_idx = kIndex0;
      tmp_batch_split_idx = kIndex1;
      tmp_n_split_idx = kIndex2;
      break;
    default:
      MS_LOG(EXCEPTION) << flashOp->name() << "unknown input_layout: " << flashOp->input_layout();
  }

  *batch_split_idx = tmp_batch_split_idx;
  *n_split_idx = tmp_n_split_idx;
  *s_split_idx = tmp_s_split_idx;
}

Strategies PrepareFlashAttentionScore(const std::shared_ptr<OperatorInfo> &op, Dimensions basic_stra,
                                      bool dyn_shape_tmp_fix) {
  std::shared_ptr<FlashAttentionScoreInfo> flashOp = std::static_pointer_cast<FlashAttentionScoreInfo>(op);

  if (flashOp->InitAttrs() != SUCCESS) {
    MS_LOG(EXCEPTION) << flashOp->name() << " : InitAttrs failed.";
  }

  Strategies expect_strategies = Strategies(ops::kFlashAttentionScoreInputsNum);
  auto is_input_passed = flashOp->is_input_passed();

  size_t batch_idx;
  size_t n_split_idx;
  size_t s_split_idx;

  FillFlashLayoutIndexes(flashOp, &batch_idx, &n_split_idx, &s_split_idx);

  int64_t batch_split_num = basic_stra[batch_idx];
  int64_t s1_split_num = basic_stra[s_split_idx];
  int64_t n1_split_num = basic_stra[n_split_idx];
  int64_t n2_split_num = flashOp->kv_split() ? n1_split_num : 1;

  Dimensions q_stra(op->inputs_shape()[ops::kFlashAttentionScoreInputQueryIndex].size(), 1);
  q_stra[batch_idx] = batch_split_num;
  q_stra[s_split_idx] = s1_split_num;
  q_stra[n_split_idx] = n1_split_num;

  Dimensions kv_stra(op->inputs_shape()[ops::kFlashAttentionScoreInputKeyIndex].size(), 1);
  kv_stra[batch_idx] = batch_split_num;
  kv_stra[n_split_idx] = n2_split_num;

  expect_strategies[ops::kFlashAttentionScoreInputQueryIndex] = q_stra;
  expect_strategies[ops::kFlashAttentionScoreInputKeyIndex] = kv_stra;
  expect_strategies[ops::kFlashAttentionScoreInputValueIndex] = kv_stra;

  if (is_input_passed[ops::kFlashAttentionScoreInputRealShiftIndex]) {
    int64_t real_shift_s1_split_num = flashOp->real_shift_have_s1_dim() ? s1_split_num : 1;
    int64_t real_shift_batch_split_num = flashOp->real_shift_have_batch_dim() ? batch_split_num : 1;
    expect_strategies[ops::kFlashAttentionScoreInputRealShiftIndex] = {real_shift_batch_split_num, n1_split_num,
                                                                       real_shift_s1_split_num, 1};
  }

  if (is_input_passed[ops::kFlashAttentionScoreInputDropMaskIndex]) {
    expect_strategies[ops::kFlashAttentionScoreInputDropMaskIndex] = {batch_split_num, n1_split_num, s1_split_num, 1};
  }

  if (is_input_passed[ops::kFlashAttentionScoreInputPaddingMaskIndex]) {
    expect_strategies[ops::kFlashAttentionScoreInputPaddingMaskIndex] = {};
  }

  if (is_input_passed[ops::kFlashAttentionScoreInputAttnMaskIndex]) {
    auto attn_mask_shape =
      flashOp->inputs_shape().at(flashOp->GetStrategyRealIndex(ops::kFlashAttentionScoreInputAttnMaskIndex));
    int64_t s1_split_num_attn_mask = flashOp->is_attn_mask_compressed() ? 1 : s1_split_num;
    if (attn_mask_shape.size() == kSizeTwo) {
      // attn_mask_shape: (S1, S2)
      expect_strategies[ops::kFlashAttentionScoreInputAttnMaskIndex] = {s1_split_num_attn_mask, 1};
    } else if (attn_mask_shape.size() == kSizeFour) {
      // attn_mask_shape: (B, N1, S1, S2) or (B, 1, S1, S2)
      auto attn_mask_n1_split_num = flashOp->attn_mask_have_n1_dim() ? n1_split_num : 1;
      auto attn_batch_split_num = flashOp->attn_mask_have_batch_dim() ? batch_split_num : 1;
      expect_strategies[ops::kFlashAttentionScoreInputAttnMaskIndex] = {attn_batch_split_num, attn_mask_n1_split_num,
                                                                        s1_split_num_attn_mask, 1};
    }
  }

  if (is_input_passed[ops::kFlashAttentionScoreInputPrefixIndex]) {
    expect_strategies[ops::kFlashAttentionScoreInputPrefixIndex] = {batch_split_num};
  }

  if (is_input_passed[ops::kFlashAttentionScoreInputActualSeqQlenIndex]) {
    expect_strategies[ops::kFlashAttentionScoreInputActualSeqQlenIndex] = {NO_SPLIT_STRATEGY};
  }
  if (is_input_passed[ops::kFlashAttentionScoreInputActualSeqKVlenIndex]) {
    expect_strategies[ops::kFlashAttentionScoreInputActualSeqKVlenIndex] = {NO_SPLIT_STRATEGY};
  }

  expect_strategies.erase(std::remove(expect_strategies.begin(), expect_strategies.end(), Shape{}),
                          expect_strategies.end());
  return expect_strategies;
}

Strategies PrepareFillV2(const std::shared_ptr<OperatorInfo> &op, Dimensions basic_stra, bool dyn_shape_tmp_fix) {
  Strategies strategies;

  if (op->outputs_shape().size() == 0) {
    MS_LOG(EXCEPTION) << op->name() << " output tensor info is empty.";
  }

  for (size_t i = basic_stra.size(); i < op->outputs_shape()[0].size(); i++) {
    basic_stra.push_back(1);
  }

  strategies.push_back(basic_stra);
  basic_stra.clear();
  strategies.push_back(basic_stra);
  return strategies;
}

Dimensions PrepareMatMulStrategy(Graph::NodeType *node, bool transpose_a, bool transpose_b, size_t iter_op_inputs) {
  Dimensions strategy;
  if (transpose_a && (iter_op_inputs == 0)) {
    strategy.push_back(static_cast<int64_t>(1.0 / node->apply.arguments[iter_op_inputs].tensor_str.str_w));
    strategy.push_back(static_cast<int64_t>(1.0 / node->apply.arguments[iter_op_inputs].tensor_str.str_h));
  } else if (transpose_b && (iter_op_inputs == 1)) {
    strategy.push_back(static_cast<int64_t>(1.0 / node->apply.arguments[iter_op_inputs].tensor_str.str_w));
    strategy.push_back(static_cast<int64_t>(1.0 / node->apply.arguments[iter_op_inputs].tensor_str.str_h));
  } else {
    strategy.push_back(static_cast<int64_t>(1.0 / node->apply.arguments[iter_op_inputs].tensor_str.str_h));
    strategy.push_back(static_cast<int64_t>(1.0 / node->apply.arguments[iter_op_inputs].tensor_str.str_w));
  }
  return strategy;
}

Strategies PrepareMatMul(Graph::NodeType *node, const std::shared_ptr<OperatorInfo> &op) {
  Strategies strategies;
  auto input_value = op->input_value();
  bool transpose_a = input_value[2]->cast<BoolImmPtr>()->value();
  bool transpose_b = input_value[3]->cast<BoolImmPtr>()->value();

  for (size_t iter_op_inputs = 0; iter_op_inputs < op->inputs_shape().size(); iter_op_inputs++) {
    Dimensions strategy = PrepareMatMulStrategy(node, transpose_a, transpose_b, iter_op_inputs);
    strategies.push_back(strategy);
  }
  return strategies;
}

Strategies PreparePropagateBatchMatMul(const std::shared_ptr<OperatorInfo> &op, Dimensions basic_stra,
                                       bool dyn_shape_tmp_fix) {
  if (dyn_shape_tmp_fix) {
    return CheckDivisible(op, basic_stra);
  }
  // This backward propagation does NOT complete strategy on k. Could be done later
  Strategies stra;
  auto input_value = op->input_value();
  bool transpose_a = input_value[2]->cast<BoolImmPtr>()->value();
  bool transpose_b = input_value[3]->cast<BoolImmPtr>()->value();

  size_t first_input_size = op->inputs_shape()[0].size();
  size_t second_input_size = op->inputs_shape()[1].size();

  Dimensions first_input_dim(first_input_size);
  Dimensions second_input_dim(second_input_size);

  // first input
  if (!transpose_a) {
    first_input_dim[first_input_size - 1] = 1;                                  // k axis
    first_input_dim[first_input_size - 2] = basic_stra[basic_stra.size() - 2];  // i axis
  } else {
    first_input_dim[first_input_size - 2] = 1;                                  // k axis
    first_input_dim[first_input_size - 1] = basic_stra[basic_stra.size() - 2];  // i axis
  }

  for (size_t idx = 3; idx <= first_input_size; idx++) {
    first_input_dim[first_input_size - idx] = basic_stra[basic_stra.size() - idx];
  }

  // second input
  if (!transpose_b) {
    second_input_dim[second_input_size - 2] = 1;                                  // k axis
    second_input_dim[second_input_size - 1] = basic_stra[basic_stra.size() - 1];  // j axis
  } else {
    second_input_dim[second_input_size - 1] = 1;                                  // k axis
    second_input_dim[second_input_size - 2] = basic_stra[basic_stra.size() - 1];  // j axis
  }

  for (size_t idx = 3; idx <= second_input_size; idx++) {
    second_input_dim[second_input_size - idx] = basic_stra[basic_stra.size() - idx];
  }

  stra.push_back(first_input_dim);
  stra.push_back(second_input_dim);
  return stra;
}

Dimensions PrepareBatchMatMulStrategy(Graph::NodeType *node, const bool transpose_a, const bool transpose_b,
                                      const size_t iter_op_inputs, const size_t dim_num) {
  if (node->apply.arguments[iter_op_inputs].tensor_str.str_n == 0 ||
      node->apply.arguments[iter_op_inputs].tensor_str.str_c == 0 ||
      node->apply.arguments[iter_op_inputs].tensor_str.str_h == 0 ||
      node->apply.arguments[iter_op_inputs].tensor_str.str_w == 0) {
    MS_LOG(EXCEPTION) << "The strategy is 0";
  }

  Dimensions strategy;
  if (dim_num >= SIZE_FOUR) {
    strategy.push_back(static_cast<int64_t>(1.0 / node->apply.arguments[iter_op_inputs].tensor_str.str_n));
  }
  if (dim_num >= SIZE_THREE) {
    strategy.push_back(static_cast<int64_t>(1.0 / node->apply.arguments[iter_op_inputs].tensor_str.str_c));
  }
  if (transpose_a && (iter_op_inputs == 0)) {
    strategy.push_back(static_cast<int64_t>(1.0 / node->apply.arguments[iter_op_inputs].tensor_str.str_w));
    strategy.push_back(static_cast<int64_t>(1.0 / node->apply.arguments[iter_op_inputs].tensor_str.str_h));
  } else if (transpose_b && (iter_op_inputs == 1)) {
    strategy.push_back(static_cast<int64_t>(1.0 / node->apply.arguments[iter_op_inputs].tensor_str.str_w));
    strategy.push_back(static_cast<int64_t>(1.0 / node->apply.arguments[iter_op_inputs].tensor_str.str_h));
  } else {
    strategy.push_back(static_cast<int64_t>(1.0 / node->apply.arguments[iter_op_inputs].tensor_str.str_h));
    strategy.push_back(static_cast<int64_t>(1.0 / node->apply.arguments[iter_op_inputs].tensor_str.str_w));
  }
  return strategy;
}

Strategies PrepareBatchMatMul(Graph::NodeType *node, const std::shared_ptr<OperatorInfo> &op) {
  Strategies strategies;
  auto input_value = op->input_value();
  bool transpose_a = input_value[2]->cast<BoolImmPtr>()->value();
  bool transpose_b = input_value[3]->cast<BoolImmPtr>()->value();

  for (size_t iter_op_inputs = 0; iter_op_inputs < op->inputs_shape().size(); iter_op_inputs++) {
    Dimensions strategy = PrepareBatchMatMulStrategy(node, transpose_a, transpose_b, iter_op_inputs,
                                                     op->inputs_shape()[iter_op_inputs].size());
    strategies.push_back(strategy);
  }
  return strategies;
}

Strategies PrepareBiasAdd(const std::shared_ptr<OperatorInfo> &op, Dimensions basic_stra, bool dyn_shape_tmp_fix) {
  auto strategy = std::make_shared<Dimensions>(basic_stra);
  Strategies strategies;
  strategies.push_back(*strategy);
  Dimensions s_biasadd;
  s_biasadd.push_back(strategy->at(1));
  strategies.push_back(s_biasadd);
  return strategies;
}

Strategies PrepareStandAlone(const std::shared_ptr<OperatorInfo> &op) {
  Strategies strategies;
  Dimensions strategy;

  for (size_t i = 0; i < op->outputs_tensor_info().size(); i++) {
    strategy.clear();
    for (size_t j = 0; j < op->inputs_tensor_info()[i].shape().size(); j++) {
      strategy.push_back(1);
    }
    strategies.push_back(strategy);
  }

  return strategies;
}

Strategies PrepareDataParallel(const std::shared_ptr<OperatorInfo> &op, Dimensions basic_stra, bool dyn_shape_tmp_fix) {
  size_t numDev = g_device_manager->stage_device_num();

  Strategies strategies;
  Dimensions strategy;

  if (numDev == 0) {
    MS_LOG(EXCEPTION) << "The number of devices is 0";
  }

  for (size_t i = 0; i < op->inputs_shape().size(); i++) {
    strategy.clear();
    if (LongToSize(op->inputs_shape()[i][0]) % numDev == 0) {
      strategy.push_back(numDev);
    } else {
      strategy.push_back(1);
    }
    for (size_t j = 1; j < op->inputs_shape()[i].size(); j++) {
      strategy.push_back(1);
    }
    strategies.push_back(strategy);
  }

  return strategies;
}

Dimensions PrepareOneHotOutputStrategy(const std::shared_ptr<OperatorInfo> &op) {
  auto op_strategy = op->selected_strategy();
  Dimensions strategy;

  for (size_t i = 0; i < static_cast<size_t>(op->inputs_shape().size()); i++) {
    if (op->inputs_shape()[i].size() == 0) {
      continue;
    }
    // copy the full strategy (Assume strategy has the same size as the following operator input shape)
    for (size_t j = 0; j < op_strategy->GetInputDim().at(i).size(); ++j) {
      strategy.push_back(op_strategy->GetInputDim().at(i).at(j));
    }
    break;
  }
  return strategy;
}

Strategies PrepareStridedSlice(const std::shared_ptr<OperatorInfo> &op, Dimensions basic_stra, bool dyn_shape_tmp_fix) {
  Strategies strategies;

  if (dyn_shape_tmp_fix) {
    return strategies;
  }

  auto strided_slice = std::static_pointer_cast<StridedSliceInfo>(op);
  strided_slice->GetAttrs();
  auto begin = strided_slice->begin();
  auto strides = strided_slice->strides();
  auto new_axis_mask_bitmap = strided_slice->new_axis_mask_bitmap();
  auto fully_fetch_flag = strided_slice->fully_fetch_flag();
  auto skip_redistribution = strided_slice->skip_redistribution();

  Shape strategy_in_process = Shape(basic_stra.size(), 0);
  for (size_t i = 0; i < new_axis_mask_bitmap.size() && i < begin.size() && i < basic_stra.size(); ++i) {
    if (new_axis_mask_bitmap[i]) {
      strategy_in_process[i] = 1;
    }
  }

  size_t count = 0;
  for (auto &ele : strategy_in_process) {
    if (ele != 0) {
      continue;
    }
    ele = basic_stra[count];
    count++;
  }

  (void)strategy_in_process.insert(strategy_in_process.end(), basic_stra.begin() + count, basic_stra.end());
  MS_LOG(INFO) << op->name() << ": The strategy in process is " << strategy_in_process;

  for (size_t j = 0; j < strides.size(); ++j) {
    if ((strides[j] != 1) && (strategy_in_process[j] > 1)) {
      strategy_in_process[j] = 1;
    }
  }

  for (size_t k = 0; k < begin.size(); ++k) {
    if (!fully_fetch_flag[k] && (strategy_in_process[k] != 1) && !skip_redistribution) {
      strategy_in_process[k] = 1;
    }
  }

  strategies.push_back(strategy_in_process);
  return strategies;
}

std::vector<int64_t> FindAxisProperty(const std::shared_ptr<OperatorInfo> &op) {
  std::vector<int64_t> axis_list;
  string axis_name = AXIS;
  auto input_value = op->input_value();

  auto op_name = op->name();

  if (input_value[input_value.size() - 1]->isa<ValueSequence>()) {  // Softmax axis is a tuple
    std::optional<std::vector<int64_t>> axis_opt = GetArrayValueFromInputs<int64_t>(input_value, op_name, axis_name);
    std::vector<int64_t> axis_val = axis_opt.value();
    if (axis_opt.has_value()) {
      axis_list.swap(axis_val);
    } else {
      axis_list.push_back(-1);
    }
  } else {  // LogSoftmax axis is a scaler
    std::optional<int64_t> axis_opt = GetScalarValueFromInputs<int64_t>(input_value, op_name, axis_name);
    int64_t axis_val = axis_opt.value();
    if (axis_opt.has_value()) {
      axis_list.push_back(axis_val);
    } else {
      axis_list.push_back(-1);
    }
  }
  return axis_list;
}

Strategies PrepareSoftMax(const std::shared_ptr<OperatorInfo> &op, Dimensions basic_stra, bool dyn_shape_tmp_fix) {
  Strategies strategies;
  strategies.push_back(basic_stra);
  std::vector<int64_t> axis_list = FindAxisProperty(op);

  for (auto &axis : axis_list) {
    if (axis < 0) {
      int64_t input_dim = SizeToLong(op->inputs_shape()[0].size());
      axis = input_dim + axis;
    }
    if (axis >= SizeToLong(strategies[0].size()) || axis < 0) {
      MS_LOG(EXCEPTION) << op->name() << ": axis value is out of range.";
    }
    if (strategies[0][LongToSize(axis)] != 1) {
      strategies[0][LongToSize(axis)] = 1;
      MS_LOG(INFO) << op->name() << ": adjust strategy to 1 on axis " << axis;
    }
  }

  // Strategy protection to avoid that partition number is larger than the shape of related dimension.
  for (size_t i = 0; i < op->inputs_shape().size(); i++) {
    for (size_t j = 0; j < op->inputs_shape()[i].size(); j++) {
      if (strategies[i][j] > op->inputs_shape()[i][j] || op->inputs_shape()[i][j] % strategies[i][j] != 0) {
        strategies[i][j] = 1;
      }
    }
  }

  return strategies;
}

Strategies PrepareLayerNorm(const std::shared_ptr<OperatorInfo> &op, Dimensions basic_stra, bool dyn_shape_tmp_fix) {
  Strategies strategies;
  strategies.push_back(basic_stra);
  std::vector<int64_t> axis_list;
  string axis_name = AXIS;

  auto iter = op->attrs().find(axis_name);
  if (iter != op->attrs().end()) {
    MS_EXCEPTION_IF_NULL(iter->second);
    if (iter->second->isa<Int64Imm>()) {
      axis_list.push_back(iter->second->cast<Int64ImmPtr>()->value());
    } else if (iter->second->isa<ValueTuple>()) {
      ValueTuplePtr value_tuple = iter->second->cast<ValueTuplePtr>();
      if (value_tuple == nullptr) {
        MS_LOG(EXCEPTION) << op->name() << ": The value_tuple is nullptr.";
      }

      std::vector<ValuePtr> value_vector = value_tuple->value();
      (void)std::transform(value_vector.begin(), value_vector.end(), std::back_inserter(axis_list),
                           [](const ValuePtr &value) { return static_cast<int64_t>(GetValue<int64_t>(value)); });
    } else {
      MS_LOG(EXCEPTION) << op->name() << ": The value of axis is not int64_t or tuple int64_t.";
    }
  } else {
    axis_list.push_back(-1);
  }

  for (auto &axis : axis_list) {
    if (axis < 0) {
      int64_t input_dim = SizeToLong(op->inputs_shape()[0].size());
      axis = input_dim + axis;
    }
    if (axis >= SizeToLong(strategies[0].size()) || axis < 0) {
      MS_LOG(EXCEPTION) << op->name() << ": axis value is out of range.";
    }
    if (strategies[0][LongToSize(axis)] != 1) {
      strategies[0][LongToSize(axis)] = 1;
      MS_LOG(INFO) << op->name() << ": adjust strategy to 1 on axis " << axis;
    }
  }
  Dimensions d = {1};
  strategies.push_back(d);
  strategies.push_back(d);
  return strategies;
}

Strategies PrepareRmsNorm(const std::shared_ptr<OperatorInfo> &op, Dimensions basic_stra, bool dyn_shape_tmp_fix) {
  Strategies strategies;
  auto inputs = op->inputs_shape();
  auto input = inputs[0];
  Shape strategy_in_process = Shape(input.size(), 1);
  int64_t devices = SizeToLong(g_device_manager->DeviceNum());

  if (parallel::ParallelContext::GetInstance()->pipeline_stage_split_num() == 0) {
    MS_LOG(EXCEPTION) << "divisors cannot be 0!";
  }
  int64_t max_cut = devices / parallel::ParallelContext::GetInstance()->pipeline_stage_split_num();
  strategy_in_process[0] = input[0] < max_cut ? input[0] : max_cut;

  auto gamma = inputs[1];
  size_t gamma_diff = input.size() - gamma.size();
  Dimensions gamma_strategy;
  for (size_t j = 0; j < gamma.size(); ++j) {
    gamma_strategy.push_back(strategy_in_process[gamma_diff + j]);
  }

  strategies.push_back(strategy_in_process);
  strategies.push_back(gamma_strategy);
  return strategies;
}

Strategies PrepareOneHot(const std::shared_ptr<OperatorInfo> &op, Dimensions strategy, bool dyn_shape_tmp_fix) {
  Strategies strategies;

  // OneHot's strategy depends on its output shape.
  for (size_t i = strategy.size(); i < op->outputs_shape()[0].size(); i++) {
    strategy.push_back(1);
  }

  // Partition number should not exceed the number of devices
  for (size_t i = 0; i < op->outputs_shape()[0].size(); i++) {
    if (strategy[i] > op->outputs_shape()[0][i]) {
      strategy[i] = 1;
    }
  }

  strategies.push_back(strategy);

  // Push two empty Dimensions for the other two input tensors.
  Dimensions s_empty = {};
  strategies.push_back(s_empty);
  strategies.push_back(s_empty);

  return strategies;
}

Dimensions GenGatherStra(Shape targeted_shape) {
  Dimensions index(targeted_shape.size() - 1, 0);
  for (size_t i = 0; i < index.size(); i++) {
    index[i] = SizeToLong(i);
  }

  std::sort(index.begin(), index.end(), [&targeted_shape](const size_t &a, const size_t &b) {
    return (targeted_shape[a + 1] > targeted_shape[b + 1]);
  });
  (void)std::transform(std::begin(index), std::end(index), std::begin(index), [](int64_t x) { return x + 1; });
  (void)index.insert(index.cbegin(), 0);

  Dimensions strategie(targeted_shape.size(), 1);

  size_t num_device = LongToSize(g_device_manager->stage_device_num());
  size_t cut = 1;
  for (size_t i = 0; i < index.size(); i++) {
    size_t index_i = LongToSize(index[i]);
    while (targeted_shape[index_i] % SIZE_TWO == 0 && targeted_shape[index_i] > 0 && cut < num_device) {
      targeted_shape[index_i] /= SIZE_TWO;
      cut *= SIZE_TWO;
      strategie[index_i] *= SIZE_TWO;  // We apply 2-parts partitioning for Gather.
    }
    if (cut == num_device) {
      break;
    }
  }

  return strategie;
}

Strategies GatherForDynamicShape(const std::shared_ptr<OperatorInfo> &op, const size_t dim) {
  Strategies strategies;
  auto gather_input_0_shape = op->inputs_shape()[0];
  if (dim >= gather_input_0_shape.size()) {
    MS_LOG(EXCEPTION) << "Failure: Gather's axis out of range.";
  }
  Dimensions gather_input_0_strategy(gather_input_0_shape.size(), 1);
  int64_t num_device = g_device_manager->stage_device_num();
  if (gather_input_0_shape[dim] % num_device == 0) {
    size_t cut = 1;
    while (gather_input_0_shape[dim] > 0 && gather_input_0_shape[dim] % SIZE_TWO == 0 && cut < LongToSize(num_device)) {
      gather_input_0_shape[dim] /= SIZE_TWO;
      cut *= SIZE_TWO;
      gather_input_0_strategy[dim] *= SIZE_TWO;
    }
  }
  strategies.push_back(gather_input_0_strategy);
  for (size_t i = 1; i < op->inputs_shape().size(); i++) {
    Dimensions gather_input_i_strategy(op->inputs_shape()[i].size(), 1);
    strategies.push_back(gather_input_i_strategy);
  }
  return strategies;
}

Strategies PrepareGather(const std::shared_ptr<OperatorInfo> &op, Dimensions strategy, bool dyn_shape_tmp_fix) {
  if (dyn_shape_tmp_fix) {
    Strategies strategies;
    strategies.push_back(strategy);
    for (size_t i = 1; i < op->inputs_shape().size(); i++) {
      Dimensions gather_input_i_strategy(op->inputs_shape()[i].size(), 1);
      strategies.push_back(gather_input_i_strategy);
    }
    return strategies;
  }

  Strategies strategies;
  Shape targeted_shape = op->outputs_shape()[0];
  Dimensions strategie = GenGatherStra(targeted_shape);

  int64_t axis = GetGatherAxis(op);
  MS_LOG(INFO) << op->name() << ": the axis is " << axis;

  int64_t batch_dims = GetGatherBatchDims(op);
  MS_LOG(INFO) << op->name() << ": the batch_dims is " << batch_dims;

  if (batch_dims > 1) {
    for (size_t i = 0; i < op->inputs_shape().size(); i++) {
      strategies.push_back(strategie);
    }
    strategies[0][axis] = 1;
    return strategies;
  }

  strategy.clear();
  if (axis == 0) {
    Shape param_strategy = Shape(op->inputs_shape()[0].size(), 1);
    Shape indices_strategy = Shape(op->inputs_shape()[1].size(), 1);
    strategies.push_back(param_strategy);
    strategies.push_back(indices_strategy);
    size_t num_device = LongToSize(g_device_manager->stage_device_num());
    size_t cut = 1;
    int gather_inputs_num = SizeToInt(op->inputs_shape().size());
    for (int i = gather_inputs_num - 1; i >= 0; --i) {
      auto tensor_shape = op->inputs_shape()[i];
      while (tensor_shape[0] % SIZE_TWO == 0 && tensor_shape[0] > 0 && cut < num_device) {
        tensor_shape[0] /= SIZE_TWO;
        cut *= SIZE_TWO;
        strategies[i][0] *= SIZE_TWO;  // We apply 2-parts partitioning for Gather.
      }
      if (cut == num_device) {
        break;
      }
    }
  } else if (axis == 1) {
    strategy.push_back(strategie[0]);
    strategy.push_back(1);
    strategies.push_back(strategy);
    strategy.clear();
    for (size_t i = 0; i < op->inputs_shape()[1].size(); i++) {
      strategy.push_back(strategie[op->inputs_shape()[0].size() - 1 + i]);
    }
    strategies.push_back(strategy);
  } else {
    MS_LOG(EXCEPTION) << "Failure: Normal Gather's axis is neither 0 nor 1.";
  }

  auto gather = std::static_pointer_cast<GatherInfo>(op);
  auto gather_mode = gather->GetGatherMode(strategies[0], strategies[1]);
  MS_LOG(INFO) << op->name() << ": the gather_mode is " << gather_mode;
  if (gather_mode == SHARD_AXIS_0_DYNAMIC || gather_mode == SHARD_AXIS_0_STATIC || gather_mode == SHARD_AXIS_1) {
    if (DevicesForDimensions(strategies[1]) != 1 && strategies[0][axis] != 1) {
      strategies[0][axis] = 1;
      MS_LOG(INFO) << op->name() << ": param_strategy[" << axis << "] is changed to 1.";
    }
  }

  return strategies;
}

Dimensions PrepareGatherV2OutputStrategy(const std::shared_ptr<OperatorInfo> &op) {
  auto targeted_shape = op->outputs_shape()[0];
  Dimensions strategie = GenGatherStra(targeted_shape);
  return strategie;
}

Strategies PrepareL2Normalize(const std::shared_ptr<OperatorInfo> &op, Dimensions strategy, bool dyn_shape_tmp_fix) {
  int64_t axis = 0;
  auto iter = op->attrs().find(AXIS);
  if (iter != op->attrs().end()) {
    MS_EXCEPTION_IF_NULL(iter->second);
    if (iter->second->isa<ValueSequence>()) {
      axis = GetValue<std::vector<int64_t>>(iter->second)[0];
    } else {
      MS_LOG(EXCEPTION) << op->name() << " : The value of axis is not int64_t.";
    }
  }

  int64_t axis_index = axis;
  if (axis < 0) {
    size_t input_dim = op->inputs_shape()[0].size();
    axis_index = static_cast<int64_t>(input_dim) + axis;
  }

  strategy[LongToSize(axis_index)] = 1;

  Strategies strategies;
  strategies.push_back(strategy);
  return strategies;
}

Strategies PrepareAxisRelatedStrategy(Graph::NodeType *node, const std::vector<std::shared_ptr<OperatorInfo>> &ops,
                                      const size_t iter_ops) {
  Strategies strategies = MakeRecSearchStrategy(node, ops, iter_ops);
  if (strategies.size() < 1) {
    MS_LOG(EXCEPTION) << ops[iter_ops]->name() << ": get empty Strategy.";
  }

  std::vector<int64_t> axis_list;
  string axis_name = AXIS;
  int64_t default_axis = -1;
  if (ops[iter_ops]->type() == LAYER_NORM) {
    axis_name = "begin_norm_axis";
    default_axis = 1;
  }

  auto iter = ops[iter_ops]->attrs().find(axis_name);
  if (iter != ops[iter_ops]->attrs().end()) {
    MS_EXCEPTION_IF_NULL(iter->second);
    if (iter->second->isa<Int64Imm>()) {
      axis_list.push_back(iter->second->cast<Int64ImmPtr>()->value());
    } else if (iter->second->isa<ValueTuple>()) {
      ValueTuplePtr value_tuple = iter->second->cast<ValueTuplePtr>();
      if (value_tuple == nullptr) {
        MS_LOG(EXCEPTION) << ops[iter_ops]->name() << ": The value_tuple is nullptr.";
      }
      std::vector<ValuePtr> value_vector = value_tuple->value();
      (void)std::transform(value_vector.begin(), value_vector.end(), std::back_inserter(axis_list),
                           [](const ValuePtr &value) { return static_cast<int64_t>(GetValue<int64_t>(value)); });
    } else {
      MS_LOG(EXCEPTION) << ops[iter_ops]->name() << ": The value of axis is not int64_t or tuple int64_t.";
    }
  } else {
    axis_list.push_back(default_axis);
  }

  for (auto &axis : axis_list) {
    if (axis < 0) {
      int64_t input_dim = SizeToLong(ops[iter_ops]->inputs_shape()[0].size());
      axis = input_dim + axis;
    }
    if (axis >= SizeToLong(strategies[0].size()) || axis < 0) {
      MS_LOG(EXCEPTION) << ops[iter_ops]->name() << ": axis value is out of range.";
    }
    if (strategies[0][LongToSize(axis)] != 1) {
      strategies[0][LongToSize(axis)] = 1;
      MS_LOG(INFO) << ops[iter_ops]->name() << ": adjust strategy to 1 on axis " << axis;
    }
  }
  return strategies;
}

Strategies MakeRecSearchStrategy(Graph::NodeType *node, const std::vector<std::shared_ptr<OperatorInfo>> &ops,
                                 const size_t iter_ops) {
  if (ops.empty()) {
    MS_LOG(EXCEPTION) << "Failure: Operators is empty.";
  }
  if (iter_ops >= ops.size()) {
    MS_LOG(EXCEPTION) << "Failure: Operators' elements out of range.";
  }
  if (node->apply.op_type == kRecUnsortedSegmentOp) {
    return MakeDataParallelStrategy(node, ops, iter_ops);
  }

  Strategies strategies;
  for (size_t iter_op_inputs = 0; iter_op_inputs < ops[iter_ops]->inputs_shape().size(); iter_op_inputs++) {
    if (iter_op_inputs >= ops[iter_ops]->inputs_shape().size()) {
      MS_LOG(EXCEPTION) << "Failure: Strategy's InputDim out of range.";
    }

    size_t input_size = ops[iter_ops]->inputs_shape()[iter_op_inputs].size();
    Dimensions strategy;
    if (input_size == SIZE_FOUR) {
      strategy.push_back(static_cast<int64_t>(1.0 / node->apply.arguments[iter_op_inputs].tensor_str.str_n));
      strategy.push_back(static_cast<int64_t>(1.0 / node->apply.arguments[iter_op_inputs].tensor_str.str_c));
      strategy.push_back(static_cast<int64_t>(1.0 / node->apply.arguments[iter_op_inputs].tensor_str.str_h));
      strategy.push_back(static_cast<int64_t>(1.0 / node->apply.arguments[iter_op_inputs].tensor_str.str_w));
    } else if (input_size == SIZE_THREE) {
      // Experimental support for 3D data.
      strategy.push_back(static_cast<int64_t>(1.0 / node->apply.arguments[iter_op_inputs].tensor_str.str_c));
      strategy.push_back(static_cast<int64_t>(1.0 / node->apply.arguments[iter_op_inputs].tensor_str.str_h));
      strategy.push_back(static_cast<int64_t>(1.0 / node->apply.arguments[iter_op_inputs].tensor_str.str_w));
    } else if (input_size == SIZE_TWO) {
      strategy.push_back(static_cast<int64_t>(1.0 / node->apply.arguments[iter_op_inputs].tensor_str.str_h));
      strategy.push_back(static_cast<int64_t>(1.0 / node->apply.arguments[iter_op_inputs].tensor_str.str_w));
    } else if (input_size == SIZE_ONE) {
      strategy.push_back(static_cast<int64_t>(1.0 / node->apply.arguments[iter_op_inputs].tensor_str.str_w));
    } else if (input_size == SIZE_ZERO) {
      strategy = {};
    } else {
      MS_LOG(EXCEPTION) << ops[iter_ops]->name() << ": Tensor's input size is unexcepted.";
    }
    strategies.push_back(strategy);
  }
  return strategies;
}

Strategies MakeDataParallelStrategy(Graph::NodeType *node, const std::vector<std::shared_ptr<OperatorInfo>> &ops,
                                    const size_t iter_ops) {
  if (ops.empty()) {
    MS_LOG(EXCEPTION) << "Failure: Operators is empty.";
  }
  if (iter_ops >= ops.size()) {
    MS_LOG(EXCEPTION) << "Failure: Operators' elements out of range.";
  }

  Strategies strategies;
  size_t max_device_num = LongToSize(g_device_manager->stage_device_num());
  size_t target_tensor_batch = LongToUlong(ops[iter_ops]->inputs_shape()[0][0]);
  for (size_t iter_op_inputs = 0; iter_op_inputs < ops[iter_ops]->inputs_shape().size(); iter_op_inputs++) {
    if (iter_op_inputs >= ops[iter_ops]->inputs_shape().size()) {
      MS_LOG(EXCEPTION) << "Failure: Strategy's InputDim out of range.";
    }

    Dimensions strategy;
    size_t input_size = ops[iter_ops]->inputs_shape()[iter_op_inputs].size();
    for (size_t dim = 0; dim < input_size; dim++) {
      // Experimental support for 3D data (input_size == 3).
      if (input_size >= SIZE_ONE && input_size <= STR_DIM_NUM) {
        if (dim == 0) {
          strategy.push_back(std::min(max_device_num, target_tensor_batch));
        } else {
          strategy.push_back(1);
        }
      } else if (input_size == 0) {
        strategy = {};
      } else {
        MS_LOG(EXCEPTION) << ops[iter_ops]->name() << ": Tensor shape " << input_size << " is unexpected.";
      }
    }
    strategies.push_back(strategy);
  }
  // Set default strategy.
  node->tensor_parm.tensor_str.str_n = 1.0;
  node->tensor_parm.tensor_str.str_c = 1.0;
  node->tensor_parm.tensor_str.str_h = 1.0;
  node->tensor_parm.tensor_str.str_w = 1.0;

  // Update data parallel strategy.
  if (ops[iter_ops]->outputs_shape().size() == SIZE_ZERO) {
    MS_LOG(EXCEPTION) << ops[iter_ops]->name() << " output tensor info is empty.";
  }
  if (ops[iter_ops]->outputs_shape()[0].size() == SIZE_ONE) {
    node->tensor_parm.tensor_str.str_w = 1.0 / std::min(max_device_num, target_tensor_batch);
  } else if (ops[iter_ops]->outputs_shape()[0].size() == SIZE_TWO) {
    node->tensor_parm.tensor_str.str_h = 1.0 / std::min(max_device_num, target_tensor_batch);
  } else if (ops[iter_ops]->outputs_shape()[0].size() == SIZE_THREE) {
    // Experimental support for 3D data.
    node->tensor_parm.tensor_str.str_c = 1.0 / std::min(max_device_num, target_tensor_batch);
  } else if (ops[iter_ops]->outputs_shape()[0].size() == SIZE_FOUR) {  // Experimental support for 4D data.
    node->tensor_parm.tensor_str.str_n = 1.0 / std::min(max_device_num, target_tensor_batch);
  } else {
    MS_LOG(INFO) << ops[iter_ops]->name() << " output tensor shape is unexpected, using default value instead.";
  }

  return strategies;
}

Strategies MakeFullBatchStrategy(Graph::NodeType *node, const std::vector<std::shared_ptr<OperatorInfo>> &ops,
                                 const size_t iter_ops) {
  if (ops.empty()) {
    MS_LOG(EXCEPTION) << "Failure: Operators is empty.";
  }
  if (iter_ops >= ops.size()) {
    MS_LOG(EXCEPTION) << "Failure: Operators' elements out of range.";
  }

  Strategies strategies;
  for (size_t iter_op_inputs = 0; iter_op_inputs < ops[iter_ops]->inputs_shape().size(); iter_op_inputs++) {
    if (iter_op_inputs >= ops[iter_ops]->inputs_shape().size()) {
      MS_LOG(EXCEPTION) << "Failure: Strategy's InputDim out of range.";
    }
    Dimensions strategy;
    size_t input_size = ops[iter_ops]->inputs_shape()[iter_op_inputs].size();
    for (size_t dim = 0; dim < input_size; dim++) {
      if (input_size >= SIZE_ONE && input_size <= SIZE_FOUR) {
        strategy.push_back(1);
      } else if (input_size == 0) {
        strategy = {};
      } else {
        MS_LOG(EXCEPTION) << ops[iter_ops]->name() << ": Tensor shape " << input_size << " is unexpected.";
      }
    }
    strategies.push_back(strategy);
  }
  // Update the output strategy of Rec Graph
  node->tensor_parm.tensor_str.str_n = 1.0;
  node->tensor_parm.tensor_str.str_c = 1.0;
  node->tensor_parm.tensor_str.str_h = 1.0;
  node->tensor_parm.tensor_str.str_w = 1.0;

  return strategies;
}

void SetBackToRawStrategy(const std::shared_ptr<OperatorInfo> &op) {
  Strategies strategies;

  for (size_t iter_strategy = 0; iter_strategy < op->inputs_shape().size(); iter_strategy++) {
    Dimensions strategy;
    size_t strategy_size = op->inputs_shape()[iter_strategy].size();
    for (size_t dim = 0; dim < strategy_size; dim++) {
      if (strategy_size >= SIZE_ONE && strategy_size <= SIZE_FOUR) {
        strategy.push_back(1);
      } else if (strategy_size == 0) {
        strategy = {};
      } else {
        MS_LOG(EXCEPTION) << op->name() << ": Strategy size " << strategy_size << " is unmatched.";
      }
    }
    strategies.push_back(strategy);
  }

  StrategyPtr sp = std::make_shared<Strategy>(0, strategies);

  op->SetSelectedStrategyAndCost(sp, op->selected_cost());
}

Strategies PrepareStrategy(Graph::NodeType *node, const std::vector<std::shared_ptr<OperatorInfo>> &ops,
                           const size_t iter_ops, const bool dyn_shape_tmp_fix) {
  if (ops.empty()) {
    MS_LOG(EXCEPTION) << "Failure: Operators is empty.";
  }
  if (iter_ops >= ops.size()) {
    MS_LOG(EXCEPTION) << "Failure: Operators' elements out of range.";
  }
  MS_EXCEPTION_IF_NULL(ops[iter_ops]);

  auto type = ops[iter_ops]->type();
  MS_LOG(INFO) << "Processing main operator " << ops[iter_ops]->name() << " (type=" << type << ")";
  if (type == MATMUL) {
    return PrepareMatMul(node, ops[iter_ops]);
  } else if (dyn_shape_tmp_fix && type == BATCH_MATMUL) {
    return PrepareBatchMatMul(node, ops[iter_ops]);
  } else if (type == LAYER_NORM) {
    return PrepareAxisRelatedStrategy(node, ops, iter_ops);
  } else if (type == SPARSE_SOFTMAX_CROSS_ENTROPY_WITH_LOGITS) {
    return MakeDataParallelStrategy(node, ops, iter_ops);
  } else if (type == VIRTUAL_DATA_SET) {
    if (ParallelContext::GetInstance()->full_batch()) {
      return MakeFullBatchStrategy(node, ops, iter_ops);
    } else {
      return MakeDataParallelStrategy(node, ops, iter_ops);
    }
  } else {
    return MakeRecSearchStrategy(node, ops, iter_ops);
  }
}

float CheckVirtualDatasetStrategy(Graph::NodeType *node) {
  // The values for str can only be 1.0, 0.5, 0.25, 0.125…
  // We want to find out the first str that is smaller than 1
  if (node->tensor_parm.tensor_str.str_n < 0.9) {
    return node->tensor_parm.tensor_str.str_n;
  }
  if (node->tensor_parm.tensor_str.str_c < 0.9) {
    return node->tensor_parm.tensor_str.str_c;
  }
  if (node->tensor_parm.tensor_str.str_h < 0.9) {
    return node->tensor_parm.tensor_str.str_h;
  }
  if (node->tensor_parm.tensor_str.str_w < 0.9) {
    return node->tensor_parm.tensor_str.str_w;
  }
  return 1.0;
}

Dimensions CopyVirtualDataset(Graph::NodeType *node, const std::shared_ptr<OperatorInfo> &op,
                              float epsilon = 0.00005f) {
  Dimensions strategy;
  auto input_stra_dim = op->inputs_shape()[0].size();
  auto virtual_dataset_str = CheckVirtualDatasetStrategy(node);
  MS_EXCEPTION_IF_ZERO("Virtual_Dataset", virtual_dataset_str);
  if (input_stra_dim == 0) {
    return strategy;
  } else {
    if (std::fabs(virtual_dataset_str) < epsilon) {
      strategy.push_back(1);
    } else {
      strategy.push_back(FloatToLong(1 / virtual_dataset_str));
    }
    for (size_t i = 1; i < input_stra_dim; i++) {
      strategy.push_back(1);
    }
  }
  return strategy;
}

Dimensions CopyIncomingOperatorOutputStrategy(Graph::NodeType *node,
                                              const std::vector<std::shared_ptr<OperatorInfo>> &ops,
                                              const size_t iter_ops, const size_t incoming_op_index) {
  Dimensions strategy;

  if (ops[incoming_op_index]->type() == VIRTUAL_DATA_SET) {
    strategy = CopyVirtualDataset(node, ops[iter_ops]);
    return strategy;
  }

  for (auto inputs_shape : ops[iter_ops]->inputs_shape()) {
    auto input_stra_dim = inputs_shape.size();
    if (input_stra_dim == SIZE_ZERO) {
      continue;
    }
    if (input_stra_dim == SIZE_ONE) {
      strategy.push_back(FloatToLong(1 / node->tensor_parm.tensor_str.str_w));
    } else if (input_stra_dim == SIZE_TWO) {
      strategy.push_back(FloatToLong(1 / node->tensor_parm.tensor_str.str_h));
      strategy.push_back(FloatToLong(1 / node->tensor_parm.tensor_str.str_w));
    } else if (input_stra_dim == SIZE_THREE) {
      // Experimental support for 3D data.
      strategy.push_back(FloatToLong(1 / node->tensor_parm.tensor_str.str_c));
      strategy.push_back(FloatToLong(1 / node->tensor_parm.tensor_str.str_h));
      strategy.push_back(FloatToLong(1 / node->tensor_parm.tensor_str.str_w));
    } else if (input_stra_dim == SIZE_FOUR) {
      strategy.push_back(FloatToLong(1 / node->tensor_parm.tensor_str.str_n));
      strategy.push_back(FloatToLong(1 / node->tensor_parm.tensor_str.str_c));
      strategy.push_back(FloatToLong(1 / node->tensor_parm.tensor_str.str_h));
      strategy.push_back(FloatToLong(1 / node->tensor_parm.tensor_str.str_w));
    } else {
      MS_LOG(EXCEPTION) << ops[iter_ops]->name() << ": Tensor's shape is unknown.";
    }
    break;
  }
  return strategy;
}

Dimensions PrepareReshape(std::vector<int64_t> from_shape, std::vector<int64_t> to_shape,
                          std::vector<int64_t> from_strat) {
  Dimensions to_strat(to_shape.size(), 1);
  std::vector<int64_t> from_shape_cpy(from_shape);
  std::vector<int64_t> to_shape_cpy(to_shape);
  size_t from_idx = 0;
  size_t to_idx = 0;

  // Attempt to assign full strategy to one dimension
  while (from_idx < from_shape.size() && to_idx < to_shape.size()) {
    if (from_shape[from_idx] > to_shape[to_idx]) {
      if (to_shape[to_idx] % from_strat[from_idx] == 0) {
        to_strat[to_idx] *= from_strat[from_idx];
        from_strat[from_idx] = 1;
      }
      from_shape[from_idx] /= to_shape[to_idx];
      to_idx++;
    } else if (from_shape[from_idx] < to_shape[to_idx]) {
      to_shape[to_idx] /= from_shape[from_idx];
      from_idx++;
    } else {
      if (to_shape[to_idx] % from_strat[from_idx] == 0) {
        to_strat[to_idx] *= from_strat[from_idx];
        from_strat[from_idx] = 1;
      }
      from_idx++;
      to_idx++;
    }
  }

  // Reset shapes & indices
  from_idx = 0;
  to_idx = 0;
  from_shape = from_shape_cpy;
  to_shape = to_shape_cpy;

  // Assign remaining strategy
  while (from_idx < from_shape.size() && to_idx < to_shape.size()) {
    if (from_shape[from_idx] > to_shape[to_idx]) {
      int64_t d = std::gcd(from_strat[from_idx], to_shape[to_idx]);
      to_strat[to_idx] *= d;
      from_strat[from_idx] /= d;
      from_shape[from_idx] /= to_shape[to_idx];
      to_idx++;
    } else if (from_shape[from_idx] < to_shape[to_idx]) {
      to_strat[to_idx] *= from_strat[from_idx];
      to_shape[to_idx] /= from_shape[from_idx];
      from_idx++;
    } else {  // equal case
      to_strat[to_idx] *= from_strat[from_idx];
      from_idx++;
      to_idx++;
    }
  }
  return to_strat;
}

Dimensions PrepareReshapeOutputStrategy(const std::shared_ptr<OperatorInfo> &op) {
  auto output_shape = op->outputs_shape()[0];
  auto input_shape = op->inputs_shape()[0];
  auto strategy = op->selected_strategy();

  return PrepareReshape(input_shape, output_shape, strategy->GetInputDim()[0]);
}

Dimensions PrepareTransposeOutputStrategy(const std::shared_ptr<OperatorInfo> &op) {
  Dimensions strategy;
  auto permutation = GetValue<std::vector<int64_t>>(op->input_value().at(1));
  auto op_strategy = op->selected_strategy();
  // The strategies are assigned according to the order in permutation (user defined).
  for (size_t i = 0; i < permutation.size(); i++) {
    strategy.push_back(op_strategy->GetInputDim()[0][LongToSize(permutation[i])]);
  }
  return strategy;
}

Dimensions PrepareExpandDimsOutputStrategy(const std::shared_ptr<OperatorInfo> &op) {
  Dimensions strategy;

  auto axis_input = GetValue<int64_t>(op->input_value().at(1));
  auto op_strategy = op->selected_strategy();
  bool already_expand = false;

  // axis_input can be negative, in which case the index is computed backward from the shape size.
  if (axis_input < 0) {
    axis_input = SizeToLong(op->inputs_shape()[0].size()) + axis_input + 1;
  }

  // The strategy of the expanded dimension will be assigned 1, the others take the strategies of corresponding
  // dimensions.
  for (size_t i = 0; i < op->inputs_shape()[0].size() + 1; i++) {
    if (UlongToLong(i) == axis_input) {
      strategy.push_back(1);
      already_expand = true;
    } else if (UlongToLong(i) != axis_input && !already_expand) {
      strategy.push_back(op_strategy->GetInputDim()[0][i]);
    } else {
      if (i < 1) {
        MS_LOG(EXCEPTION) << "The index i -1 is less than 0. Please check the situation.";
      }
      strategy.push_back(op_strategy->GetInputDim()[0][i - 1]);
    }
  }

  return strategy;
}

Dimensions PrepareCumOutputStrategy(const std::shared_ptr<OperatorInfo> &op) {
  Dimensions strategy;

  int64_t axis_input = 1;

  if (op->input_value().at(1)->isa<Int64Imm>()) {
    axis_input = GetValue<int64_t>(op->input_value().at(1));
    MS_LOG(INFO) << op->name() << "is a prefix sum on axis " << axis_input;
  } else {
    MS_LOG(INFO) << op->name() << "that is supposedly a cum op, has an axis that is NOT an int64";
  }

  auto op_strategy = op->selected_strategy();

  // axis_input can be negative, in which case the index is computed backward from the shape size.
  if (axis_input < 0) {
    axis_input = op->inputs_shape()[0].size() + axis_input + 1;
  }

  // The strategy of the cumulated axis will be assigned 1, the others take the strategies of corresponding dimensions.
  for (size_t i = 0; i < op->inputs_shape()[0].size(); i++) {
    if ((int64_t)i == axis_input) {
      strategy.push_back(1);
    } else {
      strategy.push_back(op_strategy->GetInputDim()[0][i]);
    }
  }

  return strategy;
}

ShapeVector GetReduceAxisList(const std::shared_ptr<OperatorInfo> &op) {
  ShapeVector axis_list;
  auto input_value = op->input_value();
  auto input_dim = op->inputs_shape()[0].size();

  if (input_value.back()->isa<ValueTuple>()) {
    auto attr_axis = GetValue<std::vector<int64_t>>(input_value.back());
    if (attr_axis.empty()) {
      for (size_t i = 0; i < input_dim; i++) {
        axis_list.push_back(i);
      }
    } else {
      axis_list = attr_axis;
    }
  } else if (input_value.back()->isa<Int64Imm>()) {
    int64_t axis = GetValue<int64_t>(input_value.back());
    axis_list.push_back(axis < 0 ? axis + SizeToLong(input_dim) : axis);
  } else {
    MS_LOG(EXCEPTION) << "Failure: Axis type is invalid." << std::endl;
  }

  return axis_list;
}

Dimensions PrepareCumInputStrategy(const std::vector<std::shared_ptr<OperatorInfo>> &ops, size_t i_ops,
                                   size_t outgoing_op_index, size_t i_input) {
  Dimensions strategy;
  int64_t axis_input = 1;

  if (ops[i_ops]->input_value().at(1)->isa<Int64Imm>()) {
    axis_input = GetValue<int64_t>(ops[i_ops]->input_value().at(1));
    MS_LOG(INFO) << ops[i_ops]->name() << "is a prefix sum on axis " << axis_input;
  } else {
    MS_LOG(INFO) << ops[i_ops]->name() << "that is supposedly a cumulative op has an axis that is NOT an int64";
  }

  auto op_strategy = ops[outgoing_op_index]->selected_strategy();

  size_t n_dim = op_strategy->GetInputDim()[i_input].size();

  if (axis_input < 0) {
    axis_input = n_dim + LongToSize(axis_input);
  }

  MS_EXCEPTION_IF_CHECK_FAIL(axis_input >= 0, "Input axis is lower than 0");

  for (size_t i_dim = 0; i_dim < n_dim; ++i_dim) {
    if (i_dim == size_t(axis_input)) {
      strategy.push_back(1);
    } else {
      strategy.push_back(op_strategy->GetInputDim()[i_input][i_dim]);
    }
  }

  return strategy;
}

Dimensions PrepareIncomingArithmeticOpeartorInputStrategy(const std::shared_ptr<OperatorInfo> &op) {
  Dimensions strategy;
  size_t max = 0;
  for (size_t i = 1; i < op->inputs_shape().size(); i++) {
    if (op->inputs_shape()[i].size() > op->inputs_shape()[max].size()) {
      max = i;
    }
  }

  for (size_t j = 0; j < op->inputs_shape()[max].size(); j++) {
    strategy.push_back(op->selected_strategy()->GetInputDim()[max][j]);
  }

  return strategy;
}

Dimensions PrepareIncomingOperatorInputStrategy(const std::shared_ptr<OperatorInfo> &op) {
  Dimensions strategy;

  if (op->type() == GATHERV2) {
    auto pos = op->name().find("Info");
    if (pos == std::string::npos) {
      return strategy;
    }
    auto name = op->name().substr(0, pos);
    if (name == "Gather") {
      return PrepareGatherV2OutputStrategy(op);
    } else {
      MS_LOG(EXCEPTION) << "Failure: Unknown type of GatherV2.";
    }
  }

  if (!HasStrategy(op)) {
    return strategy;
  }

  auto op_strategy = op->selected_strategy();
  if (op_strategy->GetInputNumber() == 0) {
    return strategy;
  }

  if (op->type() == MUL || op->type() == SUB || op->type() == ADD || op->type() == BIAS_ADD) {
    strategy = PrepareIncomingArithmeticOpeartorInputStrategy(op);
    return strategy;
  }

  if (op->type() == RESHAPE) {
    return PrepareReshapeOutputStrategy(op);
  } else if (op->type() == TRANSPOSE) {
    return PrepareTransposeOutputStrategy(op);
  } else if (op->type() == EXPAND_DIMS) {
    return PrepareExpandDimsOutputStrategy(op);
  } else if (op->type() == CUM_SUM || op->type() == CUM_PROD) {
    return PrepareCumOutputStrategy(op);
  } else if (op->type() == ONEHOT) {
    return PrepareOneHotOutputStrategy(op);
  }

  for (size_t i = 0; i < static_cast<size_t>(op->inputs_shape().size()); i++) {
    if (op->inputs_shape()[i].size() == 0) {
      continue;
    }
    for (size_t j = 0; j < op->inputs_shape()[i].size(); ++j) {
      strategy.push_back(op_strategy->GetInputDim()[i][j]);
    }
    break;
  }
  return strategy;
}

Dimensions GetAxisList(const std::vector<std::shared_ptr<OperatorInfo>> &ops, const int64_t iter_ops) {
  Dimensions axis_list;
  auto axis_param = ops[LongToSize(iter_ops)]->attrs().find(AXIS)->second;
  std::vector<ValuePtr> elements;
  if (axis_param->isa<ValueTuple>()) {
    elements = axis_param->cast<ValueTuplePtr>()->value();
  } else if (axis_param->isa<ValueList>()) {
    elements = axis_param->cast<ValueListPtr>()->value();
  } else {
    MS_LOG(EXCEPTION) << "Failure: Axis type is invalid, neither tuple nor list.";
  }

  for (auto &element : elements) {
    if (!element->isa<Int64Imm>()) {
      MS_LOG(EXCEPTION) << "Failure: Dimension indexes is not Int32.";
    }
    auto axis = element->cast<Int64ImmPtr>()->value();
    axis_list.push_back(axis);
  }
  return axis_list;
}

Dimensions ModifyStrategyIfSqueezeIncoming(const std::vector<std::shared_ptr<OperatorInfo>> &ops,
                                           const size_t incoming_op_index, Dimensions strategy) {
  Dimensions s_Squeeze;
  Dimensions stra_dim_list;
  for (size_t i = 0; i < strategy.size(); i++) {
    stra_dim_list.push_back(SizeToLong(i));
  }

  auto axis_list = GetAxisList(ops, SizeToLong(incoming_op_index));
  for (auto axis : axis_list) {
    axis = (axis < 0) ? (strategy.size() + axis) : axis;
    auto it = find(stra_dim_list.begin(), stra_dim_list.end(), axis);
    if (it == stra_dim_list.end()) {
      MS_LOG(EXCEPTION) << "Failure: Can not find dimension indexes in Axis.";
    }
    if (ops[incoming_op_index]->inputs_shape()[0][LongToSize(axis)] != 1) {
      MS_LOG(EXCEPTION) << "Failure: Removed dimension's shape is not 1.";
    }
    (void)stra_dim_list.erase(it);
  }

  for (size_t i = 0; i < stra_dim_list.size(); i++) {
    s_Squeeze.push_back(strategy[LongToSize(stra_dim_list[i])]);
  }
  return s_Squeeze;
}

Dimensions ModifyStrategyIfReduceIncoming(const std::shared_ptr<OperatorInfo> &op, Dimensions strategy) {
  Dimensions s_Reduce;
  Dimensions axis_list;
  for (size_t i = 0; i < strategy.size(); i++) {
    axis_list.push_back(SizeToLong(i));
  }

  auto dim_list = GetDimList(op);
  for (auto axis : dim_list) {
    auto it = find(axis_list.begin(), axis_list.end(), axis);
    if (it == axis_list.end()) {
      MS_LOG(EXCEPTION) << "Failure: Can not find dimension indexes in Axis.";
    }
    (void)axis_list.erase(it);
  }

  for (size_t i = 0; i < axis_list.size(); i++) {
    s_Reduce.push_back(strategy[LongToSize(axis_list[i])]);
  }
  return s_Reduce;
}

Dimensions GetDimListFromAttrs(const std::shared_ptr<OperatorInfo> &op) {
  Dimensions dim_list;
  auto iter = op->attrs().find(AXIS);
  if (iter == op->attrs().end()) {
    MS_LOG(EXCEPTION) << op->name() << ": Don't have attr axis.";
  }
  auto input_dim = op->inputs_shape()[0].size();
  MS_EXCEPTION_IF_NULL(iter->second);
  if (iter->second->isa<ValueTuple>()) {
    auto attr_axis = GetValue<std::vector<int64_t>>(iter->second);
    if (attr_axis.empty()) {
      for (size_t i = 0; i < input_dim; ++i) {
        dim_list.push_back(SizeToLong(i));
      }
    } else {
      for (auto &axis : attr_axis) {
        axis < 0 ? dim_list.push_back(axis + SizeToLong(input_dim)) : dim_list.push_back(axis);
      }
    }
  } else if (iter->second->isa<Int64Imm>()) {
    int64_t axis = GetValue<int64_t>(iter->second);
    axis < 0 ? dim_list.push_back(axis + SizeToLong(input_dim)) : dim_list.push_back(axis);
  } else {
    MS_LOG(EXCEPTION) << "Axis type is invalid.";
  }
  return dim_list;
}

Dimensions ModifyStrategyIfArgIncoming(const std::shared_ptr<OperatorInfo> &op, Dimensions strategy) {
  bool keepdims = GetKeepDims(op);
  if (keepdims) {
    return strategy;
  }

  Dimensions s_Arg;
  Dimensions axis_list;
  for (size_t i = 0; i < strategy.size(); i++) {
    axis_list.push_back(SizeToLong(i));
  }

  auto dim_list = GetDimListFromAttrs(op);
  for (auto axis : dim_list) {
    auto it = find(axis_list.begin(), axis_list.end(), axis);
    if (it == axis_list.end()) {
      MS_LOG(EXCEPTION) << "Failure: Can not find dimension indexes in Axis.";
    }
    (void)axis_list.erase(it);
  }

  for (size_t i = 0; i < axis_list.size(); i++) {
    s_Arg.push_back(strategy[LongToSize(axis_list[i])]);
  }
  return s_Arg;
}

Dimensions ModifyStrategyIfFlattenIncoming(const std::shared_ptr<OperatorInfo> &op, Dimensions strategy) {
  Dimensions new_strategy;
  int start_dim = 1, end_dim = strategy.size() - 1;
  auto start_dim_iter = op->attrs().find("start_dim");
  if (start_dim_iter != op->attrs().end()) {
    start_dim = GetValue<int64_t>(start_dim_iter->second);
  }
  auto end_dim_iter = op->attrs().find("end_dim");
  if (end_dim_iter != op->attrs().end() && GetValue<int64_t>(end_dim_iter->second) >= 0) {
    end_dim = GetValue<int64_t>(end_dim_iter->second);
  }

  for (int idx = 0; idx < start_dim; idx++) {
    new_strategy.push_back(strategy[idx]);
  }

  int flatten_strategy = 1;
  for (int idx = start_dim; idx < end_dim + 1; idx++) {
    flatten_strategy *= strategy[idx];
  }
  new_strategy.push_back(flatten_strategy);
  if (IntToSize(end_dim + 1) < strategy.size()) {
    for (size_t idx = end_dim + 1; idx < strategy.size(); idx++) {
      new_strategy.push_back(strategy[idx]);
    }
  }

  return new_strategy;
}

Dimensions CopyIncomingOperatorInputStrategy(const std::vector<std::shared_ptr<OperatorInfo>> &ops,
                                             const size_t iter_ops, const size_t incoming_op_index) {
  Dimensions strategy;
  if (ops[iter_ops]->type() == ONEHOT) {
    return strategy;
  }
  if (ops[iter_ops]->type() == TRANSPOSE) {
    return strategy;
  }
  if (ops[incoming_op_index]->type() == STRIDED_SLICE) {
    return strategy;
  }
  strategy = PrepareIncomingOperatorInputStrategy(ops[incoming_op_index]);
  if (strategy.size() != 0) {
    if (ops[incoming_op_index]->type() == SQUEEZE) {
      strategy = ModifyStrategyIfSqueezeIncoming(ops, incoming_op_index, strategy);
    }
    if (ops[incoming_op_index]->type() == REDUCE_SUM || ops[incoming_op_index]->type() == REDUCE_MAX ||
        ops[incoming_op_index]->type() == REDUCE_MIN || ops[incoming_op_index]->type() == REDUCE_MEAN) {
      strategy = ModifyStrategyIfReduceIncoming(ops[incoming_op_index], strategy);
    }
    if (ops[incoming_op_index]->type() == ARGMAXWITHVALUE || ops[incoming_op_index]->type() == ARGMINWITHVALUE) {
      strategy = ModifyStrategyIfArgIncoming(ops[incoming_op_index], strategy);
    }
    if (ops[incoming_op_index]->type() == FLATTEN) {
      strategy = ModifyStrategyIfFlattenIncoming(ops[incoming_op_index], strategy);
    }
  }
  return strategy;
}

Strategies PrepareDropoutDoMask(const std::shared_ptr<OperatorInfo> &op, Dimensions basic_stra,
                                bool dyn_shape_tmp_fix) {
  // Dropout's strategy shape must be 1.
  Strategies strategies;
  strategies.clear();
  strategies.push_back(basic_stra);
  return strategies;
}

// Function to deal with ops with broadcasting, like TensorAdd/Sub/Mul/Div etc.
Strategies CheckBroadcast(const std::shared_ptr<OperatorInfo> &op, Dimensions strategy, bool dyn_shape_tmp_fix) {
  Strategies strategies;

  size_t first_tensor_dim = op->inputs_shape()[0].size();
  size_t second_tensor_dim = op->inputs_shape()[1].size();
  size_t s_dim = strategy.size();
  // Do Broadcasting in the second tensor.
  if (second_tensor_dim < first_tensor_dim) {
    if (s_dim == first_tensor_dim) {
      bool broadcast_first_tensor = false;
      strategies.push_back(strategy);
      strategies.push_back(ApplyBroadcast(op, strategy, broadcast_first_tensor));
    } else {
      // When the strategy is from the smaller tensor, make the strategy all 1.
      Dimensions broadcast_revise_s(first_tensor_dim, 1);
      strategies.push_back(broadcast_revise_s);
      Dimensions broadcast_s(strategy.size(), 1);
      strategies.push_back(broadcast_s);
    }
  } else if (second_tensor_dim > first_tensor_dim) {  // Do Broadcasting in the first tensor.
    if (s_dim == second_tensor_dim) {
      bool broadcast_first_tensor = true;
      strategies.push_back(ApplyBroadcast(op, strategy, broadcast_first_tensor));
      strategies.push_back(strategy);
    } else {
      // When the strategy is from the smaller tensor, make the strategy all 1.
      Dimensions broadcast_s(strategy.size(), 1);
      strategies.push_back(broadcast_s);
      Dimensions broadcast_revise_s(second_tensor_dim, 1);
      strategies.push_back(broadcast_revise_s);
    }
  } else {  // Broadcasting can be ignored or No broadcasting needs to be applied.
    strategies = CheckDivisible(op, strategy);
  }
  // Strategy protection to avoid that partition number is larger than the shape of related dimension.
  for (size_t i = 0; i < op->inputs_shape().size(); i++) {
    for (size_t j = 0; j < op->inputs_shape()[i].size(); j++) {
      if (strategies[i][j] > op->inputs_shape()[i][j] || op->inputs_shape()[i][j] % strategies[i][j] != 0) {
        strategies[i][j] = 1;
      }
    }
  }

  return strategies;
}

void InitializeStrategyMap() {
  if (g_prepare_stra_map.empty()) {
    g_prepare_stra_map =
      std::map<std::string, PrepareStraFuncPtr>{{FILLV2, &PrepareFillV2},
                                                {BIAS_ADD, &PrepareBiasAdd},
                                                {STRIDED_SLICE, &PrepareStridedSlice},
                                                {GATHERV2, &PrepareGather},
                                                {ONEHOT, &PrepareOneHot},
                                                {L2_NORMALIZE, &PrepareL2Normalize},
                                                {ADD, &CheckBroadcast},
                                                {SUB, &CheckBroadcast},
                                                {MUL, &CheckBroadcast},
                                                {DIV, &CheckBroadcast},
                                                {SOFTMAX, &PrepareSoftMax},
                                                {LOG_SOFTMAX, &PrepareSoftMax},
                                                {FLATTEN, &PrepareDataParallel},
                                                {GATHERD, &PrepareDataParallel},
                                                {LAYER_NORM, &PrepareLayerNorm},
                                                {RMS_NORM, &PrepareRmsNorm},
                                                {BATCH_MATMUL, &PreparePropagateBatchMatMul},
                                                {DROPOUT_DO_MASK, &PrepareDropoutDoMask},
                                                {FLASH_ATTENTION_SCORE, &PrepareFlashAttentionScore}};
  }
}

Strategies GenerateStrategiesFromStrategy(const std::vector<std::shared_ptr<OperatorInfo>> &ops, const size_t iter_ops,
                                          Dimensions basic_stra, bool dyn_shape_tmp_fix) {
  MS_EXCEPTION_IF_NULL(ops[iter_ops]);

  if (iter_ops >= ops.size()) {
    MS_LOG(EXCEPTION) << "Failure: Operators' elements out of range.";
  }

  Strategies strategies;
  if (basic_stra.size() == 0) {
    for (size_t iter_op_inputs = 0; iter_op_inputs < static_cast<size_t>(ops[iter_ops]->inputs_shape().size());
         iter_op_inputs++) {
      strategies.push_back(basic_stra);
    }
    return strategies;
  }
  InitializeStrategyMap();
  auto type = ops[iter_ops]->type();
  auto iter_stra_func = g_prepare_stra_map.find(type);
  if (iter_stra_func != g_prepare_stra_map.end()) {
    auto stra = iter_stra_func->second(ops[iter_ops], basic_stra, dyn_shape_tmp_fix);
    return stra;
  }

  return CheckDivisible(ops[iter_ops], basic_stra);
}

Dimensions ApplyBroadcast(const std::shared_ptr<OperatorInfo> &op, const Dimensions &strategy,
                          bool broadcast_first_tensor) {
  Dimensions s_broadcast;
  size_t target_tensor_index = 0;
  size_t target_tensor_dim = 1;

  // Indexing target and refer tensor.
  if (!broadcast_first_tensor) {
    target_tensor_index = 1;
  }

  target_tensor_dim = op->inputs_shape()[target_tensor_index].size();
  for (size_t iter = 0; iter < target_tensor_dim; iter++) {
    if (op->inputs_shape()[target_tensor_index][target_tensor_dim - 1 - iter] == 1) {
      s_broadcast.insert(s_broadcast.begin(), 1);
    } else {
      s_broadcast.insert(s_broadcast.begin(), strategy[strategy.size() - 1 - iter]);
    }
  }

  return s_broadcast;
}

// Check whether the operator can be divided by the current strategy.
Strategies CheckDivisible(const std::shared_ptr<OperatorInfo> &op, const Dimensions &basic_stra) {
  Dimensions s_empty = {};
  Strategies strategies;

  // For all the input tensors.
  for (size_t iter_op_inputs = 0; iter_op_inputs < op->inputs_shape().size(); iter_op_inputs++) {
    // If input tensor is empty, return strategy as void.
    if (op->inputs_shape()[iter_op_inputs].size() == 0) {
      strategies.push_back(s_empty);
      continue;
    }

    Dimensions tmp_stra;

    // Make sure each tensor's dim shape is greater than 1. If not, push back strategy as 1 instead.
    for (size_t j = 0; j < op->inputs_shape()[iter_op_inputs].size(); j++) {
      if (op->inputs_shape()[iter_op_inputs][j] == 1) {
        tmp_stra.push_back(1);
      } else if (j < basic_stra.size()) {
        tmp_stra.push_back(basic_stra[j]);
      } else {
        tmp_stra.push_back(1);
      }
    }
    strategies.push_back(tmp_stra);
  }

  return strategies;
}

Dimensions ModifyStrategyIfSqueezeOutgoing(const std::vector<std::shared_ptr<OperatorInfo>> &ops, const size_t iter_ops,
                                           Dimensions strategy) {
  Dimensions s_Squeeze;
  auto axis_list = GetAxisList(ops, SizeToLong(iter_ops));
  size_t s_index = 0;
  size_t axis_list_index = 0;
  for (size_t i = 0; i < strategy.size() + axis_list.size(); i++) {
    if (axis_list[axis_list_index] > 0 && i == LongToSize(axis_list[axis_list_index])) {
      s_Squeeze.push_back(1);
      axis_list_index++;
    } else {
      s_Squeeze.push_back(strategy[s_index]);
      s_index++;
    }
  }

  size_t cut = 1;
  for (size_t i = 0; i < s_Squeeze.size(); i++) {
    cut *= LongToSize(s_Squeeze[i]);
  }
  if (cut != size_t(g_device_manager->stage_device_num())) {
    s_Squeeze.clear();
  }

  return s_Squeeze;
}

Dimensions PrepareExpandDimsInputStrategy(const std::vector<std::shared_ptr<OperatorInfo>> &ops, size_t i_ops,
                                          size_t outgoing_op_index, size_t i_input) {
  Dimensions strategy;

  int64_t axis_input = GetValue<int64_t>(ops[i_ops]->input_value().at(1));

  auto op_strategy = ops[outgoing_op_index]->selected_strategy();

  size_t n_dim = op_strategy->GetInputDim()[i_input].size();

  if (axis_input < 0) {
    axis_input = SizeToLong(n_dim) + axis_input;
  }

  MS_EXCEPTION_IF_CHECK_FAIL(axis_input >= 0, "Input axis is lower than 0");

  for (size_t i_dim = 0; i_dim < n_dim; ++i_dim) {
    if (i_dim != size_t(axis_input)) {
      strategy.push_back(op_strategy->GetInputDim()[i_input][i_dim]);
    }
  }

  return strategy;
}

Dimensions PrepareReshapeInputStrategy(const std::vector<std::shared_ptr<OperatorInfo>> &ops, size_t i_ops,
                                       size_t outgoing_op_index, size_t iter_op_inputs, bool dyn_shape_tmp_fix) {
  if (dyn_shape_tmp_fix) {
    Dimensions empty_strategy;
    return empty_strategy;
  }
  auto output_shape = ops[i_ops]->outputs_shape()[0];
  auto input_shape = ops[i_ops]->inputs_shape()[0];
  auto strategy = ops[outgoing_op_index]->selected_strategy();

  return PrepareReshape(output_shape, input_shape, strategy->GetInputDim()[iter_op_inputs]);
}

Dimensions PrepareGatherV2InputStrategy(const std::shared_ptr<OperatorInfo> &op, size_t i_input) {
  auto targeted_shape = op->inputs_shape()[i_input];
  Dimensions strategie = GenGatherStra(targeted_shape);
  return strategie;
}

Dimensions PrepareReduceOutputStrategy(const std::shared_ptr<OperatorInfo> &op) {
  bool keep_dims = GetKeepDims(op);
  auto axis_list = GetDimList(op);
  auto basic_stra = op->selected_strategy()->GetInputDim().at(0);

  Dimensions strategy;

  for (size_t i = 0; i < basic_stra.size(); ++i) {
    if (std::find(axis_list.begin(), axis_list.end(), i) != axis_list.end()) {
      if (keep_dims) {
        strategy.push_back(1);
      }
    } else {
      strategy.push_back(basic_stra.at(i));
    }
  }

  return strategy;
}

Dimensions PrepareReduceInputStrategy(const std::vector<std::shared_ptr<OperatorInfo>> &ops, size_t i_ops,
                                      size_t outgoing_op_index, size_t i_input) {
  bool keep_dims = GetKeepDims(ops[i_ops]);

  auto axis_list = GetDimList(ops[i_ops]);

  Dimensions strategy;

  auto basic_stra = ops[outgoing_op_index]->selected_strategy()->GetInputDim().at(i_input);

  for (size_t i = 0, i_stra = 0; i < ops[i_ops]->inputs_shape()[0].size(); ++i) {
    if (std::find(axis_list.begin(), axis_list.end(), i) != axis_list.end()) {
      strategy.push_back(1);
      if (keep_dims) {
        ++i_stra;
      }
    } else {
      strategy.push_back(basic_stra.at(i_stra++));
    }
  }

  return strategy;
}

Dimensions PrepareTransposeInputStrategy(const std::vector<std::shared_ptr<OperatorInfo>> &ops, size_t i_ops,
                                         size_t outgoing_op_index, size_t iter_op_inputs) {
  Dimensions strategy;
  auto permutation = GetValue<std::vector<int64_t>>(ops[i_ops]->input_value().at(1));
  auto op_strategy = ops[outgoing_op_index]->selected_strategy();
  // The strategies are assigned according to the order in permutation (user defined).
  for (size_t i = 0; i < permutation.size(); i++) {
    strategy.push_back(op_strategy->GetInputDim()[iter_op_inputs][LongToSize(permutation[i])]);
  }
  return strategy;
}

Dimensions CopyOutgoingOperatorInputStrategy(const std::vector<std::shared_ptr<OperatorInfo>> &ops, size_t iter_ops,
                                             size_t outgoing_op_index, size_t iter_op_inputs, bool dyn_shape_tmp_fix) {
  Dimensions strategy;
  // Propagation not implemented for these operators
  if (ops[iter_ops]->type() == ARGMAXWITHVALUE || ops[iter_ops]->type() == ARGMINWITHVALUE) {
    return strategy;
  }

  // Propagation not allowed for these operators
  if (ops[iter_ops]->type() == FLATTEN) {
    return strategy;
  }

  if (outgoing_op_index != SIZE_MAX && iter_op_inputs != SIZE_MAX) {
    std::string type = ops[iter_ops]->type();
    if (type == EXPAND_DIMS) {
      strategy = PrepareExpandDimsInputStrategy(ops, iter_ops, outgoing_op_index, iter_op_inputs);
    } else if (type == RESHAPE) {
      strategy = PrepareReshapeInputStrategy(ops, iter_ops, outgoing_op_index, iter_op_inputs, dyn_shape_tmp_fix);
      return strategy;
    } else if (type == GATHERV2) {
      strategy = PrepareGatherV2InputStrategy(ops[outgoing_op_index], iter_op_inputs);
      return strategy;
    } else if (type == REDUCE_MEAN || type == REDUCE_MAX || type == REDUCE_MIN || type == REDUCE_SUM) {
      strategy = PrepareReduceInputStrategy(ops, iter_ops, outgoing_op_index, iter_op_inputs);
    } else if (type == TRANSPOSE) {
      strategy = PrepareTransposeInputStrategy(ops, iter_ops, outgoing_op_index, iter_op_inputs);
      return strategy;
    } else {
      for (size_t k = 0; k < ops[iter_ops]->outputs_shape()[0].size(); ++k) {
        strategy.push_back(ops[outgoing_op_index]->selected_strategy()->GetInputDim()[iter_op_inputs][k]);
      }
    }
    if (!IsDimensionsEmpty(strategy) && ops[iter_ops]->type() == SQUEEZE) {
      strategy = ModifyStrategyIfSqueezeOutgoing(ops, iter_ops, strategy);
    }
  }

  return strategy;
}

void RecStrategyPropagator::ApplyStrategy(size_t i_op, const Strategies &strategies) {
  StrategyPtr sp = std::make_shared<Strategy>(0, strategies);
  ops_[i_op]->SetSelectedStrategyAndCost(sp, ops_[i_op]->selected_cost());
}

size_t RecStrategyPropagator::GetMaxDimNum(size_t i_op) {
  size_t max_dim_num = 0;
  for (size_t iter_op_inputs = 0; iter_op_inputs < ops_[i_op]->inputs_shape().size(); iter_op_inputs++) {
    if (ops_[i_op]->inputs_shape()[iter_op_inputs].size() > max_dim_num) {
      max_dim_num = ops_[i_op]->inputs_shape()[iter_op_inputs].size();
    }
  }

  return max_dim_num;
}

Dimensions RecStrategyPropagator::GetDefaultStrategy(size_t i_op) {
  Dimensions strategy;
  size_t max_dim_num = GetMaxDimNum(i_op);
  for (size_t i = 0; i < max_dim_num; i++) {
    strategy.push_back(1);
  }

  return strategy;
}

bool StopPropAtOP(std::string op_type) {
  const std::set<std::string> stop_at = {GATHERV2, ASSIGN, EXPAND_DIMS};
  return stop_at.find(op_type) != stop_at.end();
}

size_t RecStrategyPropagator::GenerateEliminatedOperatorStrategyForward(size_t min_devices) {
  MS_LOG(INFO) << "There are " << no_stra_op_list_->size() << " operators left that do not have strategy.";
  size_t changes = 0;
  if (no_stra_op_list_->empty()) {
    return changes;
  }

  std::vector<size_t> no_stra_op_list_bis;
  for (size_t iter_list = no_stra_op_list_->size(); iter_list > 0; iter_list--) {
    size_t iter_ops = no_stra_op_list_->at(iter_list - 1);
    Strategies strategies;
    size_t incoming_op_index = FindIndexOfOperatorIncoming(ops_, input_tensor_names_, iter_ops);
    Dimensions strategy = GetInputStrategy(graph_, ops_, index_list_, iter_ops, incoming_op_index);
    if (IsDimensionsEmpty(strategy) || DevicesForDimensions(strategy) < min_devices ||
        StopPropAtOP(ops_[incoming_op_index]->type())) {
      no_stra_op_list_bis.push_back(iter_ops);
    } else {
      strategies = GenerateStrategiesFromStrategy(ops_, iter_ops, strategy, graph_->dyn_shape_tmp_fix);
      ApplyStrategy(iter_ops, strategies);
      ++changes;
      MS_LOG(INFO) << ops_[iter_ops]->name() << " assigned strategies " << StrategyToString(strategies) << " from "
                   << ops_[incoming_op_index]->name() << " with strategy " << strategy;
    }
  }
  *no_stra_op_list_ = no_stra_op_list_bis;

  return changes;
}

size_t RecStrategyPropagator::GenerateEliminatedOperatorStrategyBackward(size_t min_devices) {
  MS_LOG(INFO) << "There are " << no_stra_op_list_->size() << " operators left that do not have strategy.";
  size_t changes = 0;
  if (no_stra_op_list_->empty()) {
    return changes;
  }

  std::vector<size_t> no_stra_op_list_bis;
  for (size_t iter_list = no_stra_op_list_->size(); iter_list > 0; iter_list--) {
    auto iter_ops = no_stra_op_list_->at(iter_list - 1);
    Strategies strategies;
    std::pair<size_t, size_t> idx = FindIndexOfOperatorOutgoing(ops_, input_tensor_names_, iter_ops);
    size_t outgoing_op_index = idx.first;
    size_t iter_op_inputs = idx.second;
    Dimensions strategy =
      CopyOutgoingOperatorInputStrategy(ops_, iter_ops, outgoing_op_index, iter_op_inputs, graph_->dyn_shape_tmp_fix);
    if (IsDimensionsEmpty(strategy) || DevicesForDimensions(strategy) < min_devices ||
        StopPropAtOP(ops_[outgoing_op_index]->type())) {
      no_stra_op_list_bis.push_back(iter_ops);
    } else {
      strategies = GenerateStrategiesFromStrategy(ops_, iter_ops, strategy, graph_->dyn_shape_tmp_fix);
      ApplyStrategy(iter_ops, strategies);
      ++changes;
      MS_LOG(INFO) << ops_[iter_ops]->name() << " assigned strategies " << StrategyToString(strategies) << " from "
                   << ops_[outgoing_op_index]->name() << " with strategy " << strategy;
    }
  }
  *no_stra_op_list_ = no_stra_op_list_bis;

  return changes;
}

size_t RecStrategyPropagator::GenerateRemainingOperatorStrategy() {
  size_t changes = 0;

  if (no_stra_op_list_->empty()) {
    return changes;
  }

  size_t no_stra_op_list_size = no_stra_op_list_->size();
  do {
    no_stra_op_list_size = no_stra_op_list_->size();
    changes += GenerateEliminatedOperatorStrategyForward();
    changes += GenerateEliminatedOperatorStrategyBackward();
  } while (no_stra_op_list_size > no_stra_op_list_->size());

  for (size_t iter_list = 0; iter_list < no_stra_op_list_->size(); iter_list++) {
    auto iter_ops = no_stra_op_list_->at(iter_list);
    Dimensions strategy = GetDefaultStrategy(iter_ops);
    if (graph_->dyn_shape_tmp_fix && strategy.empty()) {
      continue;
    }
    Strategies strategies = GenerateStrategiesFromStrategy(ops_, iter_ops, strategy, graph_->dyn_shape_tmp_fix);
    ApplyStrategy(iter_ops, strategies);
    ++changes;
    MS_LOG(INFO) << ops_[iter_ops]->name() << " assigned default strategies " << StrategyToString(strategies)
                 << " with strategy  " << strategy;
  }

  return changes;
}

// param_name equals to (operator index * input index)
std::map<std::string, std::vector<std::pair<size_t, size_t>>> RecStrategyPropagator::GetParamUsers() {
  std::map<std::string, std::vector<std::pair<size_t, size_t>>> param_users;

  AnfNodePtr ret = root_->get_return();
  std::vector<AnfNodePtr> all_nodes = DeepScopedGraphSearch(ret);

  for (auto &node : all_nodes) {
    if (node->isa<Parameter>()) {
      ParameterUsersInfo parameter_users_info = FindParameterUsers(node, IsParallelCareNode, all_nodes);
      auto users_set = parameter_users_info.second.second;
      if (users_set.size() >= 1) {
        MS_LOG(INFO) << "Parameter " << parameter_users_info.first << " has " << users_set.size() << " users.";
        for (auto &user : users_set) {
          MS_LOG(INFO) << "with ID: " << user.first->UniqueId() << " and name: " << user.first->UniqueName();

          std::pair<size_t, size_t> user_index = std::make_pair(SIZE_MAX, SIZE_MAX);
          for (size_t i = 0; i < input_tensor_names_.size(); i++) {
            if (input_tensor_names_[i][0] == user.first->UniqueId()) {
              size_t input_index = 0;
              if ((ops_[i]->type() == MATMUL) || (ops_[i]->type() == BATCH_MATMUL)) {
                input_index = 1;
              }
              user_index = std::make_pair(i, input_index);
            }
          }
          if (user_index.first != SIZE_MAX) {
            param_users[parameter_users_info.first].push_back(user_index);
          }
        }
      }
    }
  }

  return param_users;
}

void RecStrategyPropagator::SetParamStrategy() {
  std::map<std::string, std::vector<std::pair<size_t, size_t>>> params_users = GetParamUsers();  // perhaps store this ?
  for (auto &param : params_users) {
    MS_LOG(INFO) << "Treat parameter " << param.first << " with " << param.second.size() << " uers";
    if (param_strategy_.find(param.first) == param_strategy_.end() && !param.second.empty()) {
      Dimensions strategy;
      Dimensions max_strat;
      int max_stra_cut_num = 1;
      int max_stra_cut_ratio = INT_MAX;

      for (auto &user : param.second) {
        MS_LOG(INFO) << "user is " << ops_[user.first]->name() << " param goes to input " << user.second;
        if (!HasStrategy(ops_[user.first])) {
          continue;
        }
        strategy = ops_[user.first]->selected_strategy()->GetInputDim()[user.second];
        if (strategy.empty()) {
          MS_LOG(INFO) << "user has no strategy";
          continue;
        }
        MS_LOG(INFO) << "This user wants strategy " << strategy;

        auto param_shape = ops_[user.first]->inputs_shape()[user.second];
        auto ratio = 0;
        for (size_t idx = 0; idx < strategy.size(); idx++) {
          MS_EXCEPTION_IF_ZERO("strategy", strategy[idx]);
          ratio += param_shape[idx] / strategy[idx];
        }

        int cut_num = DevicesForDimensions(strategy);
        if (cut_num >= max_stra_cut_num && ratio < max_stra_cut_ratio) {
          max_stra_cut_num = cut_num;
          max_stra_cut_ratio = ratio;
          max_strat = strategy;
        }
      }
      if (!max_strat.empty()) {
        param_strategy_[param.first] = max_strat;
      }
    }
  }
  MS_LOG(INFO) << "Done";
}

Strategies MakeGatherStratFromParam(const std::shared_ptr<OperatorInfo> &op, Dimensions param_strategy) {
  Strategies strategies;
  Dimensions index_strategy;
  int64_t axis = GetGatherAxis(op);
  if (param_strategy.at(LongToSize(axis)) == 1) {
    size_t num_device_used = 1;
    for (size_t i = 0; i < param_strategy.size(); i++) {
      num_device_used *= param_strategy[i];
    }
    MS_EXCEPTION_IF_ZERO("num_device_used", num_device_used);
    index_strategy.push_back(g_device_manager->stage_device_num() / num_device_used);
  } else {
    index_strategy.push_back(1);
  }

  for (size_t i = 1; i < op->inputs_shape()[1].size(); ++i) {
    index_strategy.push_back(1);
  }

  strategies.push_back(param_strategy);
  strategies.push_back(index_strategy);

  MS_LOG(INFO) << "Gather is assigned strategy " << StrategyToString(strategies);

  return strategies;
}

Strategies MakeMatMulStratFromParam(const std::shared_ptr<OperatorInfo> &op, Dimensions param_strategy) {
  Strategies new_strategy;
  Dimensions new_param_strat;
  Dimensions input0_strat = op->selected_strategy()->GetInputDim()[0];
  int64_t k_cuts = 1;

  auto input_value = op->input_value();
  bool transpose_a = input_value[2]->cast<BoolImmPtr>()->value();
  bool transpose_b = input_value[3]->cast<BoolImmPtr>()->value();

  k_cuts = param_strategy[0];
  if (transpose_b) {
    new_param_strat.push_back(param_strategy[1]);
    new_param_strat.push_back(param_strategy[0]);
  } else {
    new_param_strat.push_back(param_strategy[0]);
    new_param_strat.push_back(param_strategy[1]);
  }

  if (transpose_a) {
    input0_strat[0] = k_cuts;
    input0_strat[1] = std::min(input0_strat[1], g_device_manager->stage_device_num() / k_cuts);
  } else {
    input0_strat[1] = k_cuts;
    input0_strat[0] = std::min(input0_strat[1], g_device_manager->stage_device_num() / k_cuts);
  }

  new_strategy.push_back(input0_strat);
  new_strategy.push_back(new_param_strat);

  MS_LOG(INFO) << "Transpose B : " << transpose_b << "; Transpose A : " << transpose_a << "; K cuts : " << k_cuts;

  MS_LOG(INFO) << "MatMul is assigned strategy " << StrategyToString(new_strategy);

  return new_strategy;
}

size_t RecStrategyPropagator::ApplyParamStrategy() {
  size_t changes = 0;
  std::map<std::string, std::vector<std::pair<size_t, size_t>>> params_users = GetParamUsers();

  for (auto &param : params_users) {
    if (param_strategy_.find(param.first) != param_strategy_.end()) {
      for (auto &user : param.second) {
        if (graph_->dyn_shape_tmp_fix && ops_[user.first]->type() == GATHERV2) {
          if (param.first.find(".output.ffn.projection.weight") != std::string::npos) {
            ApplyStrategy(user.first, GatherForDynamicShape(ops_[user.first], 1));
            continue;
          }
          if (param.first.find(".output.ffn.mapping.bias") != std::string::npos) {
            ApplyStrategy(user.first, GatherForDynamicShape(ops_[user.first], 3));
            continue;
          }
          if (param.first.find(".output.ffn.mapping.weight") != std::string::npos) {
            ApplyStrategy(user.first, GatherForDynamicShape(ops_[user.first], 2));
            continue;
          }
          // This Gather uses shared parameter, but it is not treated as using shared parameter.
          // Temporary workaround until this issue is fixed.
          if (param.first.find(".embedding.word_embedding.embedding_table") != std::string::npos) {
            ApplyStrategy(user.first, GatherForDynamicShape(ops_[user.first], 0));
            continue;
          }
        }

        if (!HasStrategy(ops_[user.first]) ||
            param_strategy_[param.first] != ops_[user.first]->selected_strategy()->GetInputDim()[user.second]) {
          Strategies strategies;
          if (ops_[user.first]->type() == GATHERV2) {
            strategies = MakeGatherStratFromParam(ops_[user.first], param_strategy_[param.first]);
          } else if (ops_[user.first]->type() == MATMUL) {
            strategies = MakeMatMulStratFromParam(ops_[user.first], param_strategy_[param.first]);
          } else if (ops_[user.first]->type() == STRIDED_SLICE) {
            strategies = CheckDivisible(ops_[user.first], param_strategy_[param.first]);
          } else {
            strategies =
              GenerateStrategiesFromStrategy(ops_, user.first, param_strategy_[param.first], graph_->dyn_shape_tmp_fix);
          }
          ApplyStrategy(user.first, strategies);
          MS_LOG(INFO) << ops_[user.first]->name() << " assigned strategy " << StrategyToString(strategies)
                       << " from parameter " << param.first;
          ++changes;
        }
      }
    }
  }
  return changes;
}

size_t RecStrategyPropagator::ModifyParamSharingOpsStrategy() {
  size_t changes = 0;

  for (auto tensor : shared_tensors_ops_) {
    for (auto op_i : tensor) {
      for (auto op_j : tensor) {
        if (op_i != op_j) {
          MS_LOG(INFO) << "Operator " << ops_[op_i]->name() << " sharing parameter with operator "
                       << ops_[op_j]->name();
        }
      }
    }
  }

  for (auto tensor : shared_tensors_ops_) {
    for (auto op_i : tensor) {
      if (ops_[op_i]->type() == GATHERV2) {
        for (auto op_j : tensor) {
          if (op_i != op_j) {
            Dimensions str_j;
            if (ops_[op_j]->type() == CAST) {
              str_j = ops_[op_j]->selected_strategy()->GetInputDim()[0];
            } else if (ops_[op_j]->type() == MATMUL) {
              str_j = ops_[op_j]->selected_strategy()->GetInputDim()[1];
            } else if (ops_[op_j]->type() == MUL) {
              str_j = ops_[op_j]->selected_strategy()->GetInputDim()[0];
            } else {
              continue;
            }

            Strategies strategies;
            Dimensions param_strategy, index_strategy;

            param_strategy = str_j;

            size_t num_device_used = 1;
            for (size_t i = 0; i < str_j.size(); i++) {
              num_device_used *= LongToSize(str_j[i]);
            }
            MS_EXCEPTION_IF_ZERO("num_device_used", num_device_used);
            index_strategy.push_back(g_device_manager->stage_device_num() / num_device_used);

            for (size_t i = 1; i < ops_[op_i]->inputs_shape()[1].size(); ++i) {
              index_strategy.push_back(1);
            }

            strategies.push_back(param_strategy);
            strategies.push_back(index_strategy);

            MS_LOG(INFO) << "Changing strategy of " << ops_[op_i]->name() << " with " << ops_[op_j]->name();
            MS_LOG(INFO) << ops_[op_i]->name() << " assigned strategy " << StrategyToString(strategies)
                         << " from ModifyParamSharingOpsStrategy";

            ApplyStrategy(op_i, strategies);
            ++changes;
          }
        }
      }
    }
  }

  return changes;
}

RecStrategyPropagator::RecStrategyPropagator(const std::shared_ptr<Graph> &graph,
                                             const std::vector<std::shared_ptr<OperatorInfo>> &ops,
                                             const std::shared_ptr<std::vector<std::vector<size_t>>> &eli_list,
                                             const std::vector<std::vector<std::string>> &input_tensor_names,
                                             const std::shared_ptr<std::vector<size_t>> &index_list, bool is_training,
                                             const std::vector<std::vector<size_t>> &shared_tensors_ops,
                                             const FuncGraphPtr &root)
    : graph_(graph),
      ops_(ops),
      eli_list_(eli_list),
      input_tensor_names_(input_tensor_names),
      index_list_(index_list),
      is_training_(is_training),
      shared_tensors_ops_(shared_tensors_ops),
      root_(root) {}

size_t RecStrategyPropagator::CopyMainOperatorsStrategy() {
  size_t changes = 0;

  for (size_t i_op = 0; i_op < static_cast<size_t>(index_list_->size()); i_op++) {
    Strategies strategies;
    size_t iter_graph = index_list_->at(i_op);
    if (iter_graph != SIZE_MAX && ops_[i_op]->type() != GET_NEXT) {
      strategies = PrepareStrategy(&graph_->nodes[iter_graph], ops_, i_op, graph_->dyn_shape_tmp_fix);
    }
    if (!strategies.empty()) {
      source_ops_.push_back(i_op);
      ++changes;
    }
    StrategyPtr sp = std::make_shared<Strategy>(0, strategies);
    ops_[i_op]->SetSelectedStrategyAndCost(sp, ops_[i_op]->selected_cost());
  }

  return changes;
}

Dimensions GetInputStrategy(const std::shared_ptr<Graph> &graph, const std::vector<std::shared_ptr<OperatorInfo>> &ops,
                            const std::shared_ptr<std::vector<size_t>> &index_list, size_t i_op,
                            size_t incoming_op_index) {
  Dimensions strategy;
  if (incoming_op_index != SIZE_MAX) {
    auto iter_graph = index_list->at(incoming_op_index);
    if (iter_graph != SIZE_MAX) {
      strategy = CopyIncomingOperatorOutputStrategy(&graph->nodes[iter_graph], ops, i_op, incoming_op_index);
    } else {
      strategy = CopyIncomingOperatorInputStrategy(ops, i_op, incoming_op_index);
    }
  }

  return strategy;
}

size_t RecStrategyPropagator::PropagateFromInputs() { return 0; }

size_t RecStrategyPropagator::PropagateFromOutputs() { return 0; }

void RecStrategyPropagator::GenerateNoStraList() {
  no_stra_op_list_ = std::make_shared<std::vector<size_t>>();
  for (size_t i = 0; i < eli_list_->size(); i++) {
    no_stra_op_list_->push_back(eli_list_->at(i)[0]);
  }
}

void RecStrategyPropagator::FixInvalidStra() {
  for (auto &op : ops_) {
    bool modified = false;
    if (!HasStrategy(op)) {
      continue;
    }
    if (op->type() == FILLV2) {
      continue;
    }
    if (graph_->dyn_shape_tmp_fix && (op->type() == ASSIGN || op->type() == ONEHOT)) {
      continue;
    }
    StrategyPtr old_strategys = op->selected_strategy();
    Strategies new_strategys;
    for (size_t iter_op_inputs = 0; iter_op_inputs < old_strategys->GetInputDim().size(); iter_op_inputs++) {
      Dimensions strategies;
      for (size_t iter_op_input_stra = 0; iter_op_input_stra < op->inputs_shape()[iter_op_inputs].size();
           iter_op_input_stra++) {
        if (graph_->dyn_shape_tmp_fix && op->inputs_shape()[iter_op_inputs][iter_op_input_stra] == -1) {
          strategies.push_back(old_strategys->GetInputDim()[iter_op_inputs][iter_op_input_stra]);
          continue;
        }
        if (op->inputs_shape()[iter_op_inputs][iter_op_input_stra] <
              old_strategys->GetInputDim()[iter_op_inputs][iter_op_input_stra] ||
            op->inputs_shape()[iter_op_inputs][iter_op_input_stra] %
                old_strategys->GetInputDim()[iter_op_inputs][iter_op_input_stra] !=
              0) {
          strategies.push_back(1);
          modified = true;
        } else {
          strategies.push_back(old_strategys->GetInputDim()[iter_op_inputs][iter_op_input_stra]);
        }
      }
      new_strategys.push_back(strategies);
    }
    if (modified) {
      StrategyPtr sp = std::make_shared<Strategy>(0, new_strategys);
      op->SetSelectedStrategyAndCost(sp, op->selected_cost());
      MS_LOG(INFO) << "CHANGE INVALID STRATEGY FOR : " << op->name() << " from " << old_strategys->GetInputDim()
                   << " to " << StrategyToString(new_strategys);
    }
  }
}

void RecStrategyPropagator::AjustToNoTraining() {
  for (auto &op : ops_) {
    // Set back to raw strategy for special node in predict/eval
    if (!is_training_) {
      if ((op->is_last_node()) || (op->type() == VIRTUAL_DATA_SET)) {
        SetBackToRawStrategy(op);
      }
    }
  }
}

void RecStrategyPropagator::GenerateStrategyV1() {
  MS_EXCEPTION_IF_NULL(graph_);
  MS_EXCEPTION_IF_NULL(eli_list_);
  MS_EXCEPTION_IF_NULL(index_list_);

  no_stra_op_list_ = std::make_shared<std::vector<size_t>>();
  for (size_t i = eli_list_->size(); i > 0; i--) {
    no_stra_op_list_->push_back(eli_list_->at(i - 1)[0]);
  }

  size_t changes;
  changes = CopyMainOperatorsStrategy();
  MS_LOG(INFO) << "The strategies of " << changes << " operators are modified after CopyMainOperatorsStrategy.";

  changes = GenerateEliminatedOperatorStrategyForward();
  MS_LOG(INFO) << "The strategies of " << changes
               << " operators are modified after GenerateEliminatedOperatorStrategyForward.";

  changes = GenerateEliminatedOperatorStrategyBackward();
  MS_LOG(INFO) << "The strategies of " << changes
               << " operators are modified after GenerateEliminatedOperatorStrategyBackward.";

  changes = GenerateRemainingOperatorStrategy();
  MS_LOG(INFO) << "The strategies of " << changes << " operators are modified after GenerateRemainingOperatorStrategy.";

  if (graph_->dyn_shape_tmp_fix) {
    for (auto &op : ops_) {
      if (op->type() == ASSIGN) {
        Strategies strategies;
        auto assign_input_0_shape = op->inputs_shape()[0];
        Dimensions assign_input_0_strategy(assign_input_0_shape.size(), 1);
        size_t num_device = LongToSize(g_device_manager->stage_device_num());
        if (assign_input_0_shape[1] > 0 && assign_input_0_shape[1] % num_device == 0) {
          assign_input_0_strategy[1] = num_device;
        }
        for (size_t i = 0; i < op->inputs_shape().size(); i++) {
          strategies.push_back(assign_input_0_strategy);
        }
        StrategyPtr sp = std::make_shared<Strategy>(0, strategies);
        op->SetSelectedStrategyAndCost(sp, op->selected_cost());
      }
    }
  }

  SetParamStrategy();
  changes = ApplyParamStrategy();
  MS_LOG(INFO) << "The strategies of " << changes << " operators are modified after ApplyParamStrategy.";

  FixInvalidStra();
  AjustToNoTraining();
}

size_t RecStrategyPropagator::AssignStandaloneAndBatchParallelOpStrategy() {
  size_t changes = 0;
  for (size_t iter_ops = 0; iter_ops < ops_.size(); iter_ops++) {
    auto pos = ops_[iter_ops]->name().find("Info");
    auto name = ops_[iter_ops]->name().substr(0, pos);
    if (name == STAND_ALONE) {
      Strategies strategies = PrepareStandAlone(ops_[iter_ops]);
      ApplyStrategy(iter_ops, strategies);
      changes++;
      MS_LOG(INFO) << ops_[iter_ops]->name() << " assigned strategy " << StrategyToString(strategies);
      auto iter = find(no_stra_op_list_->begin(), no_stra_op_list_->end(), iter_ops);
      if (iter != no_stra_op_list_->end()) {
        no_stra_op_list_->erase(iter);
      }
    }
    if (name == BATCH_PARALLEL) {
      Strategies strategies;
      auto split_flag_list = ops_[iter_ops]->split_flag_list();
      auto inputs_shape = ops_[iter_ops]->inputs_shape();
      for (size_t i = 0; i < inputs_shape.size(); i++) {
        Shape temp(inputs_shape[i].size(), 1);
        if (split_flag_list[i]) {
          temp[0] = g_device_manager->stage_device_num();
        }
        strategies.push_back(temp);
      }
      ApplyStrategy(iter_ops, strategies);
      changes++;
      MS_LOG(INFO) << ops_[iter_ops]->name() << " assigned strategy " << StrategyToString(strategies);
      auto iter = find(no_stra_op_list_->begin(), no_stra_op_list_->end(), iter_ops);
      if (iter != no_stra_op_list_->end()) {
        no_stra_op_list_->erase(iter);
      }
    }
  }
  return changes;
}

static size_t CalMatmulBatchDimFactor(size_t num_device, const StrategyRec &str) {
  size_t max_shard_num = FloatToSize(1 / str.inputTensor[0].str_h) * FloatToSize(1 / str.inputTensor[0].str_w);
  max_shard_num = max_shard_num < num_device ? max_shard_num : num_device;
  return max_shard_num / (FloatToSize(1 / str.outputTensor.str_h) * FloatToSize(1 / str.outputTensor.str_w));
}

void RecStrategyPropagator::ExtraShardMatmulOnBatchDim() {
  MS_EXCEPTION_IF_NULL(graph_);
  MS_EXCEPTION_IF_NULL(eli_list_);
  MS_EXCEPTION_IF_NULL(index_list_);

  for (size_t i_op = 0; i_op < static_cast<size_t>(index_list_->size()); i_op++) {
    size_t iter_graph = index_list_->at(i_op);
    if (iter_graph == SIZE_MAX || ops_[i_op]->type() != MATMUL) {
      continue;
    }
    Graph::NodeType &node = graph_->nodes[iter_graph];
    size_t matmulBatchDimFactor = CalMatmulBatchDimFactor(g_device_manager->stage_device_num(), node.apply.str);
    if (matmulBatchDimFactor > 1) {
      MS_LOG(INFO) << ops_[i_op]->name() << " matmulBatchDimFactor " << matmulBatchDimFactor;
      node.apply.str.outputTensor.str_h /= matmulBatchDimFactor;
      node.tensor_parm.tensor_str.str_h = node.apply.str.outputTensor.str_h;

      Strategies strategies;
      Dimensions strategy;
      strategy.push_back(static_cast<int64_t>(1.0 / node.apply.str.outputTensor.str_h));
      strategy.push_back(static_cast<int64_t>(1.0 / node.apply.str.outputTensor.str_w));
      strategies.push_back(strategy);

      int64_t stage_id = g_device_manager->stage_id();
      StrategyPtr strategyPtr = NewStrategy(stage_id, strategies);
      ops_[i_op]->set_out_strategy(strategyPtr);
    }
  }
}

void RecStrategyPropagator::GenerateStrategyV3() {
  MS_EXCEPTION_IF_NULL(graph_);
  MS_EXCEPTION_IF_NULL(eli_list_);
  MS_EXCEPTION_IF_NULL(index_list_);

  GenerateNoStraList();
  size_t changes;
  changes = CopyMainOperatorsStrategy();
  MS_LOG(INFO) << "CopyMainOperatorsStrategy has " << changes << "changes";
  AssignStandaloneAndBatchParallelOpStrategy();

  for (auto min_devices = g_device_manager->stage_device_num(); min_devices > 1; min_devices /= SIZE_TWO) {
    size_t pass_changes = 1;
    while (pass_changes > 0) {
      pass_changes = 0;

      changes = GenerateEliminatedOperatorStrategyForward(min_devices);
      MS_LOG(INFO) << "GenerateEliminatedOperatorStrategyForward has " << changes << "changes";

      pass_changes += changes;
      if (changes > 0) continue;

      changes = GenerateEliminatedOperatorStrategyBackward(min_devices);
      MS_LOG(INFO) << "GenerateEliminatedOperatorStrategyBackward has " << changes << "changes";

      pass_changes += changes;
      if (changes > 0) continue;
    }
  }

  changes = GenerateRemainingOperatorStrategy();
  MS_LOG(INFO) << "GenerateRemainingOperatorStrategy has " << changes << "changes";

  changes = ModifyParamSharingOpsStrategy();
  MS_LOG(INFO) << "ModifyParamSharingOpsStrategy has " << changes << "changes";

  FixInvalidStra();
  AjustToNoTraining();
}
}  // namespace parallel
}  // namespace mindspore
