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

#include "frontend/parallel/ops_info/self_define_shard_info.h"

#include <memory>
#include <utility>
#include <algorithm>

#include "include/utils/parallel_context.h"
#include "ir/value.h"
#include "frontend/parallel/device_manager.h"
#include "frontend/parallel/device_matrix.h"
#include "frontend/parallel/dynamic_creator.h"
#include "frontend/parallel/step_parallel.h"

namespace mindspore {
namespace parallel {

constexpr char FUNC_TYPE[] = "func_type";

Status SelfDefineShardInfo::CheckStrategy(const StrategyPtr &strategy) {
  MS_LOG(ERROR) << "self define shard op " << name_ << " only support config layout rather than strategy";
  return FAILED;
}

Status SelfDefineShardInfo::UnreachableError() {
  MS_LOG(ERROR) << "For self define shard op " << name_ << ", it should not reach this function";
  return FAILED;
}

Status SelfDefineShardInfo::SetCostUnderStrategy(const StrategyPtr &strategy) { return UnreachableError(); }

std::vector<StrategyPtr> SelfDefineShardInfo::GenerateOpStrategies(int64_t stage_id) {
  MS_LOG_WITH_NODE(EXCEPTION, cnode_) << "For self define shard op " << name_ << ", it should not reach this function";
}

Status SelfDefineShardInfo::CheckLayout(const NewShapes &in_shapes, const std::vector<TensorInfoBasePtr> &tensor_info,
                                        const string &name) {
  // Check the number of input shape and input layout
  if (tensor_info.size() != in_shapes.size()) {
    MS_LOG(ERROR) << "The " << name << " shape of " << name_ << " is " << in_shapes.size()
                  << ", which is not equal to the input tensor layout size " << tensor_info.size();
    return FAILED;
  }

  for (size_t i = 0; i < tensor_info.size(); ++i) {
    auto shape = in_shapes.at(i);
    auto info = tensor_info.at(i);
    if (shape->is_list() && info->is_list()) {
      if (shape->size() != info->size()) {
        MS_LOG(ERROR) << "The " << i << "th " << name << " shape of " << name_ << " is " << shape->size()
                      << ", which is not equal to the input tensor layout size " << info->size();
        return FAILED;
      }
    } else if (!shape->is_list() && !info->is_list()) {
      continue;
    } else {
      MS_LOG(ERROR) << "The " << i << "th " << name << " shape of " << name_
                    << "shape and tensor info type are not match, got shape is_list " << shape->is_list()
                    << ", tensor info is_list " << info->is_list();
      return FAILED;
    }
  }

  // Check the device matrix
  std::vector<TensorInfo> squashed_tensor_info;
  for (const auto &info : tensor_info) {
    auto elements = info->GetAllElements();
    squashed_tensor_info.insert(squashed_tensor_info.end(), elements.begin(), elements.end());
  }
  auto prev_dev_arrangment = squashed_tensor_info.at(kIndex0).tensor_layout().device_arrangement_origin().array();
  for (size_t i = 1; i < squashed_tensor_info.size(); ++i) {
    auto current_tensor_layout = squashed_tensor_info.at(i).tensor_layout();
    if ((prev_dev_arrangment != current_tensor_layout.device_arrangement_origin().array()) &&
        current_tensor_layout.device_arrangement_origin().array().size() != 0) {
      MS_LOG(ERROR) << "The device_matrix of input " << i << " is "
                    << current_tensor_layout.device_arrangement_origin().array() << ", which is not equal to previous "
                    << name << " device_matrix " << prev_dev_arrangment;
      return FAILED;
    }
    if (current_tensor_layout.device_arrangement_origin().array().size() != 0) {
      prev_dev_arrangment = current_tensor_layout.device_arrangement_origin().array();
    }
  }
  return SUCCESS;
}

Status SelfDefineShardInfo::CheckInputLayout() {
  // Check self_define_shard attribute
  MS_LOG(WARNING) << "Use self define shard for " << name_
                  << ". User needs to ensure the accuracy and correctness of input/output layout, and framework "
                     "will only do basic check.";
  if (CheckLayout(inputs_shape_new_, inputs_tensor_info_new_, "input") != SUCCESS) {
    if (is_in_layout_propagation_) {
      MS_LOG(INFO) << name_ << " check input layout failed";
    } else {
      MS_LOG(ERROR) << name_ << " check input layout failed";
    }
    return FAILED;
  }
  return SUCCESS;
}

Status SelfDefineShardInfo::CheckOutputLayout() {
  if (CheckLayout(outputs_shape_new_, outputs_tensor_info_new_, "output") != SUCCESS) {
    if (is_in_layout_propagation_) {
      MS_LOG(INFO) << name_ << " check output layout failed";
    } else {
      MS_LOG(ERROR) << name_ << " check output layout failed";
    }
    return FAILED;
  }
  return SUCCESS;
}

Status SelfDefineShardInfo::InferOutputTensorInfo() {
  MS_LOG(ERROR) << "Please pass output layout to " << name_
                << ", self define shard ops does not support infer output tensor layout";
  return FAILED;
}

Status SelfDefineShardInfo::InferAsLossDivisorByLayout() {
  if (!ParallelContext::GetInstance()->loss_repeated_mean()) {
    as_loss_divisor_ = 1;
    return SUCCESS;
  }

  if (outputs_tensor_info_new_.empty()) {
    MS_LOG(ERROR) << name_ << ": The outputs tensor info is empty.";
    return FAILED;
  }
  auto first_out_tensor_info = outputs_tensor_info_new_[0];
  if (first_out_tensor_info->is_list()) {
    MS_LOG(ERROR) << name_ << ": The first output is list, not support yet";
    return FAILED;
  }
  TensorMaps outputs_tensor_map = first_out_tensor_info->GetValue().tensor_layout().tensor_map_before();
  if (outputs_tensor_map.empty()) {
    MS_LOG(INFO) << name_ << ": out_dev_matrix_shape is empty";
    as_loss_divisor_ = stage_device_size_;
    MS_LOG(INFO) << name_ << ": The output is a scalar, use the dev size " << as_loss_divisor_ << ", loss divisor.";
    return SUCCESS;
  }

  auto out_dev_matrix_shape = first_out_tensor_info->GetValue().tensor_layout().device_arrangement_origin().array();
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

Status SelfDefineShardInfo::InferOperatorVectorListForShapeList(const TensorInfoBasePtr &tensor_info,
                                                                const int64_t &input_idx,
                                                                std::vector<OperatorVectorBasePtr> *mirror_ops_new,
                                                                bool *group_is_empty) {
  std::vector<RankList> repeated_rank_lists;
  for (size_t i = 0; i < tensor_info->size(); ++i) {
    auto info = tensor_info->GetElement(SizeToLong(i));
    if (info->is_list()) {
      MS_LOG(ERROR) << "For " << name_ << ": does not support tuple in tuple to infer mirror ops yet ";
      return FAILED;
    }
    auto tensor_info_value = info->GetValue();
    if (tensor_info_value == TensorInfo()) {
      MS_LOG(INFO) << "In tuple there is a TensorInfo(), break";
      repeated_rank_lists.clear();
      break;
    }
    repeated_rank_lists.push_back(tensor_info_value.tensor_layout().InferRepeatedGroup());
  }
  if (repeated_rank_lists.empty()) {
    std::vector<OperatorVectorBasePtr> op_vector_list(tensor_info->size(),
                                                      std::make_shared<OperatorVectorValue>(OperatorVector()));
    mirror_ops_new->emplace_back(std::make_shared<OperatorVectorList>(op_vector_list));
    return SUCCESS;
  }
  std::vector<OperatorVectorBasePtr> mirror_ops;
  for (const auto &repeated_rank_list : repeated_rank_lists) {
    OperatorVector mirror_op;
    if (repeated_rank_list.size() == 1) {
      MS_LOG(INFO) << name_ << ": The mirror group is empty, the input index is " << input_idx;
      mirror_ops.emplace_back(std::make_shared<OperatorVectorValue>(mirror_op));
      continue;
    }

    Group mirror_group;
    if (g_device_manager->CreateGroup(repeated_rank_list, &mirror_group) != SUCCESS) {
      MS_LOG(ERROR) << name_
                    << ": Create communication group by tensor_map failed, the rank_list is: " << repeated_rank_list
                    << ", the full_name of node is: " << cnode_->fullname_with_scope();
      return FAILED;
    }
    *group_is_empty = false;
    mirror_op = CreateMirrorOps(mirror_group.name(), mirror_group.GetDevNum());
    mirror_ops.push_back(std::make_shared<OperatorVectorValue>(mirror_op));
  }
  mirror_ops_new->emplace_back(std::make_shared<OperatorVectorList>(mirror_ops));
  return SUCCESS;
}

Status SelfDefineShardInfo::InferOperatorVectorValueForShapeValue(const TensorInfoBasePtr &tensor_info,
                                                                  const int64_t &input_idx,
                                                                  std::vector<OperatorVectorBasePtr> *mirror_ops_new,
                                                                  MirrorOps *mirror_ops, bool *group_is_empty) {
  auto input_tensor_layout = tensor_info->GetValue().tensor_layout();
  auto repeated_rank_list = input_tensor_layout.InferRepeatedGroup();

  OperatorVector mirror_op;
  if (repeated_rank_list.size() == 1) {
    MS_LOG(INFO) << name_ << ": The mirror group is empty, the input index is " << input_idx;
    mirror_ops_new->emplace_back(std::make_shared<OperatorVectorValue>(mirror_op));
    mirror_ops->emplace_back(mirror_op);
    return SUCCESS;
  }

  Group mirror_group;
  if (g_device_manager->CreateGroup(repeated_rank_list, &mirror_group) != SUCCESS) {
    MS_LOG(ERROR) << name_
                  << ": Create communication group by tensor_map failed, the rank_list is: " << repeated_rank_list
                  << ", the full_name of node is: " << cnode_->fullname_with_scope();
    return FAILED;
  }
  *group_is_empty = false;
  mirror_op = CreateMirrorOps(mirror_group.name(), mirror_group.GetDevNum());
  mirror_ops_new->emplace_back(std::make_shared<OperatorVectorValue>(mirror_op));
  mirror_ops->emplace_back(mirror_op);
  return SUCCESS;
}

Status SelfDefineShardInfo::InferMirrorOpsByLayout() {
  mirror_ops_.clear();
  if (inputs_shape_new_.empty()) {
    MS_LOG(INFO) << name_ << ": The inputs size is empty";
    return SUCCESS;
  }

  bool group_is_empty = true;
  for (size_t i = 0; i < inputs_tensor_info_new_.size(); ++i) {
    auto tensor_info = inputs_tensor_info_new_[i];
    if (tensor_info->is_list()) {
      if (InferOperatorVectorListForShapeList(tensor_info, SizeToLong(i), &mirror_ops_new_, &group_is_empty) !=
          SUCCESS) {
        MS_LOG(ERROR) << name_ << ": InferOperatorVectorListForShapeList failed";
        return FAILED;
      }
      OperatorVector temp_mirror_op;
      mirror_ops_.push_back(temp_mirror_op);
    } else {
      if (tensor_info->GetValue() == TensorInfo()) {
        mirror_ops_new_.emplace_back(std::make_shared<OperatorVectorValue>(OperatorVector()));
        mirror_ops_.emplace_back(OperatorVector());
        continue;
      }
      if (InferOperatorVectorValueForShapeValue(tensor_info, SizeToLong(i), &mirror_ops_new_, &mirror_ops_,
                                                &group_is_empty) != SUCCESS) {
        MS_LOG(ERROR) << name_ << ": InferOperatorVectorValueForShapeValue failed";
        return FAILED;
      }
    }
  }
  if (group_is_empty) {
    mirror_ops_new_.clear();
    MS_LOG(INFO) << name_ << ": No need to insert mirror ops";
  }
  return SUCCESS;
}

Status CustomInfo::CheckInputLayout() {
  // Check self_define_shard attribute
  MS_LOG(WARNING) << "For custom op " << name_
                  << ". User needs to ensure the accuracy and correctness of input/output layout, and framework "
                     "will only do basic check. If communication operator is needed to ensure the accuracy, the custom "
                     "ops does not support this scenario yet";
  if (input_value_.size() != inputs_tensor_info_new_.size()) {
    MS_LOG(INFO) << "input value size is " << input_value_.size() << ", input tensor info size is "
                 << inputs_tensor_info_new_.size() << ". They are not equal, which means that scalar in it";
  }
  if (CheckLayout(inputs_shape_new_, inputs_tensor_info_new_, "input") != SUCCESS) {
    if (is_in_layout_propagation_) {
      MS_LOG(INFO) << name_ << " check input layout failed";
    } else {
      MS_LOG(ERROR) << name_ << " check input layout failed";
    }
    return FAILED;
  }
  return SUCCESS;
}

Status CustomInfo::GetAttrs() {
  auto func_type_iter = attrs_.find(FUNC_TYPE);
  if (func_type_iter == attrs_.end()) {
    MS_LOG(ERROR) << name_ << ": Can not find func_type attribute";
    return FAILED;
  }
  MS_EXCEPTION_IF_NULL(func_type_iter->second);
  auto func_type = func_type_iter->second->cast<StringImmPtr>();
  MS_EXCEPTION_IF_NULL(func_type);
  auto func_type_value = func_type->value();
  if ((func_type_value != "pyfunc") && func_type_value != "aot") {
    MS_LOG(ERROR) << name_ << ": The func_type attribute must be 'pyfunc' or 'aot', but got " << func_type;
  }
  return SUCCESS;
}

REGISTER(SelfDefineShardInfo);
REGISTER(CustomInfo);
}  // namespace parallel
}  // namespace mindspore
