/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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
#include <algorithm>

#include "minddata/dataset/kernels/ir/vision/random_select_subpolicy_ir.h"

#ifndef ENABLE_ANDROID
#include "minddata/dataset/engine/serdes.h"
#include "minddata/dataset/kernels/image/random_select_subpolicy_op.h"
#endif

#include "minddata/dataset/kernels/ir/validators.h"

namespace mindspore {
namespace dataset {
namespace vision {
#ifndef ENABLE_ANDROID
// RandomSelectSubpolicyOperation.
RandomSelectSubpolicyOperation::RandomSelectSubpolicyOperation(
  const std::vector<std::vector<std::pair<std::shared_ptr<TensorOperation>, double>>> &policy)
    : TensorOperation(true), policy_(policy) {}

RandomSelectSubpolicyOperation::~RandomSelectSubpolicyOperation() = default;

std::string RandomSelectSubpolicyOperation::Name() const { return kRandomSelectSubpolicyOperation; }

Status RandomSelectSubpolicyOperation::ValidateParams() {
  if (policy_.empty()) {
    std::string err_msg = "RandomSelectSubpolicy: policy must not be empty";
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }
  for (int32_t i = 0; i < policy_.size(); i++) {
    if (policy_[i].empty()) {
      std::string err_msg = "RandomSelectSubpolicy: policy[" + std::to_string(i) + "] must not be empty";
      MS_LOG(ERROR) << err_msg;
      RETURN_STATUS_SYNTAX_ERROR(err_msg);
    }
    for (int32_t j = 0; j < policy_[i].size(); j++) {
      if (policy_[i][j].first == nullptr) {
        std::string transform_pos = "[" + std::to_string(i) + "]" + "[" + std::to_string(j) + "]";
        std::string err_msg = "RandomSelectSubpolicy: transform in policy" + transform_pos + " must not be null";
        MS_LOG(ERROR) << err_msg;
        RETURN_STATUS_SYNTAX_ERROR(err_msg);
      } else {
        RETURN_IF_NOT_OK(policy_[i][j].first->ValidateParams());
      }
      if (policy_[i][j].second < 0.0 || policy_[i][j].second > 1.0) {
        std::string transform_pos = "[" + std::to_string(i) + "]" + "[" + std::to_string(j) + "]";
        std::string err_msg = "RandomSelectSubpolicy: probability of transform in policy" + transform_pos +
                              " must be between 0.0 and 1.0, got: " + std::to_string(policy_[i][j].second);
        MS_LOG(ERROR) << err_msg;
        RETURN_STATUS_SYNTAX_ERROR(err_msg);
      }
    }
  }
  return Status::OK();
}

std::shared_ptr<TensorOp> RandomSelectSubpolicyOperation::Build() {
  std::vector<Subpolicy> policy_tensor_ops;
  for (int32_t i = 0; i < policy_.size(); i++) {
    Subpolicy sub_policy_tensor_ops;
    for (int32_t j = 0; j < policy_[i].size(); j++) {
      sub_policy_tensor_ops.push_back(std::make_pair(policy_[i][j].first->Build(), policy_[i][j].second));
    }
    policy_tensor_ops.push_back(sub_policy_tensor_ops);
  }
  std::shared_ptr<RandomSelectSubpolicyOp> tensor_op = std::make_shared<RandomSelectSubpolicyOp>(policy_tensor_ops);
  return tensor_op;
}

Status RandomSelectSubpolicyOperation::to_json(nlohmann::json *out_json) {
  auto policy_tensor_ops = nlohmann::json::array();
  for (int32_t i = 0; i < policy_.size(); i++) {
    auto sub_policy_tensor_ops = nlohmann::json::array();
    for (int32_t j = 0; j < policy_[i].size(); j++) {
      nlohmann::json policy, args;
      auto tensor_op = policy_[i][j].first;
      RETURN_IF_NOT_OK(tensor_op->to_json(&args));
      policy["tensor_op"]["tensor_op_params"] = args;
      policy["tensor_op"]["tensor_op_name"] = tensor_op->Name();
      policy["prob"] = policy_[i][j].second;
      sub_policy_tensor_ops.push_back(policy);
    }
    policy_tensor_ops.push_back(sub_policy_tensor_ops);
  }
  (*out_json)["policy"] = policy_tensor_ops;
  return Status::OK();
}

Status RandomSelectSubpolicyOperation::from_json(nlohmann::json op_params,
                                                 std::shared_ptr<TensorOperation> *operation) {
  CHECK_FAIL_RETURN_UNEXPECTED(op_params.find("policy") != op_params.end(), "Failed to find policy");
  nlohmann::json policy_json = op_params["policy"];
  std::vector<std::vector<std::pair<std::shared_ptr<TensorOperation>, double>>> policy;
  std::vector<std::pair<std::shared_ptr<TensorOperation>, double>> policy_items;
  for (nlohmann::json item : policy_json) {
    for (nlohmann::json item_pair : item) {
      CHECK_FAIL_RETURN_UNEXPECTED(item_pair.find("prob") != item_pair.end(), "Failed to find prob");
      CHECK_FAIL_RETURN_UNEXPECTED(item_pair.find("tensor_op") != item_pair.end(), "Failed to find tensor_op");
      std::vector<std::shared_ptr<TensorOperation>> operations;
      std::pair<std::shared_ptr<TensorOperation>, double> policy_pair;
      std::shared_ptr<TensorOperation> operation;
      nlohmann::json tensor_op_json;
      double prob = item_pair["prob"];
      tensor_op_json.push_back(item_pair["tensor_op"]);
      RETURN_IF_NOT_OK(Serdes::ConstructTensorOps(tensor_op_json, &operations));
      CHECK_FAIL_RETURN_UNEXPECTED(operations.size() == 1, "There should be only 1 tensor operation");
      policy_pair = std::make_pair(operations[0], prob);
      policy_items.push_back(policy_pair);
    }
    policy.push_back(policy_items);
  }
  *operation = std::make_shared<vision::RandomSelectSubpolicyOperation>(policy);
  return Status::OK();
}
#endif
}  // namespace vision
}  // namespace dataset
}  // namespace mindspore
