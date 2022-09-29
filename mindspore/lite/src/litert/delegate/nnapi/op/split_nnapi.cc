/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include "src/litert/delegate/nnapi/op/split_nnapi.h"
#include <algorithm>
#include <vector>
#include <unordered_map>
#include "src/litert/delegate/nnapi/nnapi_utils.h"
#include "nnacl/op_base.h"

namespace mindspore {
namespace lite {
bool NNAPISplit::IsSupport() {
  if (!size_splits_.empty() &&
      std::any_of(size_splits_.begin(), size_splits_.end(), [&](int x) { return x != size_splits_.front(); })) {
    return false;
  }
  auto shape = in_tensors_.front().Shape();
  if (std::find(shape.begin(), shape.end(), -1) != shape.end()) {
    return false;
  }
  int axis = axis_ < 0 ? axis_ + static_cast<int>(shape.size()) : axis_;
  MS_CHECK_TRUE_RET(axis < shape.size(), RET_ERROR);
  if (size_splits_.front() <= 0 || shape.at(axis) % size_splits_.front() != 0) {
    return false;
  }
  split_num_ = shape.at(axis) / size_splits_.front();
  return true;
}

int NNAPISplit::InitParams() {
  auto split = op_primitive_->value_as_Split();
  MS_CHECK_TRUE_RET(split != nullptr, RET_ERROR);
  axis_ = split->axis();
  split_num_ = split->output_num();
  auto split_sizes_vector = split->size_splits();
  if (split_sizes_vector != nullptr && split_sizes_vector->size() <= split_num_) {
    (void)std::transform(split_sizes_vector->begin(), split_sizes_vector->end(), std::back_inserter(size_splits_),
                         [](int x) { return x; });
  }
  return RET_OK;
}

int NNAPISplit::AddOpToNNAPIModel(ANeuralNetworksModel *nnapi_model, std::vector<mindspore::MSTensor> *all_tensors) {
  MS_ASSERT(nnapi_model != nullptr && all_tensors != nullptr);
  OperationCode node_type = ANEURALNETWORKS_SPLIT;
  if (InitNNAPIOpInOut(*all_tensors) != RET_OK) {
    MS_LOG(ERROR) << "InitNNAPINodeInfo failed.";
    return RET_ERROR;
  }
  if (AddScalarToNNAPIModel<int>(nnapi_model, all_tensors, "axis", DataType::kNumberTypeInt32, axis_) != RET_OK) {
    MS_LOG(ERROR) << "Add axis of split to NNAPI model failed.";
    return RET_ERROR;
  }
  if (AddScalarToNNAPIModel<int>(nnapi_model, all_tensors, "split_num", DataType::kNumberTypeInt32, split_num_) !=
      RET_OK) {
    MS_LOG(ERROR) << "Add number of split to NNAPI model failed.";
    return RET_ERROR;
  }
  if (nnapi_->ANeuralNetworksModel_addOperation(nnapi_model, node_type, input_indices_.size(), input_indices_.data(),
                                                output_indices_.size(),
                                                output_indices_.data()) != ANEURALNETWORKS_NO_ERROR) {
    MS_LOG(ERROR) << "Add operation to NNAPI model failed: " << op_name_;
    return RET_ERROR;
  }
  return RET_OK;
}
}  // namespace lite
}  // namespace mindspore
