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

#include "src/litert/delegate/nnapi/op/instance_norm_nnapi.h"
#include <vector>
#include <unordered_map>
#include "src/litert/delegate/nnapi/nnapi_utils.h"
#include "nnacl/op_base.h"

namespace mindspore {
namespace lite {
bool NNAPIInstanceNorm::IsSupport() { return true; }

int NNAPIInstanceNorm::InitParams() {
  MS_CHECK_TRUE_RET(in_tensors_.size() == kInputSize2, RET_ERROR);
  auto scale_tensor = in_tensors_.at(1);
  auto bias_tensor = in_tensors_.at(2);
  MS_CHECK_TRUE_RET(scale_tensor.IsConst() && bias_tensor.IsConst(), RET_ERROR);
  MS_CHECK_TRUE_RET(
    scale_tensor.DataType() == DataType::kNumberTypeFloat32 && bias_tensor.DataType() == DataType::kNumberTypeFloat32,
    RET_ERROR);
  scale_ = *(reinterpret_cast<float *>(scale_tensor.MutableData()));
  bias_ = *(reinterpret_cast<float *>(bias_tensor.MutableData()));
  auto instance_norm = op_primitive_->value_as_InstanceNorm();
  MS_CHECK_TRUE_RET(instance_norm != nullptr, RET_ERROR);
  epsilon_ = instance_norm->epsilon();
  return RET_OK;
}

int NNAPIInstanceNorm::AddOpToNNAPIModel(ANeuralNetworksModel *nnapi_model,
                                         std::vector<mindspore::MSTensor> *all_tensors) {
  MS_ASSERT(nnapi_model != nullptr && all_tensors != nullptr);
  OperationCode node_type = ANEURALNETWORKS_INSTANCE_NORMALIZATION;
  in_tensors_ = {in_tensors_.front()};
  if (InitNNAPIOpInOut(*all_tensors) != RET_OK) {
    MS_LOG(ERROR) << "InitNNAPINodeInfo failed.";
    return RET_ERROR;
  }
  if (AddScalarToNNAPIModel<float>(nnapi_model, all_tensors, "gamma", DataType::kNumberTypeFloat32, scale_) != RET_OK) {
    MS_LOG(ERROR) << "Add gamma of instance norm to NNAPI model failed.";
    return RET_ERROR;
  }
  if (AddScalarToNNAPIModel<float>(nnapi_model, all_tensors, "beta", DataType::kNumberTypeFloat32, bias_) != RET_OK) {
    MS_LOG(ERROR) << "Add beta of instance norm to NNAPI model failed.";
    return RET_ERROR;
  }
  if (AddScalarToNNAPIModel<float>(nnapi_model, all_tensors, "epsilon", DataType::kNumberTypeFloat32, epsilon_) !=
      RET_OK) {
    MS_LOG(ERROR) << "Add epsilon of instance norm to NNAPI model failed.";
    return RET_ERROR;
  }
  // set nchw to an input of nnapi node.
  if (AddScalarToNNAPIModel<bool>(nnapi_model, all_tensors, "nchw", DataType::kNumberTypeBool, false) != RET_OK) {
    MS_LOG(ERROR) << "set nchw format for instance norm to NNAPI model failed.";
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
