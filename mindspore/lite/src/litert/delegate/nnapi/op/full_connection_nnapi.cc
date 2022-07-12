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

#include "src/litert/delegate/nnapi/op/full_connection_nnapi.h"
#include <vector>
#include <unordered_map>
#include "src/litert/delegate/nnapi/nnapi_utils.h"
#include "nnacl/op_base.h"

namespace mindspore {
namespace lite {
bool NNAPIFullConnection::IsSupport() {
  auto weight = in_tensors_.at(1);
  return weight.Shape().size() == DIMENSION_2D;
}

int NNAPIFullConnection::InitParams() {
  has_bias_ = in_tensors_.size() == kInputSize2;
  auto full_connection = op_primitive_->value_as_FullConnection();
  MS_CHECK_TRUE_RET(full_connection != nullptr, RET_ERROR);
  act_type_ = full_connection->activation_type();
  return RET_OK;
}

int NNAPIFullConnection::AddOpToNNAPIModel(ANeuralNetworksModel *nnapi_model,
                                           std::vector<mindspore::MSTensor> *all_tensors) {
  MS_ASSERT(nnapi_model != nullptr && all_tensors != nullptr);
  OperationCode node_type = ANEURALNETWORKS_FULLY_CONNECTED;
  if (InitNNAPIOpInOut(*all_tensors) != RET_OK) {
    MS_LOG(ERROR) << "InitNNAPINodeInfo failed.";
    return RET_ERROR;
  }
  if (!has_bias_) {
    auto weight_type = in_tensors_.at(1).DataType();
    auto bias_type = (weight_type != DataType::kNumberTypeInt8 && weight_type != DataType::kNumberTypeUInt8)
                       ? weight_type
                       : DataType::kNumberTypeInt32;
    auto num_units = in_tensors_.at(1).Shape().at(0);
    MSTensorInfo bias_info{op_name_ + "_bias", bias_type, {num_units}, nullptr, 0};
    if (AddTensorToNNAPIModel(nnapi_model, all_tensors, bias_info) != RET_OK) {
      MS_LOG(ERROR) << "NNAPI does not support full connection without bias, and create zero-value tenosr failed.";
      return RET_ERROR;
    }
  }
  if (AddScalarToNNAPIModel<int>(nnapi_model, all_tensors, "act_type", DataType::kNumberTypeInt32, act_type_) !=
      RET_OK) {
    MS_LOG(ERROR) << "Add act type of full connection to NNAPI model failed: " << op_name_;
    return RET_ERROR;
  }
  if (nnapi_->ANeuralNetworksModel_addOperation(nnapi_model, node_type, input_indices_.size(), input_indices_.data(),
                                                output_indices_.size(),
                                                output_indices_.data()) != ANEURALNETWORKS_NO_ERROR) {
    MS_LOG(ERROR) << "Add operation to NNAPI model failed.";
    return RET_ERROR;
  }
  return RET_OK;
}
}  // namespace lite
}  // namespace mindspore
