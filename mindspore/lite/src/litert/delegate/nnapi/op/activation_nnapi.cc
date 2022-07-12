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

#include "src/litert/delegate/nnapi/op/activation_nnapi.h"
#include <unordered_map>
#include "src/litert/delegate/nnapi/nnapi_utils.h"
#include "nnacl/op_base.h"

namespace mindspore {
namespace lite {
namespace {
std::unordered_map<schema::ActivationType, OperationCode> activation_type = {
  {schema::ActivationType_RELU, ANEURALNETWORKS_RELU},
  {schema::ActivationType_RELU6, ANEURALNETWORKS_RELU6},
  {schema::ActivationType_HSWISH, ANEURALNETWORKS_HARD_SWISH},
  {schema::ActivationType_SIGMOID, ANEURALNETWORKS_LOGISTIC},
  {schema::ActivationType_TANH, ANEURALNETWORKS_TANH},
};
}  // namespace
bool NNAPIActivation::IsSupport() {
  if (activation_type.find(act_type_) == activation_type.end()) {
    MS_LOG(WARNING) << "Unsupported activation type: " << act_type_;
    return false;
  }
  return in_tensors_.front().Shape().size() <= DIMENSION_4D;
}

int NNAPIActivation::InitParams() {
  act_type_ = op_primitive_->value_as_Activation()->activation_type();
  return RET_OK;
}

int NNAPIActivation::AddOpToNNAPIModel(ANeuralNetworksModel *nnapi_model,
                                       std::vector<mindspore::MSTensor> *all_tensors) {
  MS_ASSERT(nnapi_model != nullptr && all_tensors != nullptr);
  OperationCode node_type = activation_type.at(act_type_);
  if (InitNNAPIOpInOut(*all_tensors) != RET_OK) {
    MS_LOG(ERROR) << "InitNNAPINodeInfo failed.";
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
