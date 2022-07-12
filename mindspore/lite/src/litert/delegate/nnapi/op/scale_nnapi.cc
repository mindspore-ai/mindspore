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

#include "src/litert/delegate/nnapi/op/scale_nnapi.h"
#include <vector>
#include <unordered_map>
#include "src/litert/delegate/nnapi/nnapi_utils.h"
#include "nnacl/op_base.h"

namespace mindspore {
namespace lite {
bool NNAPIScale::IsSupport() {
  auto input_shape = in_tensors_.front().Shape();
  bool valid_input = std::find(input_shape.begin(), input_shape.end(), -1) == input_shape.end();
  return valid_input;
}

int NNAPIScale::InitParams() {
  has_bias_ = in_tensors_.size() == kInputSize2;
  auto scale = op_primitive_->value_as_ScaleFusion();
  MS_CHECK_TRUE_RET(scale != nullptr, RET_ERROR);
  act_type_ = scale->activation_type();
  return RET_OK;
}

int NNAPIScale::AddOpToNNAPIModel(ANeuralNetworksModel *nnapi_model, std::vector<mindspore::MSTensor> *all_tensors) {
  MS_ASSERT(nnapi_model != nullptr && all_tensors != nullptr);
  /* scale ==> mul + add */
  // add mul
  auto inputs = in_tensors_;
  auto outputs = out_tensors_;
  OperationCode node_type = ANEURALNETWORKS_MUL;
  if (has_bias_) {
    in_tensors_ = {inputs.at(0), inputs.at(1)};
    auto tensor =
      MSTensor::CreateTensor("mul_out", in_tensors_.front().DataType(), in_tensors_.front().Shape(), nullptr, 0);
    MS_CHECK_TRUE_RET(tensor != nullptr, RET_ERROR);
    if (AddNNAPIOperand(nnapi_model, *tensor, static_cast<int>(all_tensors->size())) != RET_OK) {
      MS_LOG(ERROR) << "Add temporary output for scale failed.";
      delete tensor;
      return RET_ERROR;
    }
    all_tensors->push_back(*tensor);
    op_attribute_tensors_.push_back(tensor);
    out_tensors_ = {*tensor};
  }
  if (InitNNAPIOpInOut(*all_tensors) != RET_OK) {
    MS_LOG(ERROR) << "InitNNAPINodeInfo failed.";
    return RET_ERROR;
  }
  if (AddScalarToNNAPIModel<int>(nnapi_model, all_tensors, "act_type", DataType::kNumberTypeInt32, 0) != RET_OK) {
    MS_LOG(ERROR) << "Add act_type of scale to NNAPI model failed: " << op_name_;
    return RET_ERROR;
  }
  if (nnapi_->ANeuralNetworksModel_addOperation(nnapi_model, node_type, input_indices_.size(), input_indices_.data(),
                                                output_indices_.size(), output_indices_.data()) != RET_OK) {
    MS_LOG(ERROR) << "Add operation to NNAPI model failed.";
    return RET_ERROR;
  }

  // add bias
  if (has_bias_) {
    node_type = ANEURALNETWORKS_ADD;
    in_tensors_ = {out_tensors_.front(), inputs.at(2)};
    out_tensors_ = outputs;
    if (InitNNAPIOpInOut(*all_tensors) != RET_OK) {
      MS_LOG(ERROR) << "InitNNAPINodeInfo failed.";
      return RET_ERROR;
    }
    if (AddScalarToNNAPIModel<int>(nnapi_model, all_tensors, "act_type", DataType::kNumberTypeInt32, act_type_) !=
        RET_OK) {
      MS_LOG(ERROR) << "Add axis of softmax to NNAPI model failed.";
      return RET_ERROR;
    }
    if (nnapi_->ANeuralNetworksModel_addOperation(nnapi_model, node_type, input_indices_.size(), input_indices_.data(),
                                                  output_indices_.size(),
                                                  output_indices_.data()) != ANEURALNETWORKS_NO_ERROR) {
      MS_LOG(ERROR) << "Add operation to NNAPI model failed.";
      return RET_ERROR;
    }
  }
  return RET_OK;
}
}  // namespace lite
}  // namespace mindspore
