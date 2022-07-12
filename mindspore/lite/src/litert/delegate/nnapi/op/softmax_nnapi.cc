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

#include "src/litert/delegate/nnapi/op/softmax_nnapi.h"
#include <vector>
#include <unordered_map>
#include "src/litert/delegate/nnapi/nnapi_utils.h"
#include "nnacl/op_base.h"

namespace mindspore {
namespace lite {
bool NNAPISoftmax::IsSupport() {
  auto input = in_tensors_.front();
  if (nnapi_->android_sdk_version >= ANEURALNETWORKS_FEATURE_LEVEL_3) {
    return true;
  }
  return input.Shape().size() == DIMENSION_2D || input.Shape().size() == DIMENSION_4D;
}

int NNAPISoftmax::InitParams() {
  auto softmax = op_primitive_->value_as_Softmax();
  MS_CHECK_TRUE_RET(softmax != nullptr, RET_ERROR);
  auto axis_data = softmax->axis();
  MS_CHECK_TRUE_RET(axis_data != nullptr && axis_data->size() == 1, RET_ERROR);
  axis_ = axis_data->data()[0];
  return RET_OK;
}

int NNAPISoftmax::AddOpToNNAPIModel(ANeuralNetworksModel *nnapi_model, std::vector<mindspore::MSTensor> *all_tensors) {
  MS_ASSERT(nnapi_model != nullptr && all_tensors != nullptr);
  OperationCode node_type = ANEURALNETWORKS_SOFTMAX;
  if (InitNNAPIOpInOut(*all_tensors) != RET_OK) {
    MS_LOG(ERROR) << "InitNNAPINodeInfo failed.";
    return RET_ERROR;
  }

  if (AddScalarToNNAPIModel<float>(nnapi_model, all_tensors, "scale", DataType::kNumberTypeFloat32, 1) != RET_OK) {
    MS_LOG(ERROR) << "Add scale of softmax to NNAPI model failed.";
    return RET_ERROR;
  }
  if (AddScalarToNNAPIModel<int>(nnapi_model, all_tensors, "axis", DataType::kNumberTypeInt32, axis_) != RET_OK) {
    MS_LOG(ERROR) << "Add axis of softmax to NNAPI model failed.";
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
