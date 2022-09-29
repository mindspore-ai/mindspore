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

#include "src/litert/delegate/nnapi/op/padding_nnapi.h"
#include <unordered_map>
#include "src/litert/delegate/nnapi/nnapi_utils.h"
#include "nnacl/op_base.h"

namespace mindspore {
namespace lite {
bool NNAPIPadding::IsSupport() {
  if (nnapi_->android_sdk_version < ANEURALNETWORKS_FEATURE_LEVEL_2) {
    return false;
  } else if (const_value_ != 0 && nnapi_->android_sdk_version < ANEURALNETWORKS_FEATURE_LEVEL_3) {
    return false;
  }
  return true;
}

int NNAPIPadding::InitParams() {
  auto pad = op_primitive_->value_as_PadFusion();
  MS_ASSERT(pad != nullptr);
  pad_mode_ = pad->padding_mode();
  const_value_ = pad->constant_value();
  return RET_OK;
}

int NNAPIPadding::AddOpToNNAPIModel(ANeuralNetworksModel *nnapi_model, std::vector<mindspore::MSTensor> *all_tensors) {
  MS_ASSERT(nnapi_model != nullptr && all_tensors != nullptr);
  OperationCode node_type = const_value_ == 0 ? ANEURALNETWORKS_PAD : ANEURALNETWORKS_PAD_V2;

  if (InitNNAPIOpInOut(*all_tensors) != RET_OK) {
    MS_LOG(ERROR) << "InitNNAPINodeInfo failed.";
    return RET_ERROR;
  }
  if (const_value_ != 0) {
    if (AddScalarToNNAPIModel<int>(nnapi_model, all_tensors, "const_value", DataType::kNumberTypeInt32, const_value_) !=
        RET_OK) {
      MS_LOG(ERROR) << "Add constant value for pad to NNAPI model failed: " << op_name_;
      return RET_ERROR;
    }
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
