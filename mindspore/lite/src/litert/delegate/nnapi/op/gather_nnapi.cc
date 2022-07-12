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

#include "src/litert/delegate/nnapi/op/gather_nnapi.h"
#include <vector>
#include "src/litert/delegate/nnapi/nnapi_utils.h"
#include "nnacl/op_base.h"

namespace mindspore {
namespace lite {
bool NNAPIGather::IsSupport() { return true; }

int NNAPIGather::InitParams() {
  MS_CHECK_TRUE_RET(in_tensors_.size() == kInputSize2, RET_ERROR);
  auto axis_tensor = in_tensors_.at(2);
  MS_CHECK_TRUE_RET(axis_tensor.IsConst() && axis_tensor.DataType() == DataType::kNumberTypeInt32, RET_ERROR);
  axis_ = *(reinterpret_cast<int *>(axis_tensor.MutableData()));
  return RET_OK;
}

int NNAPIGather::AddOpToNNAPIModel(ANeuralNetworksModel *nnapi_model, std::vector<mindspore::MSTensor> *all_tensors) {
  MS_ASSERT(nnapi_model != nullptr && all_tensors != nullptr);
  OperationCode node_type = ANEURALNETWORKS_GATHER;

  in_tensors_ = {in_tensors_.at(0), in_tensors_.at(1)};  // input and indices.
  if (InitNNAPIOpInOut(*all_tensors) != RET_OK) {
    MS_LOG(ERROR) << "InitNNAPINodeInfo failed.";
    return RET_ERROR;
  }
  if (AddScalarToNNAPIModel<int>(nnapi_model, all_tensors, "axis", DataType::kNumberTypeInt32, axis_) != RET_OK) {
    MS_LOG(ERROR) << "Add axis of gather to NNAPI model failed.";
    return RET_ERROR;
  }
  // adjust input order to {input, axis, indices}.
  input_indices_ = {input_indices_.at(0), input_indices_.at(2), input_indices_.at(1)};
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
