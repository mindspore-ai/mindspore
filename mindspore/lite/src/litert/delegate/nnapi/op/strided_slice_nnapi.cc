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

#include "src/litert/delegate/nnapi/op/strided_slice_nnapi.h"
#include <vector>
#include <unordered_map>
#include "src/litert/delegate/nnapi/nnapi_utils.h"
#include "nnacl/op_base.h"

namespace mindspore {
namespace lite {
bool NNAPIStridedSlice::IsSupport() { return true; }

int NNAPIStridedSlice::InitParams() {
  auto strided_slice = op_primitive_->value_as_StridedSlice();
  MS_CHECK_TRUE_RET(strided_slice != nullptr, RET_ERROR);
  begin_mask_ = strided_slice->begin_mask();
  end_mask_ = strided_slice->end_mask();
  shrink_axis_mask_ = strided_slice->shrink_axis_mask();
  return RET_OK;
}

int NNAPIStridedSlice::AddOpToNNAPIModel(ANeuralNetworksModel *nnapi_model,
                                         std::vector<mindspore::MSTensor> *all_tensors) {
  MS_ASSERT(nnapi_model != nullptr && all_tensors != nullptr);
  OperationCode node_type = ANEURALNETWORKS_STRIDED_SLICE;
  if (InitNNAPIOpInOut(*all_tensors) != RET_OK) {
    MS_LOG(ERROR) << "InitNNAPINodeInfo failed.";
    return RET_ERROR;
  }
  if (AddScalarToNNAPIModel<int>(nnapi_model, all_tensors, "begin_mask", DataType::kNumberTypeInt32, begin_mask_) !=
      RET_OK) {
    MS_LOG(ERROR) << "Add begin mask of strided slice to NNAPI model failed.";
    return RET_ERROR;
  }
  if (AddScalarToNNAPIModel<int>(nnapi_model, all_tensors, "end_mask", DataType::kNumberTypeInt32, end_mask_) !=
      RET_OK) {
    MS_LOG(ERROR) << "Add begin mask of strided slice to NNAPI model failed.";
    return RET_ERROR;
  }
  if (AddScalarToNNAPIModel<int>(nnapi_model, all_tensors, "shrink_axis_mask", DataType::kNumberTypeInt32,
                                 shrink_axis_mask_) != RET_OK) {
    MS_LOG(ERROR) << "Add begin mask of strided slice to NNAPI model failed.";
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
