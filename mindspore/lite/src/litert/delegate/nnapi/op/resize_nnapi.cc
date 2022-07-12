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

#include "src/litert/delegate/nnapi/op/resize_nnapi.h"
#include <vector>
#include <unordered_map>
#include "src/litert/delegate/nnapi/nnapi_utils.h"
#include "nnacl/op_base.h"

namespace mindspore {
namespace lite {
bool NNAPIResize::IsSupport() {
  auto input = in_tensors_.front();
  bool valid_input = input.Shape().size() == DIMENSION_4D;
  bool valid_method = (method_ == static_cast<int>(schema::ResizeMethod_LINEAR) ||
                       method_ == static_cast<int>(schema::ResizeMethod_NEAREST));
  return valid_input && valid_method;
}

int NNAPIResize::InitParams() {
  auto resize = op_primitive_->value_as_Resize();
  MS_CHECK_TRUE_RET(resize != nullptr, RET_ERROR);
  method_ = static_cast<int>(resize->method());
  height_.ix_ = resize->new_height();
  width_.ix_ = resize->new_width();
  if (in_tensors_.size() == DIMENSION_2D) {
    auto new_size_tensor = in_tensors_.at(1);
    if (!new_size_tensor.IsConst() ||
        (new_size_tensor.ElementNum() != DIMENSION_2D && new_size_tensor.ElementNum() != DIMENSION_4D)) {
      MS_LOG(ERROR) << "The new size of resize must be const value.";
      return RET_ERROR;
    }
    data_type_ = static_cast<int>(new_size_tensor.DataType());
    auto new_size = new_size_tensor.MutableData();
    int height_idx = new_size_tensor.ElementNum() == DIMENSION_2D ? 0 : 1;
    int width_idx = new_size_tensor.ElementNum() == DIMENSION_2D ? 1 : 2;
    switch (static_cast<DataType>(data_type_)) {
      case DataType::kNumberTypeInt32:
        height_.ix_ = reinterpret_cast<int *>(new_size)[height_idx];
        width_.ix_ = reinterpret_cast<int *>(new_size)[width_idx];
        break;
      case DataType::kNumberTypeFloat32:
        height_.fx_ = reinterpret_cast<float *>(new_size)[height_idx];
        width_.fx_ = reinterpret_cast<float *>(new_size)[width_idx];
        break;
      default:
        MS_LOG(ERROR) << "The new size should be an int value or a float value.";
        return RET_ERROR;
    }
  }
  return RET_OK;
}

int NNAPIResize::AddOpToNNAPIModel(ANeuralNetworksModel *nnapi_model, std::vector<mindspore::MSTensor> *all_tensors) {
  MS_ASSERT(nnapi_model != nullptr && all_tensors != nullptr);
  OperationCode node_type = method_ == static_cast<int>(schema::ResizeMethod_LINEAR)
                              ? ANEURALNETWORKS_RESIZE_BILINEAR
                              : ANEURALNETWORKS_RESIZE_NEAREST_NEIGHBOR;
  in_tensors_ = {in_tensors_.front()};
  if (InitNNAPIOpInOut(*all_tensors) != RET_OK) {
    MS_LOG(ERROR) << "InitNNAPINodeInfo failed.";
    return RET_ERROR;
  }
  switch (static_cast<DataType>(data_type_)) {
    case DataType::kNumberTypeInt32:
      if (AddScalarToNNAPIModel<int>(nnapi_model, all_tensors, "new_height", DataType::kNumberTypeInt32, height_.ix_) !=
          RET_OK) {
        MS_LOG(ERROR) << "Add new height of resize to NNAPI model failed.";
        return RET_ERROR;
      }
      if (AddScalarToNNAPIModel<int>(nnapi_model, all_tensors, "new_width", DataType::kNumberTypeInt32, width_.ix_) !=
          RET_OK) {
        MS_LOG(ERROR) << "Add new width of resize to NNAPI model failed.";
        return RET_ERROR;
      }
      break;
    case DataType::kNumberTypeFloat32:
      if (AddScalarToNNAPIModel<float>(nnapi_model, all_tensors, "new_height", DataType::kNumberTypeFloat32,
                                       height_.fx_) != RET_OK) {
        MS_LOG(ERROR) << "Add new height of resize to NNAPI model failed.";
        return RET_ERROR;
      }
      if (AddScalarToNNAPIModel<float>(nnapi_model, all_tensors, "new_width", DataType::kNumberTypeFloat32,
                                       width_.fx_) != RET_OK) {
        MS_LOG(ERROR) << "Add new width of resize to NNAPI model failed.";
        return RET_ERROR;
      }
      break;
    default:
      MS_LOG(ERROR) << "The new size should be an int value or a float value.";
      return RET_ERROR;
  }
  if (AddScalarToNNAPIModel<bool>(nnapi_model, all_tensors, "nchw", DataType::kNumberTypeBool, false) != RET_OK) {
    MS_LOG(ERROR) << "Add is_nchw of resize to NNAPI model failed: " << op_name_;
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
