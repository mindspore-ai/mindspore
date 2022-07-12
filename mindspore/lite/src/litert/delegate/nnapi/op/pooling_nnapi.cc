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

#include "src/litert/delegate/nnapi/op/pooling_nnapi.h"
#include <unordered_map>
#include "src/litert/delegate/nnapi/nnapi_utils.h"
#include "nnacl/op_base.h"

namespace mindspore {
namespace lite {
int NNAPIPooling::InitParams() {
  if (type_ == schema::PrimitiveType_AvgPoolFusion) {
    auto pool = op_primitive_->value_as_AvgPoolFusion();
    MS_ASSERT(pool != nullptr);
    act_type_ = pool->activation_type();
    MS_CHECK_TRUE_RET(pool->pad()->size() == DIMENSION_4D, RET_ERROR);
    pad_list_.push_back(static_cast<int>(*(pool->pad()->begin() + PAD_LEFT)));
    pad_list_.push_back(static_cast<int>(*(pool->pad()->begin() + PAD_RIGHT)));
    pad_list_.push_back(static_cast<int>(*(pool->pad()->begin() + PAD_UP)));
    pad_list_.push_back(static_cast<int>(*(pool->pad()->begin() + PAD_DOWN)));

    MS_CHECK_TRUE_RET(pool->strides()->size() == DIMENSION_2D, RET_ERROR);
    strides_.push_back(static_cast<int>(*(pool->strides()->begin() + 1)));
    strides_.push_back(static_cast<int>(*(pool->strides()->begin())));
    if (pool->global()) {
      MS_CHECK_TRUE_RET(in_tensors_.at(0).Shape().size() == DIMENSION_4D, RET_ERROR);
      kernel_size_.at(0) = in_tensors_.at(0).Shape().at(2);
      kernel_size_.at(1) = in_tensors_.at(0).Shape().at(1);
    } else if (pool->kernel_size() != nullptr && pool->kernel_size()->size() == DIMENSION_2D) {
      kernel_size_.at(0) = static_cast<int>(*(pool->kernel_size()->begin()));
      kernel_size_.at(1) = static_cast<int>(*(pool->kernel_size()->begin() + 1));
    }
  } else {
    auto pool = op_primitive_->value_as_MaxPoolFusion();
    MS_ASSERT(pool != nullptr);
    act_type_ = pool->activation_type();
    MS_CHECK_TRUE_RET(pool->pad()->size() == DIMENSION_4D, RET_ERROR);
    pad_list_.push_back(static_cast<int>(*(pool->pad()->begin() + PAD_LEFT)));
    pad_list_.push_back(static_cast<int>(*(pool->pad()->begin() + PAD_RIGHT)));
    pad_list_.push_back(static_cast<int>(*(pool->pad()->begin() + PAD_UP)));
    pad_list_.push_back(static_cast<int>(*(pool->pad()->begin() + PAD_DOWN)));

    MS_CHECK_TRUE_RET(pool->strides()->size() == DIMENSION_2D, RET_ERROR);
    strides_.push_back(static_cast<int>(*(pool->strides()->begin() + 1)));
    strides_.push_back(static_cast<int>(*(pool->strides()->begin())));

    if (pool->kernel_size() != nullptr && pool->kernel_size()->size() == DIMENSION_2D) {
      kernel_size_.at(0) = static_cast<int>(*(pool->kernel_size()->begin()));
      kernel_size_.at(1) = static_cast<int>(*(pool->kernel_size()->begin() + 1));
    }
  }

  return RET_OK;
}

int NNAPIPooling::AddOpToNNAPIModel(ANeuralNetworksModel *nnapi_model, std::vector<mindspore::MSTensor> *all_tensors) {
  MS_ASSERT(nnapi_model != nullptr && all_tensors != nullptr);
  OperationCode node_type =
    type_ == schema::PrimitiveType_AvgPoolFusion ? ANEURALNETWORKS_AVERAGE_POOL_2D : ANEURALNETWORKS_MAX_POOL_2D;

  if (InitNNAPIOpInOut(*all_tensors) != RET_OK) {
    MS_LOG(ERROR) << "InitNNAPINodeInfo failed.";
    return RET_ERROR;
  }
  for (auto pad : pad_list_) {
    if (AddScalarToNNAPIModel<int>(nnapi_model, all_tensors, "pad", DataType::kNumberTypeInt32, pad) != RET_OK) {
      MS_LOG(ERROR) << "Add paddings for conv to NNAPI model failed.";
      return RET_ERROR;
    }
  }
  for (auto stride : strides_) {
    if (AddScalarToNNAPIModel<int>(nnapi_model, all_tensors, "stride", DataType::kNumberTypeInt32, stride) != RET_OK) {
      MS_LOG(ERROR) << "Add pad mode for conv to NNAPI model failed.";
      return RET_ERROR;
    }
  }
  for (auto kernel_size : kernel_size_) {
    if (AddScalarToNNAPIModel<int>(nnapi_model, all_tensors, "kernel_size", DataType::kNumberTypeInt32, kernel_size) !=
        RET_OK) {
      MS_LOG(ERROR) << "Add pad mode for conv to NNAPI model failed.";
      return RET_ERROR;
    }
  }
  // convert act_type to an input of nnapi node.
  if (AddScalarToNNAPIModel<int>(nnapi_model, all_tensors, "act_type", DataType::kNumberTypeInt32, act_type_) !=
      RET_OK) {
    MS_LOG(ERROR) << "Add activation type for add to NNAPI model failed.";
    return RET_ERROR;
  }
  // set nchw to an input of nnapi node.
  if (AddScalarToNNAPIModel<bool>(nnapi_model, all_tensors, "nchw", DataType::kNumberTypeBool, false) != RET_OK) {
    MS_LOG(ERROR) << "set nchw format for add to NNAPI model failed.";
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
