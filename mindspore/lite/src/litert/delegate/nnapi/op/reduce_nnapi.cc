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

#include "src/litert/delegate/nnapi/op/reduce_nnapi.h"
#include <vector>
#include <unordered_map>
#include "src/litert/delegate/nnapi/nnapi_utils.h"
#include "nnacl/op_base.h"

namespace mindspore {
namespace lite {
bool NNAPIReduce::IsSupport() {
  bool valid_mode = static_cast<schema::ReduceMode>(mode_) == schema::ReduceMode_ReduceMean ||
                    static_cast<schema::ReduceMode>(mode_) == schema::ReduceMode_ReduceMax ||
                    static_cast<schema::ReduceMode>(mode_) == schema::ReduceMode_ReduceMin ||
                    static_cast<schema::ReduceMode>(mode_) == schema::ReduceMode_ReduceProd ||
                    static_cast<schema::ReduceMode>(mode_) == schema::ReduceMode_ReduceSum;
  auto input = in_tensors_.front();
  return valid_mode && input.Shape().size() <= DIMENSION_4D;
}

int NNAPIReduce::InitParams() {
  auto reduce = op_primitive_->value_as_ReduceFusion();
  MS_CHECK_TRUE_RET(reduce != nullptr, RET_ERROR);
  mode_ = reduce->mode();
  keep_dims_ = reduce->keep_dims();
  return RET_OK;
}

int NNAPIReduce::AddOpToNNAPIModel(ANeuralNetworksModel *nnapi_model, std::vector<mindspore::MSTensor> *all_tensors) {
  MS_ASSERT(nnapi_model != nullptr && all_tensors != nullptr);
  OperationCode node_type;
  switch (static_cast<schema::ReduceMode>(mode_)) {
    case schema::ReduceMode_ReduceMean:
      node_type = ANEURALNETWORKS_MEAN;
      break;
    case schema::ReduceMode_ReduceMax:
      node_type = ANEURALNETWORKS_REDUCE_MAX;
      break;
    case schema::ReduceMode_ReduceMin:
      node_type = ANEURALNETWORKS_REDUCE_MIN;
      break;
    case schema::ReduceMode_ReduceProd:
      node_type = ANEURALNETWORKS_REDUCE_PROD;
      break;
    case schema::ReduceMode_ReduceSum:
      node_type = ANEURALNETWORKS_REDUCE_SUM;
      break;
    default:
      MS_LOG(ERROR) << "Unsupported reduce mode: " << mode_;
      return RET_ERROR;
  }
  if (InitNNAPIOpInOut(*all_tensors) != RET_OK) {
    MS_LOG(ERROR) << "InitNNAPINodeInfo failed.";
    return RET_ERROR;
  }
  if (AddScalarToNNAPIModel<int>(nnapi_model, all_tensors, "keep_dims", DataType::kNumberTypeInt32, keep_dims_) !=
      RET_OK) {
    MS_LOG(ERROR) << "Add keep_dims of reduce to NNAPI model failed.";
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
