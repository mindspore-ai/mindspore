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

#include "src/litert/delegate/nnapi/op/stack_nnapi.h"
#include <algorithm>
#include <vector>
#include <unordered_map>
#include "src/litert/delegate/nnapi/nnapi_utils.h"
#include "nnacl/op_base.h"

namespace mindspore {
namespace lite {
bool NNAPIStack::IsSupport() {
  auto output_shape = out_tensors_.front().Shape();
  return output_shape.size() <= DIMENSION_4D &&
         std::find(output_shape.begin(), output_shape.end(), -1) == output_shape.end();
}

int NNAPIStack::InitParams() {
  auto stack = op_primitive_->value_as_Stack();
  MS_CHECK_TRUE_RET(stack != nullptr, RET_ERROR);
  axis_ = stack->axis();
  return RET_OK;
}

int NNAPIStack::AddOpToNNAPIModel(ANeuralNetworksModel *nnapi_model, std::vector<mindspore::MSTensor> *all_tensors) {
  MS_ASSERT(nnapi_model != nullptr && all_tensors != nullptr);
  // stack ==> reshape + concat.
  auto outputs = out_tensors_;
  auto output_shape = out_tensors_.front().Shape();
  axis_ = axis_ < 0 ? axis_ + static_cast<int>(output_shape.size()) : axis_;
  MS_CHECK_TRUE_RET(axis_ >= 0 && axis_ < static_cast<int>(output_shape.size()), RET_ERROR);
  std::vector<MSTensor> concat_inputs;
  for (auto input : in_tensors_) {
    OperationCode node_type = ANEURALNETWORKS_RESHAPE;
    auto in_shape = input.Shape();
    std::vector<int64_t> reshape_out_shape(in_shape);
    reshape_out_shape.insert(reshape_out_shape.begin() + axis_, 1);
    auto reshape_output =
      MSTensor::CreateTensor(input.Name() + "_reshape", input.DataType(), reshape_out_shape, nullptr, 0);
    MS_CHECK_TRUE_RET(reshape_output != nullptr, RET_ERROR);
    if (AddNNAPIOperand(nnapi_model, *reshape_output, static_cast<int>(all_tensors->size())) != RET_OK) {
      MS_LOG(ERROR) << "Add temporary output for stack failed.";
      delete reshape_output;
      return RET_ERROR;
    }
    in_tensors_ = {input};
    out_tensors_ = {*reshape_output};
    all_tensors->push_back(*reshape_output);
    op_attribute_tensors_.push_back(reshape_output);
    concat_inputs.push_back(*reshape_output);

    if (InitNNAPIOpInOut(*all_tensors) != RET_OK) {
      MS_LOG(ERROR) << "InitNNAPINodeInfo failed.";
      return RET_ERROR;
    }
    std::vector<int> shape;
    (void)std::transform(reshape_out_shape.begin(), reshape_out_shape.end(), std::back_inserter(shape),
                         [](int64_t x) { return static_cast<int>(x); });
    MSTensorInfo tensor_info{"reshape_shape",
                             DataType::kNumberTypeInt32,
                             {static_cast<int64_t>(shape.size())},
                             shape.data(),
                             shape.size() * sizeof(int)};
    if (AddTensorToNNAPIModel(nnapi_model, all_tensors, tensor_info) != RET_OK) {
      MS_LOG(ERROR) << "Add shape tensor to reshape failed.";
      return RET_ERROR;
    }
    if (nnapi_->ANeuralNetworksModel_addOperation(nnapi_model, node_type, input_indices_.size(), input_indices_.data(),
                                                  output_indices_.size(), output_indices_.data()) != RET_OK) {
      MS_LOG(ERROR) << "Add operation to NNAPI model failed: " << op_name_;
      return RET_ERROR;
    }
  }
  OperationCode node_type = ANEURALNETWORKS_CONCATENATION;
  // input shape size is equal to output size minus 1.
  MS_CHECK_TRUE_RET(axis_ != 0, RET_ERROR);
  in_tensors_ = concat_inputs;
  out_tensors_ = outputs;
  if (InitNNAPIOpInOut(*all_tensors) != RET_OK) {
    MS_LOG(ERROR) << "InitNNAPINodeInfo failed.";
    return RET_ERROR;
  }
  if (AddScalarToNNAPIModel<int>(nnapi_model, all_tensors, "axis", DataType::kNumberTypeInt32, axis_) != RET_OK) {
    MS_LOG(ERROR) << "Add axis of stack to NNAPI model failed.";
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
