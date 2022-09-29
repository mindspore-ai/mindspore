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

#include "src/litert/delegate/nnapi/op/arithmetic_nnapi.h"
#include <vector>
#include <unordered_map>
#include "src/litert/delegate/nnapi/nnapi_utils.h"
#include "nnacl/op_base.h"

namespace mindspore {
namespace lite {
namespace {
std::unordered_map<schema::PrimitiveType, OperationCode> arithmetic_type = {
  {schema::PrimitiveType_AddFusion, ANEURALNETWORKS_ADD},
  {schema::PrimitiveType_SubFusion, ANEURALNETWORKS_SUB},
  {schema::PrimitiveType_MulFusion, ANEURALNETWORKS_MUL},
  {schema::PrimitiveType_DivFusion, ANEURALNETWORKS_DIV}};
}  // namespace

bool NNAPIArithmetic::IsSupport() {
  auto input = in_tensors_.front();
  return input.Shape().size() <= DIMENSION_4D;
}

int NNAPIArithmetic::InitParams() {
  switch (type_) {
    case schema::PrimitiveType_AddFusion:
      act_type_ = op_primitive_->value_as_AddFusion()->activation_type();
      break;
    case schema::PrimitiveType_SubFusion:
      act_type_ = op_primitive_->value_as_SubFusion()->activation_type();
      break;
    case schema::PrimitiveType_MulFusion:
      act_type_ = op_primitive_->value_as_MulFusion()->activation_type();
      break;
    case schema::PrimitiveType_DivFusion:
      act_type_ = op_primitive_->value_as_DivFusion()->activation_type();
      break;
    default:
      act_type_ = schema::ActivationType_NO_ACTIVATION;
  }
  return RET_OK;
}

int NNAPIArithmetic::AddOpToNNAPIModel(ANeuralNetworksModel *nnapi_model,
                                       std::vector<mindspore::MSTensor> *all_tensors) {
  MS_ASSERT(nnapi_model != nullptr && all_tensors != nullptr);
  if (arithmetic_type.find(type_) == arithmetic_type.end()) {
    MS_LOG(ERROR) << "Unsupported arithmetic type: " << type_;
  }
  OperationCode node_type = arithmetic_type.at(type_);
  if (InitNNAPIOpInOut(*all_tensors) != RET_OK) {
    MS_LOG(ERROR) << "InitNNAPINodeInfo failed.";
    return RET_ERROR;
  }
  // convert act_type to an input of nnapi node.
  if (AddScalarToNNAPIModel<int>(nnapi_model, all_tensors, "act_type", DataType::kNumberTypeInt32, act_type_) !=
      RET_OK) {
    MS_LOG(ERROR) << "Add activation type for add to NNAPI model failed.";
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
