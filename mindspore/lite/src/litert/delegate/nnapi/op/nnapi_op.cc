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

#include "src/litert/delegate/nnapi/op/nnapi_op.h"
#include <unordered_map>
#include "src/litert/delegate/nnapi/nnapi_utils.h"
#include "src/tensor.h"
#include "nnacl/op_base.h"

namespace mindspore {
namespace lite {
namespace {
std::unordered_map<schema::PrimitiveType, OperationCode> common_op_type = {
  {schema::PrimitiveType_Rsqrt, ANEURALNETWORKS_RSQRT},
  {schema::PrimitiveType_Equal, ANEURALNETWORKS_EQUAL},
  {schema::PrimitiveType_ExpFusion, ANEURALNETWORKS_EXP},
  {schema::PrimitiveType_Floor, ANEURALNETWORKS_FLOOR}};
}  // namespace

int NNAPIOp::ConvertInOutQuantSymmToASymm() {
  for (auto input : in_tensors_) {
    ConverTensorQuantSymmToASymm(&input);
  }

  for (auto output : out_tensors_) {
    ConverTensorQuantSymmToASymm(&output);
  }
  return RET_OK;
}

int NNAPIOp::InitNNAPIOpInOut(const std::vector<mindspore::MSTensor> &all_tensors) {
  input_indices_.clear();
  output_indices_.clear();
  for (auto input : in_tensors_) {
    auto itr = std::find(all_tensors.begin(), all_tensors.end(), input);
    MS_CHECK_TRUE_RET(itr != all_tensors.end(), RET_ERROR);
    input_indices_.push_back(itr - all_tensors.begin());
  }
  for (auto output : out_tensors_) {
    auto itr = std::find(all_tensors.begin(), all_tensors.end(), output);
    MS_CHECK_TRUE_RET(itr != all_tensors.end(), RET_ERROR);
    output_indices_.push_back(itr - all_tensors.begin());
  }
  return RET_OK;
}

int NNAPIOp::AddTensorToNNAPIModel(ANeuralNetworksModel *nnapi_model, std::vector<mindspore::MSTensor> *all_tensors,
                                   MSTensorInfo tensor_info) {
  auto tensor = MSTensor::CreateTensor(tensor_info.name_, tensor_info.type_, tensor_info.shape_, tensor_info.data_,
                                       tensor_info.data_len_);
  if (tensor == nullptr) {
    MS_LOG(ERROR) << "Create bias tensor failed.";
    return RET_ERROR;
  }
  if (AddNNAPIOperand(nnapi_model, *tensor, static_cast<int>(all_tensors->size())) != RET_OK) {
    MS_LOG(ERROR) << "Add NNAPI operand failed.";
    delete tensor;
    return RET_ERROR;
  }
  input_indices_.push_back(all_tensors->size());
  all_tensors->push_back(*tensor);
  op_attribute_tensors_.push_back(tensor);
  return RET_OK;
}

int NNAPICommon::AddOpToNNAPIModel(ANeuralNetworksModel *nnapi_model, std::vector<mindspore::MSTensor> *all_tensors) {
  MS_ASSERT(nnapi_model != nullptr && all_tensors != nullptr);
  MS_CHECK_TRUE_RET(common_op_type.find(type_) != common_op_type.end(), RET_ERROR);
  OperationCode node_type = common_op_type.at(type_);
  if (InitNNAPIOpInOut(*all_tensors) != RET_OK) {
    MS_LOG(ERROR) << "InitNNAPINodeInfo failed.";
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
