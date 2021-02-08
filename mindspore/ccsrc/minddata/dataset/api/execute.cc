/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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

#include "minddata/dataset/include/execute.h"
#include "minddata/dataset/core/de_tensor.h"
#include "minddata/dataset/core/tensor_row.h"
#include "minddata/dataset/include/tensor.h"
#include "minddata/dataset/include/type_id.h"
#include "minddata/dataset/kernels/tensor_op.h"
#ifndef ENABLE_ANDROID
#include "utils/log_adapter.h"
#else
#include "mindspore/lite/src/common/log_adapter.h"
#endif

namespace mindspore {
namespace dataset {

Execute::Execute(std::shared_ptr<TensorOperation> op) { ops_.emplace_back(std::move(op)); }

Execute::Execute(std::vector<std::shared_ptr<TensorOperation>> ops) : ops_(std::move(ops)) {}

Status Execute::operator()(const mindspore::MSTensor &input, mindspore::MSTensor *output) {
  // Validate input tensor
  CHECK_FAIL_RETURN_UNEXPECTED(input.DataSize() > 0, "Input Tensor has no data");
  CHECK_FAIL_RETURN_UNEXPECTED(!ops_.empty(), "Input TensorOperation should be provided");

  // Validate and build runtime ops
  std::vector<std::shared_ptr<TensorOp>> transforms;
  for (int32_t i = 0; i < ops_.size(); i++) {
    CHECK_FAIL_RETURN_UNEXPECTED(ops_[i] != nullptr, "Input TensorOperation[" + std::to_string(i) + "] is null");
    RETURN_IF_NOT_OK(ops_[i]->ValidateParams());
    transforms.emplace_back(ops_[i]->Build());
  }

  // Convert mindspore::Tensor to dataset::Tensor
  std::shared_ptr<dataset::Tensor> de_tensor;
  Status rc = dataset::Tensor::CreateFromMemory(dataset::TensorShape(input.Shape()),
                                                MSTypeToDEType(static_cast<TypeId>(input.DataType())),
                                                (const uchar *)(input.Data().get()), input.DataSize(), &de_tensor);
  if (rc.IsError()) {
    MS_LOG(ERROR) << rc;
    RETURN_IF_NOT_OK(rc);
  }

  // Apply transforms on tensor
  for (auto &t : transforms) {
    std::shared_ptr<dataset::Tensor> de_output;
    Status rc_ = t->Compute(de_tensor, &de_output);
    if (rc_.IsError()) {
      MS_LOG(ERROR) << rc_;
      RETURN_IF_NOT_OK(rc_);
    }

    // For next transform
    de_tensor = std::move(de_output);
  }

  // Convert dataset::Tensor to mindspore::Tensor
  CHECK_FAIL_RETURN_UNEXPECTED(de_tensor->HasData(), "Apply transform failed, output tensor has no data");
  *output = mindspore::MSTensor(std::make_shared<DETensor>(de_tensor));
  return Status::OK();
}

Status Execute::operator()(const std::vector<MSTensor> &input_tensor_list, std::vector<MSTensor> *output_tensor_list) {
  // Validate input tensor
  CHECK_FAIL_RETURN_UNEXPECTED(!input_tensor_list.empty(), "Input Tensor is not valid");
  for (auto &tensor : input_tensor_list) {
    CHECK_FAIL_RETURN_UNEXPECTED(tensor.DataSize() > 0, "Input Tensor has no data");
  }
  CHECK_FAIL_RETURN_UNEXPECTED(!ops_.empty(), "Input TensorOperation should be provided");

  // Validate and build runtime ops
  std::vector<std::shared_ptr<TensorOp>> transforms;
  for (int32_t i = 0; i < ops_.size(); i++) {
    CHECK_FAIL_RETURN_UNEXPECTED(ops_[i] != nullptr, "Input TensorOperation[" + std::to_string(i) + "] is null");
    RETURN_IF_NOT_OK(ops_[i]->ValidateParams());
    transforms.emplace_back(ops_[i]->Build());
  }

  TensorRow de_tensor_list;
  for (auto &tensor : input_tensor_list) {
    std::shared_ptr<dataset::Tensor> de_tensor;
    Status rc = dataset::Tensor::CreateFromMemory(dataset::TensorShape(tensor.Shape()),
                                                  MSTypeToDEType(static_cast<TypeId>(tensor.DataType())),
                                                  (const uchar *)(tensor.Data().get()), tensor.DataSize(), &de_tensor);
    if (rc.IsError()) {
      MS_LOG(ERROR) << rc;
      RETURN_IF_NOT_OK(rc);
    }
    de_tensor_list.emplace_back(std::move(de_tensor));
  }

  // Apply transforms on tensor
  for (auto &t : transforms) {
    TensorRow de_output_list;
    Status rc = t->Compute(de_tensor_list, &de_output_list);
    if (rc.IsError()) {
      MS_LOG(ERROR) << rc;
      RETURN_IF_NOT_OK(rc);
    }
    // For next transform
    de_tensor_list = std::move(de_output_list);
  }

  for (auto &tensor : de_tensor_list) {
    CHECK_FAIL_RETURN_UNEXPECTED(tensor->HasData(), "Apply transform failed, output tensor has no data");
    auto ms_tensor = mindspore::MSTensor(std::make_shared<DETensor>(tensor));
    output_tensor_list->emplace_back(ms_tensor);
  }
  CHECK_FAIL_RETURN_UNEXPECTED(!output_tensor_list->empty(), "Output Tensor is not valid");
  return Status::OK();
}

}  // namespace dataset
}  // namespace mindspore
