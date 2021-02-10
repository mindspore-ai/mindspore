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
#include "minddata/dataset/core/device_tensor.h"
#include "minddata/dataset/core/tensor_row.h"
#include "minddata/dataset/include/tensor.h"
#include "minddata/dataset/include/type_id.h"
#include "minddata/dataset/kernels/tensor_op.h"
#ifndef ENABLE_ANDROID
#include "utils/log_adapter.h"
#else
#include "mindspore/lite/src/common/log_adapter.h"
#endif
#ifdef ENABLE_ACL
#include "minddata/dataset/core/ascend_resource.h"
#endif

namespace mindspore {
namespace dataset {

// FIXME - Temporarily overload Execute to support both TensorOperation and TensorTransform
Execute::Execute(std::shared_ptr<TensorOperation> op, std::string deviceType) {
  ops_.emplace_back(std::move(op));
  device_type_ = deviceType;
  MS_LOG(INFO) << "Running Device: " << device_type_;
#ifdef ENABLE_ACL
  if (device_type_ == "Ascend310") {
    device_resource_ = std::make_shared<AscendResource>();
    Status rc = device_resource_->InitResource();
    if (!rc.IsOk()) {
      device_resource_ = nullptr;
      MS_LOG(ERROR) << "Initialize Ascend310 resource fail";
    }
  }
#endif
}

Execute::Execute(std::shared_ptr<TensorTransform> op, std::string deviceType) {
  // Convert op from TensorTransform to TensorOperation
  std::shared_ptr<TensorOperation> operation = op->Parse();
  ops_.emplace_back(std::move(operation));
  device_type_ = deviceType;
  MS_LOG(INFO) << "Running Device: " << device_type_;
#ifdef ENABLE_ACL
  if (device_type_ == "Ascend310") {
    device_resource_ = std::make_shared<AscendResource>();
    Status rc = device_resource_->InitResource();
    if (!rc.IsOk()) {
      device_resource_ = nullptr;
      MS_LOG(ERROR) << "Initialize Ascend310 resource fail";
    }
  }
#endif
}

/*
Execute::Execute(TensorTransform op, std::string deviceType) {
  // Convert op from TensorTransform to TensorOperation
  std::shared_ptr<TensorOperation> operation = op.Parse();
  ops_.emplace_back(std::move(operation));
  device_type_ = deviceType;
  MS_LOG(INFO) << "Running Device: " << device_type_;
#ifdef ENABLE_ACL
  if (device_type_ == "Ascend310") {
    device_resource_ = std::make_shared<AscendResource>();
    Status rc = device_resource_->InitResource();
    if (!rc.IsOk()) {
      device_resource_ = nullptr;
      MS_LOG(ERROR) << "Initialize Ascend310 resource fail";
    }
  }
#endif
}
*/

// Execute function for the example case: auto decode(new vision::Decode());
Execute::Execute(TensorTransform *op, std::string deviceType) {
  // Convert op from TensorTransform to TensorOperation
  std::shared_ptr<TensorOperation> operation = op->Parse();
  ops_.emplace_back(std::move(operation));
  device_type_ = deviceType;
  MS_LOG(INFO) << "Running Device: " << device_type_;
#ifdef ENABLE_ACL
  if (device_type_ == "Ascend310") {
    device_resource_ = std::make_shared<AscendResource>();
    Status rc = device_resource_->InitResource();
    if (!rc.IsOk()) {
      device_resource_ = nullptr;
      MS_LOG(ERROR) << "Initialize Ascend310 resource fail";
    }
  }
#endif
}

Execute::Execute(std::vector<std::shared_ptr<TensorOperation>> ops, std::string deviceType)
    : ops_(std::move(ops)), device_type_(deviceType) {
  MS_LOG(INFO) << "Running Device: " << device_type_;
#ifdef ENABLE_ACL
  if (device_type_ == "Ascend310") {
    device_resource_ = std::make_shared<AscendResource>();
    Status rc = device_resource_->InitResource();
    if (!rc.IsOk()) {
      device_resource_ = nullptr;
      MS_LOG(ERROR) << "Initialize Ascend310 resource fail";
    }
  }
#endif
}

Execute::Execute(std::vector<std::shared_ptr<TensorTransform>> ops, std::string deviceType) {
  // Convert ops from TensorTransform to TensorOperation
  (void)std::transform(
    ops.begin(), ops.end(), std::back_inserter(ops_),
    [](std::shared_ptr<TensorTransform> operation) -> std::shared_ptr<TensorOperation> { return operation->Parse(); });
  device_type_ = deviceType;
  MS_LOG(INFO) << "Running Device: " << device_type_;
#ifdef ENABLE_ACL
  if (device_type_ == "Ascend310") {
    device_resource_ = std::make_shared<AscendResource>();
    Status rc = device_resource_->InitResource();
    if (!rc.IsOk()) {
      device_resource_ = nullptr;
      MS_LOG(ERROR) << "Initialize Ascend310 resource fail";
    }
  }
#endif
}

Execute::Execute(const std::vector<std::reference_wrapper<TensorTransform>> ops, std::string deviceType) {
  // Convert ops from TensorTransform to TensorOperation
  (void)std::transform(
    ops.begin(), ops.end(), std::back_inserter(ops_),
    [](TensorTransform &operation) -> std::shared_ptr<TensorOperation> { return operation.Parse(); });
  device_type_ = deviceType;
  MS_LOG(INFO) << "Running Device: " << device_type_;
#ifdef ENABLE_ACL
  if (device_type_ == "Ascend310") {
    device_resource_ = std::make_shared<AscendResource>();
    Status rc = device_resource_->InitResource();
    if (!rc.IsOk()) {
      device_resource_ = nullptr;
      MS_LOG(ERROR) << "Initialize Ascend310 resource fail";
    }
  }
#endif
}

// Execute function for the example vector case: auto decode(new vision::Decode());
Execute::Execute(std::vector<TensorTransform *> ops, std::string deviceType) {
  // Convert ops from TensorTransform to TensorOperation
  (void)std::transform(
    ops.begin(), ops.end(), std::back_inserter(ops_),
    [](TensorTransform *operation) -> std::shared_ptr<TensorOperation> { return operation->Parse(); });
  device_type_ = deviceType;
  MS_LOG(INFO) << "Running Device: " << device_type_;
#ifdef ENABLE_ACL
  if (device_type_ == "Ascend310") {
    device_resource_ = std::make_shared<AscendResource>();
    Status rc = device_resource_->InitResource();
    if (!rc.IsOk()) {
      device_resource_ = nullptr;
      MS_LOG(ERROR) << "Initialize Ascend310 resource fail";
    }
  }
#endif
}

Execute::~Execute() {
#ifdef ENABLE_ACL
  if (device_type_ == "Ascend310") {
    if (device_resource_) {
      device_resource_->FinalizeResource();
    } else {
      MS_LOG(ERROR) << "Device resource is nullptr which is illegal under case Ascend310";
    }
  }
#endif
}

Status Execute::operator()(const mindspore::MSTensor &input, mindspore::MSTensor *output) {
  // Validate input tensor
  CHECK_FAIL_RETURN_UNEXPECTED(input.DataSize() > 0, "Input Tensor has no data");
  CHECK_FAIL_RETURN_UNEXPECTED(!ops_.empty(), "Input TensorOperation should be provided");
  CHECK_FAIL_RETURN_UNEXPECTED(validate_device_(), "Device Type should be 'Ascend310' or 'CPU'");
  // Validate and build runtime ops
  std::vector<std::shared_ptr<TensorOp>> transforms;  // record the transformations
  for (int32_t i = 0; i < ops_.size(); i++) {
    CHECK_FAIL_RETURN_UNEXPECTED(ops_[i] != nullptr, "Input TensorOperation[" + std::to_string(i) + "] is null");
    RETURN_IF_NOT_OK(ops_[i]->ValidateParams());
    transforms.emplace_back(ops_[i]->Build());
  }
  if (device_type_ == "CPU") {
    // Convert mindspore::Tensor to dataset::Tensor
    std::shared_ptr<dataset::Tensor> de_tensor;
    Status rc = dataset::Tensor::CreateFromMemory(dataset::TensorShape(input.Shape()),
                                                  MSTypeToDEType(static_cast<TypeId>(input.DataType())),
                                                  (const uchar *)(input.Data().get()), input.DataSize(), &de_tensor);
    if (rc.IsError()) {
      MS_LOG(ERROR) << rc;
      return rc;
    }

    // Apply transforms on tensor
    for (auto &t : transforms) {
      std::shared_ptr<dataset::Tensor> de_output;
      Status rc_ = t->Compute(de_tensor, &de_output);
      if (rc_.IsError()) {
        MS_LOG(ERROR) << rc_;
        return rc_;
      }

      // For next transform
      de_tensor = std::move(de_output);
    }

    // Convert dataset::Tensor to mindspore::Tensor
    CHECK_FAIL_RETURN_UNEXPECTED(de_tensor->HasData(), "Apply transform failed, output tensor has no data");
    *output = mindspore::MSTensor(std::make_shared<DETensor>(de_tensor));
  } else {  // Ascend310 case, where we must set Ascend resource on each operators
#ifdef ENABLE_ACL
    CHECK_FAIL_RETURN_UNEXPECTED(device_resource_, "Device resource is nullptr which is illegal under case Ascend310");
    std::shared_ptr<mindspore::dataset::DeviceTensor> device_input;
    RETURN_IF_NOT_OK(device_resource_->Sink(input, &device_input));
    for (auto &t : transforms) {
      std::shared_ptr<DeviceTensor> device_output;
      RETURN_IF_NOT_OK(t->SetAscendResource(device_resource_));
      RETURN_IF_NOT_OK(t->Compute(device_input, &device_output));

      // For next transform
      device_input = std::move(device_output);
    }
    CHECK_FAIL_RETURN_UNEXPECTED(device_input->HasDeviceData(), "Apply transform failed, output tensor has no data");
    *output = mindspore::MSTensor(std::make_shared<DETensor>(device_input, true));
#endif
  }
  return Status::OK();
}

Status Execute::operator()(const std::vector<MSTensor> &input_tensor_list, std::vector<MSTensor> *output_tensor_list) {
  // Validate input tensor
  CHECK_FAIL_RETURN_UNEXPECTED(!input_tensor_list.empty(), "Input Tensor is not valid");
  for (auto &tensor : input_tensor_list) {
    CHECK_FAIL_RETURN_UNEXPECTED(tensor.DataSize() > 0, "Input Tensor has no data");
  }
  CHECK_FAIL_RETURN_UNEXPECTED(!ops_.empty(), "Input TensorOperation should be provided");
  CHECK_FAIL_RETURN_UNEXPECTED(validate_device_(), "Device Type should be 'Ascend310' or 'CPU'");

  // Validate and build runtime ops
  std::vector<std::shared_ptr<TensorOp>> transforms;
  for (int32_t i = 0; i < ops_.size(); i++) {
    CHECK_FAIL_RETURN_UNEXPECTED(ops_[i] != nullptr, "Input TensorOperation[" + std::to_string(i) + "] is null");
    RETURN_IF_NOT_OK(ops_[i]->ValidateParams());
    transforms.emplace_back(ops_[i]->Build());
  }
  if (device_type_ == "CPU") {  // Case CPU
    TensorRow de_tensor_list;
    for (auto &tensor : input_tensor_list) {
      std::shared_ptr<dataset::Tensor> de_tensor;
      Status rc = dataset::Tensor::CreateFromMemory(
        dataset::TensorShape(tensor.Shape()), MSTypeToDEType(static_cast<TypeId>(tensor.DataType())),
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
      RETURN_IF_NOT_OK(t->Compute(de_tensor_list, &de_output_list));
      // For next transform
      de_tensor_list = std::move(de_output_list);
    }

    for (auto &tensor : de_tensor_list) {
      CHECK_FAIL_RETURN_UNEXPECTED(tensor->HasData(), "Apply transform failed, output tensor has no data");
      auto ms_tensor = mindspore::MSTensor(std::make_shared<DETensor>(tensor));
      output_tensor_list->emplace_back(ms_tensor);
    }
    CHECK_FAIL_RETURN_UNEXPECTED(!output_tensor_list->empty(), "Output Tensor is not valid");
  } else {  // Case Ascend310
#ifdef ENABLE_ACL
    CHECK_FAIL_RETURN_UNEXPECTED(device_resource_, "Device resource is nullptr which is illegal under case Ascend310");
    for (auto &input_tensor : input_tensor_list) {
      std::shared_ptr<dataset::DeviceTensor> device_input;
      RETURN_IF_NOT_OK(device_resource_->Sink(input_tensor, &device_input));
      for (auto &t : transforms) {
        std::shared_ptr<DeviceTensor> device_output;
        RETURN_IF_NOT_OK(t->SetAscendResource(device_resource_));
        RETURN_IF_NOT_OK(t->Compute(device_input, &device_output));

        // For next transform
        device_input = std::move(device_output);
      }
      CHECK_FAIL_RETURN_UNEXPECTED(device_input->HasDeviceData(), "Apply transform failed, output tensor has no data");
      // Due to the limitation of Ascend310 memory, we have to pop every data onto host memory
      // So the speed of this batch method is slower than solo mode
      std::shared_ptr<mindspore::dataset::Tensor> host_output;
      RETURN_IF_NOT_OK(device_resource_->Pop(device_input, &host_output));
      auto ms_tensor = mindspore::MSTensor(std::make_shared<DETensor>(host_output));
      output_tensor_list->emplace_back(ms_tensor);
      RETURN_IF_NOT_OK(device_resource_->DeviceDataRelease());
    }
    CHECK_FAIL_RETURN_UNEXPECTED(!output_tensor_list->empty(), "Output Tensor vector is empty");
#endif
  }
  return Status::OK();
}

Status Execute::validate_device_() {
  if (device_type_ != "CPU" && device_type_ != "Ascend310") {
    std::string err_msg = device_type_ + " is not supported. (Option: CPU or Ascend310)";
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_UNEXPECTED(err_msg);
  }
  return Status::OK();
}

Status Execute::DeviceMemoryRelease() {
  CHECK_FAIL_RETURN_UNEXPECTED(device_resource_, "Device resource is nullptr which is illegal under case Ascend310");
  Status rc = device_resource_->DeviceDataRelease();
  if (rc.IsError()) {
    std::string err_msg = "Error in device data release";
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_UNEXPECTED(err_msg);
  }
  return Status::OK();
}

}  // namespace dataset
}  // namespace mindspore
