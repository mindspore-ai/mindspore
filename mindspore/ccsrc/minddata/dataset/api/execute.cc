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
#include "acl/acl.h"
#include "minddata/dataset/kernels/image/dvpp/utils/ResourceManager.h"
#include "minddata/dataset/kernels/image/dvpp/utils/ErrorCode.h"
#include "minddata/dataset/kernels/image/dvpp/utils/MDAclProcess.h"
#include "minddata/dataset/kernels/image/dvpp/utils/CommonDataType.h"
#include "minddata/dataset/kernels/image/dvpp/utils/DvppCommon.h"
#endif

namespace mindspore {
namespace dataset {
#ifdef ENABLE_ACL
class AscendResource {
 public:
  AscendResource();
  ~AscendResource() = default;

  Status InitChipResource();

  Status FinalizeChipResource();

  Status Sink(const mindspore::MSTensor &host_input, std::shared_ptr<DeviceTensor> *device_input);

  Status Pop(std::shared_ptr<DeviceTensor> device_output, std::shared_ptr<Tensor> *host_output);

  Status DeviceDataRelease();

  std::shared_ptr<MDAclProcess> processor_;
  std::shared_ptr<ResourceManager> ascend_resource_;
};

AscendResource::AscendResource() { InitChipResource(); }

Status AscendResource::InitChipResource() {
  ResourceInfo resource;
  resource.aclConfigPath = "";
  resource.deviceIds.insert(mindspore::GlobalContext::GetGlobalDeviceID());
  ascend_resource_ = ResourceManager::GetInstance();
  APP_ERROR ret = ascend_resource_->InitResource(resource);
  if (ret != APP_ERR_OK) {
    ascend_resource_->Release();
    std::string err_msg = "Error in Init D-chip:" + std::to_string(ret);
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_UNEXPECTED(err_msg);
  }
  int device_id = *(resource.deviceIds.begin());
  aclrtContext context = ascend_resource_->GetContext(device_id);
  processor_ = std::make_shared<MDAclProcess>(context, false);
  ret = processor_->InitResource();
  if (ret != APP_ERR_OK) {
    ascend_resource_->Release();
    std::string err_msg = "Error in Init resource:" + std::to_string(ret);
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_UNEXPECTED(err_msg);
  }
  MS_LOG(INFO) << "Ascend resource all initialized!";
  return Status::OK();
}

Status AscendResource::FinalizeChipResource() {
  processor_->Release();
  return Status::OK();
}

Status AscendResource::Sink(const mindspore::MSTensor &host_input, std::shared_ptr<DeviceTensor> *device_input) {
  std::shared_ptr<mindspore::dataset::Tensor> de_input;
  Status rc = dataset::Tensor::CreateFromMemory(dataset::TensorShape(host_input.Shape()),
                                                MSTypeToDEType(static_cast<TypeId>(host_input.DataType())),
                                                (const uchar *)(host_input.Data().get()), &de_input);
  RETURN_IF_NOT_OK(rc);
  APP_ERROR ret = processor_->H2D_Sink(de_input, *device_input);
  if (ret != APP_ERR_OK) {
    ascend_resource_->Release();
    std::string err_msg = "Error in data sink process:" + std::to_string(ret);
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_UNEXPECTED(err_msg);
  }
  MS_LOG(INFO) << "Process data sink successfully";
  return Status::OK();
}

Status AscendResource::Pop(std::shared_ptr<DeviceTensor> device_output, std::shared_ptr<Tensor> *host_output) {
  APP_ERROR ret = processor_->D2H_Pop(device_output, *host_output);
  if (ret != APP_ERR_OK) {
    ascend_resource_->Release();
    std::string err_msg = "Error in data pop processing:" + std::to_string(ret);
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_UNEXPECTED(err_msg);
  }
  return Status::OK();
}

Status AscendResource::DeviceDataRelease() {
  APP_ERROR ret = processor_->device_memory_release();
  if (ret != APP_ERR_OK) {
    ascend_resource_->Release();
    std::string err_msg = "Error in device data release:" + std::to_string(ret);
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_UNEXPECTED(err_msg);
  }
  return Status::OK();
}
#endif

Execute::Execute(std::shared_ptr<TensorOperation> op, std::string deviceType) {
  ops_.emplace_back(std::move(op));
  device_type_ = deviceType;
  MS_LOG(INFO) << "Running Device: " << device_type_;
#ifdef ENABLE_ACL
  if (device_type_ == "Ascend310") {
    D_resource_ = std::make_shared<AscendResource>();
  }
#endif
}

Execute::Execute(std::vector<std::shared_ptr<TensorOperation>> ops, std::string deviceType)
    : ops_(std::move(ops)), device_type_(deviceType) {
  MS_LOG(INFO) << "Running Device: " << device_type_;
#ifdef ENABLE_ACL
  if (device_type_ == "Ascend310") {
    D_resource_ = std::make_shared<AscendResource>();
  }
#endif
}

Execute::~Execute() {
#ifdef ENABLE_ACL
  if (device_type_ == "Ascend310") {
    D_resource_->FinalizeChipResource();
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
    std::shared_ptr<mindspore::dataset::DeviceTensor> device_input;
    RETURN_IF_NOT_OK(D_resource_->Sink(input, &device_input));
    for (auto &t : transforms) {
      std::shared_ptr<DeviceTensor> device_output;
      RETURN_IF_NOT_OK(t->SetAscendResource(D_resource_->processor_));
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
    for (auto &input_tensor : input_tensor_list) {
      std::shared_ptr<dataset::DeviceTensor> device_input;
      RETURN_IF_NOT_OK(D_resource_->Sink(input_tensor, &device_input));
      for (auto &t : transforms) {
        std::shared_ptr<DeviceTensor> device_output;
        RETURN_IF_NOT_OK(t->SetAscendResource(D_resource_->processor_));
        RETURN_IF_NOT_OK(t->Compute(device_input, &device_output));

        // For next transform
        device_input = std::move(device_output);
      }
      CHECK_FAIL_RETURN_UNEXPECTED(device_input->HasDeviceData(), "Apply transform failed, output tensor has no data");
      // Due to the limitation of Ascend310 memory, we have to pop every data onto host memory
      // So the speed of this method is slower than solo mode
      std::shared_ptr<mindspore::dataset::Tensor> host_output;
      RETURN_IF_NOT_OK(D_resource_->Pop(device_input, &host_output));
      auto ms_tensor = mindspore::MSTensor(std::make_shared<DETensor>(host_output));
      output_tensor_list->emplace_back(ms_tensor);
      RETURN_IF_NOT_OK(D_resource_->DeviceDataRelease());
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

#ifdef ENABLE_ACL
Status Execute::DeviceMemoryRelease() {
  Status rc = D_resource_->DeviceDataRelease();
  if (rc.IsError()) {
    D_resource_->ascend_resource_->Release();
    std::string err_msg = "Error in device data release";
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_UNEXPECTED(err_msg);
  }
  return Status::OK();
}
#endif
}  // namespace dataset
}  // namespace mindspore
