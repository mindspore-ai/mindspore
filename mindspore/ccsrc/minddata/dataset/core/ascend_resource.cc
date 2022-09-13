/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "include/api/types.h"
#include "minddata/dataset/core/ascend_resource.h"
#include "minddata/dataset/core/type_id.h"
#include "minddata/dataset/kernels/image/dvpp/acl_adapter.h"
#include "minddata/dataset/kernels/image/dvpp/utils/CommonDataType.h"
#include "minddata/dataset/kernels/image/dvpp/utils/ErrorCode.h"
#include "minddata/dataset/kernels/image/image_utils.h"

namespace mindspore {
namespace dataset {
Status AscendResource::InitResource(uint32_t device_id) {
  ResourceInfo resource;
  resource.deviceIds.insert(device_id);
  APP_ERROR ret = AclAdapter::GetInstance().InitResource(&resource);
  if (ret != APP_ERR_OK) {
    AclAdapter::GetInstance().Release();
    std::string err_msg = "Error in Init D-chip:" + std::to_string(ret);
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_UNEXPECTED(err_msg);
  }
  int cur_device_id = *(resource.deviceIds.begin());
  void *context = AclAdapter::GetInstance().GetContext(cur_device_id);
  processor_ = std::shared_ptr<void>(AclAdapter::GetInstance().CreateAclProcess(context, false, nullptr, nullptr),
                                     [](void *ptr) { AclAdapter::GetInstance().DestroyAclProcess(ptr); });
  ret = AclAdapter::GetInstance().InitAclProcess(processor_.get());
  if (ret != APP_ERR_OK) {
    AclAdapter::GetInstance().Release();
    std::string err_msg = "Error in Init resource:" + std::to_string(ret);
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_UNEXPECTED(err_msg);
  }
  MS_LOG(INFO) << "Ascend resource all initialized!";
  return Status::OK();
}

Status AscendResource::FinalizeResource() {
  AclAdapter::GetInstance().ReleaseAclProcess(processor_.get());
  return Status::OK();
}

Status AscendResource::Sink(const mindspore::MSTensor &host_input, std::shared_ptr<DeviceTensor> *device_input) {
  std::shared_ptr<mindspore::dataset::Tensor> de_input;
  Status rc = dataset::Tensor::CreateFromMemory(dataset::TensorShape(host_input.Shape()),
                                                MSTypeToDEType(static_cast<TypeId>(host_input.DataType())),
                                                (const uchar *)(host_input.Data().get()), &de_input);
  RETURN_IF_NOT_OK(rc);
  if (!IsNonEmptyJPEG(de_input)) {
    std::string err_msg = "Dvpp operators can only support processing JPEG image";
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_UNEXPECTED(err_msg);
  }

  APP_ERROR ret = AclAdapter::GetInstance().H2D_Sink(processor_.get(), de_input, device_input);
  if (ret != APP_ERR_OK) {
    AclAdapter::GetInstance().Release();
    std::string err_msg = "Error in data sink process:" + std::to_string(ret);
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_UNEXPECTED(err_msg);
  }
  MS_LOG(INFO) << "Process data sink successfully";
  return Status::OK();
}

Status AscendResource::Pop(const std::shared_ptr<DeviceTensor> &device_output, std::shared_ptr<Tensor> *host_output) {
  APP_ERROR ret = AclAdapter::GetInstance().D2H_Pop(processor_.get(), device_output, host_output);
  if (ret != APP_ERR_OK) {
    AclAdapter::GetInstance().Release();
    std::string err_msg = "Error in data pop processing:" + std::to_string(ret);
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_UNEXPECTED(err_msg);
  }
  return Status::OK();
}

Status AscendResource::DeviceDataRelease() {
  APP_ERROR ret = AclAdapter::GetInstance().DeviceMemoryRelease(processor_.get());
  if (ret != APP_ERR_OK) {
    AclAdapter::GetInstance().Release();
    std::string err_msg = "Error in device data release:" + std::to_string(ret);
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_UNEXPECTED(err_msg);
  }
  return Status::OK();
}

std::shared_ptr<void> AscendResource::GetInstance() { return processor_; }

void *AscendResource::GetContext() { return AclAdapter::GetInstance().GetContextFromAclProcess(processor_.get()); }

void *AscendResource::GetStream() { return AclAdapter::GetInstance().GetStreamFromAclProcess(processor_.get()); }

}  // namespace dataset
}  // namespace mindspore
