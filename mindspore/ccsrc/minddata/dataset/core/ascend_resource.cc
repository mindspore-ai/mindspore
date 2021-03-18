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

#include "include/api/context.h"
#include "include/api/types.h"
#include "minddata/dataset/core/ascend_resource.h"
#include "minddata/dataset/core/type_id.h"
#include "minddata/dataset/kernels/image/image_utils.h"

namespace mindspore {
namespace dataset {

Status AscendResource::InitResource(uint32_t device_id) {
  ResourceInfo resource;
  resource.aclConfigPath = "";
  resource.deviceIds.insert(device_id);
  ascend_resource_ = ResourceManager::GetInstance();
  APP_ERROR ret = ascend_resource_->InitResource(resource);
  if (ret != APP_ERR_OK) {
    ascend_resource_->Release();
    std::string err_msg = "Error in Init D-chip:" + std::to_string(ret);
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_UNEXPECTED(err_msg);
  }
  int cur_device_id = *(resource.deviceIds.begin());
  aclrtContext context = ascend_resource_->GetContext(cur_device_id);
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

Status AscendResource::FinalizeResource() {
  processor_->Release();
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

Status AscendResource::Pop(const std::shared_ptr<DeviceTensor> &device_output, std::shared_ptr<Tensor> *host_output) {
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

std::shared_ptr<void> AscendResource::GetInstance() { return processor_; }

void *AscendResource::GetContext() { return processor_->GetContext(); }

void *AscendResource::GetStream() { return processor_->GetStream(); }

}  // namespace dataset
}  // namespace mindspore
