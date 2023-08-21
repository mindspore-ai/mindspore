/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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
#include "minddata/dataset/core/device_tensor_ascend910b.h"

#include <string>

#include "minddata/dataset/core/global_context.h"
#include "minddata/dataset/core/type_id.h"
#include "minddata/dataset/kernels/image/dvpp/acl_adapter.h"
#include "minddata/dataset/kernels/image/image_utils.h"
#include "minddata/dataset/util/status.h"

namespace mindspore {
namespace dataset {
DeviceTensorAscend910B::DeviceTensorAscend910B(const TensorShape &shape, const DataType &type,
                                               device::DeviceContext *device_context, const size_t &stream_id)
    : device_context_(device_context),
      stream_id_(stream_id),
      device_address_ptr_(nullptr),
      tensor_(nullptr),
      tensor_shape_(shape),
      data_type_(type) {}

// create device_tensor by empty
Status DeviceTensorAscend910B::CreateDeviceTensor(const TensorShape &shape, const DataType &type,
                                                  device::DeviceContext *device_context, const size_t &stream_id,
                                                  std::shared_ptr<DeviceTensorAscend910B> *out) {
  RETURN_UNEXPECTED_IF_NULL(device_context);
  RETURN_UNEXPECTED_IF_NULL(out);

  // change the shape to NHWC
  TensorShape shape_nhwc(shape);
  if (shape_nhwc.Rank() == kMinImageRank) {  // expand HW to 1HW1
    shape_nhwc = shape_nhwc.AppendDim(1);
    shape_nhwc = shape_nhwc.PrependDim(1);
  } else if (shape_nhwc.Rank() == kDefaultImageRank) {  // expand HWC to 1HWC
    if (shape_nhwc.AsVector()[2] != 1 && shape_nhwc.AsVector()[2] != 3) {
      RETURN_STATUS_UNEXPECTED("The channel of input tensor HWC is not 1 or 3. It is " +
                               std::to_string(shape_nhwc.AsVector()[2]));
    }
    shape_nhwc = shape_nhwc.PrependDim(1);
  } else if (shape_nhwc.Rank() == kDefaultImageRank + 1) {  // NHWC
    if (shape_nhwc.AsVector()[3] != 1 && shape_nhwc.AsVector()[3] != 3) {
      RETURN_STATUS_UNEXPECTED("The channel of input tensor NHWC is not 1 or 3. It is " +
                               std::to_string(shape_nhwc.AsVector()[3]));
    }
  } else {
    RETURN_STATUS_UNEXPECTED("The input tensor is not HW, HWC or NHWC.");
  }

  *out = std::make_shared<DeviceTensorAscend910B>(shape_nhwc, type, device_context, stream_id);

  // create new device address for data copy
  auto device_address_ptr = device_context->device_res_manager_->CreateDeviceAddress(
    nullptr, (*out)->GetShape().NumOfElements() * (*out)->GetType().SizeInBytes(), "",
    DETypeToMSType((*out)->GetType()), (*out)->GetShape().AsVector());

  MS_EXCEPTION_IF_NULL(device_address_ptr);
  if (device_address_ptr->GetPtr() == nullptr &&
      !device_context->device_res_manager_->AllocateMemory(device_address_ptr.get())) {
    RETURN_STATUS_UNEXPECTED("Allocate dynamic workspace memory failed");
  }

  (*out)->SetDeviceAddress(device_address_ptr);

  // create the stride
  std::vector<int64_t> strides((*out)->GetShape().Rank(), 1);
  for (int64_t i = (*out)->GetShape().Rank() - 2; i >= 0; i--) {
    strides[i] = (*out)->GetShape()[i + 1] * strides[i + 1];
  }

  // create aclTensor
  auto device_tensor = aclCreateTensor((*out)->GetShape().AsVector().data(), (*out)->GetShape().Rank(),
                                       DETypeToaclDataType((*out)->GetType()), strides.data(), 0,
                                       aclFormat::ACL_FORMAT_NHWC, (*out)->GetShape().AsVector().data(),
                                       (*out)->GetShape().Rank(), (*out)->GetDeviceAddress()->GetMutablePtr());
  CHECK_FAIL_RETURN_UNEXPECTED(device_tensor != nullptr, "aclCreateTensor failed.");

  (*out)->SetDeviceTensor(device_tensor);

  return Status::OK();
}

// create device_tensor by host tensor
Status DeviceTensorAscend910B::CreateDeviceTensor(std::shared_ptr<Tensor> tensor, device::DeviceContext *device_context,
                                                  const size_t &stream_id,
                                                  std::shared_ptr<DeviceTensorAscend910B> *out) {
  RETURN_UNEXPECTED_IF_NULL(tensor);
  RETURN_UNEXPECTED_IF_NULL(device_context);
  RETURN_UNEXPECTED_IF_NULL(out);

  RETURN_IF_NOT_OK(
    DeviceTensorAscend910B::CreateDeviceTensor(tensor->shape(), tensor->type(), device_context, stream_id, out));

  CHECK_FAIL_RETURN_UNEXPECTED(
    tensor->SizeInBytes() == (*out)->GetShape().NumOfElements() * (*out)->GetType().SizeInBytes(),
    "The device SizeInBytes is not equal the input tensor.");

  // copy the host data to device
  auto ret =
    aclrtMemcpy((*out)->GetDeviceAddress()->GetMutablePtr(), tensor->SizeInBytes(),
                reinterpret_cast<void *>(tensor->GetMutableBuffer()), tensor->SizeInBytes(), ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_FAIL_RETURN_UNEXPECTED(ret == ACL_SUCCESS, "aclrtMemcpy failed. ERROR: " + std::to_string(ret));

  return Status::OK();
}

Status DeviceTensorAscend910B::ToHostTensor(std::shared_ptr<Tensor> *host_tensor) {
  CHECK_FAIL_RETURN_UNEXPECTED(host_tensor != nullptr, "The host tensor is nullptr pointer.");
  CHECK_FAIL_RETURN_UNEXPECTED(device_address_ptr_ != nullptr, "The device tensor is nullptr pointer.");

  if (!device_context_->device_res_manager_->SyncStream(stream_id_)) {
    std::string err_msg = "SyncStream stream id: " + std::to_string(stream_id_) + " failed.";
    RETURN_STATUS_UNEXPECTED(err_msg);
  }

  RETURN_IF_NOT_OK(Tensor::CreateEmpty(GetShape(), GetType(), host_tensor));

  // copy the host data to device
  auto ret = aclrtMemcpy((*host_tensor)->GetMutableBuffer(), (*host_tensor)->SizeInBytes(),
                         device_address_ptr_->GetPtr(), (*host_tensor)->SizeInBytes(), ACL_MEMCPY_DEVICE_TO_HOST);
  if (ret != ACL_SUCCESS) {
    std::string err_msg = "aclrtMemcpy failed. ERROR: " + std::to_string(ret);
    RETURN_STATUS_UNEXPECTED(err_msg);
  }

  // release the device memory
  device_context_->device_res_manager_->FreeMemory(device_address_ptr_.get());

  (*host_tensor)->Squeeze();

  return Status::OK();
}
}  // namespace dataset
}  // namespace mindspore
