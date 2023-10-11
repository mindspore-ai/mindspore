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
#include "minddata/dataset/kernels/image/dvpp/utils/ErrorCode.h"
#include "minddata/dataset/kernels/image/image_utils.h"
#include "minddata/dataset/util/status.h"

namespace mindspore {
namespace dataset {
DeviceTensorAscend910B::DeviceTensorAscend910B(const TensorShape &shape, const DataType &type,
                                               device::DeviceContext *device_context, const size_t &stream_id,
                                               bool is_hwc)
    : device_context_(device_context),
      stream_id_(stream_id),
      device_address_(nullptr),
      tensor_(nullptr),
      tensor_shape_(shape),
      data_type_(type),
      is_hwc_(is_hwc) {}

DeviceTensorAscend910B::~DeviceTensorAscend910B() {}

Status ValidShape(TensorShape *input_shape, bool is_hwc) {
  // change the shape from HWC to 1HWC
  if (input_shape->Rank() == 1) {                                                    // no expand
    MS_LOG(DEBUG) << "The input is not RGB which is no need convert to 1HWC/1CHW.";  // used by Dvpp Decode
    return Status::OK();
  }

  const auto kChannelIndexHWC = 2;
  const auto kChannelIndexNHWC = 3;
  const auto kDefaultImageChannel = 3;
  if (is_hwc) {
    // change the shape from HWC to 1HWC
    if (input_shape->Rank() == kMinImageRank) {  // expand HW to 1HW1
      *input_shape = input_shape->AppendDim(1);
      *input_shape = input_shape->PrependDim(1);
    } else if (input_shape->Rank() == kDefaultImageRank) {  // expand HWC to 1HWC
      if (input_shape->AsVector()[kChannelIndexHWC] != 1 &&
          input_shape->AsVector()[kChannelIndexHWC] != kDefaultImageChannel) {
        RETURN_STATUS_UNEXPECTED("The channel of the input tensor of shape [H,W,C] is not 1 or 3, but got: " +
                                 std::to_string(input_shape->AsVector()[kChannelIndexHWC]));
      }
      *input_shape = input_shape->PrependDim(1);
    } else if (input_shape->Rank() == kDefaultImageRank + 1) {  // NHWC
      if (input_shape->AsVector()[kChannelIndexNHWC] != 1 &&
          input_shape->AsVector()[kChannelIndexNHWC] != kDefaultImageChannel) {
        RETURN_STATUS_UNEXPECTED("The channel of the input tensor of shape [N,H,W,C] is not 1 or 3, but got: " +
                                 std::to_string(input_shape->AsVector()[kChannelIndexNHWC]));
      }
      if (input_shape->AsVector()[0] != 1) {
        RETURN_STATUS_UNEXPECTED("The input tensor NHWC should be 1HWC or HWC.");
      }
    } else {
      RETURN_STATUS_UNEXPECTED("The input tensor is not of shape [H,W], [H,W,C] or [N,H,W,C].");
    }
  } else {
    // change the shape from CHW to 1CHW
    if (input_shape->Rank() == kMinImageRank) {  // expand HW to 11HW
      *input_shape = input_shape->PrependDim(1);
      *input_shape = input_shape->PrependDim(1);
    } else if (input_shape->Rank() == kDefaultImageRank) {  // expand CHW to 1CHW
      if (input_shape->AsVector()[0] != 1 && input_shape->AsVector()[0] != kDefaultImageChannel) {
        RETURN_STATUS_UNEXPECTED("The channel of the input tensor of shape [C,H,W] is not 1 or 3, but got: " +
                                 std::to_string(input_shape->AsVector()[0]));
      }
      *input_shape = input_shape->PrependDim(1);
    } else if (input_shape->Rank() == kDefaultImageRank + 1) {  // NCHW
      if (input_shape->AsVector()[1] != 1 && input_shape->AsVector()[1] != kDefaultImageChannel) {
        RETURN_STATUS_UNEXPECTED("The channel of the input tensor of shape [N,C,H,W] is not 1 or 3, but got: " +
                                 std::to_string(input_shape->AsVector()[1]));
      }
      if (input_shape->AsVector()[0] != 1) {
        RETURN_STATUS_UNEXPECTED("The input tensor NCHW should be 1CHW or CHW.");
      }
    } else {
      RETURN_STATUS_UNEXPECTED("The input tensor is not of shape [H,W], [C,H,W] or [N,C,H,W].");
    }
  }
  return Status::OK();
}

// create device_tensor by empty
Status DeviceTensorAscend910B::CreateDeviceTensor(const TensorShape &shape, const DataType &type,
                                                  device::DeviceContext *device_context, const size_t &stream_id,
                                                  std::shared_ptr<DeviceTensorAscend910B> *out, bool is_hwc) {
  RETURN_UNEXPECTED_IF_NULL(device_context);
  RETURN_UNEXPECTED_IF_NULL(out);

  TensorShape input_shape(shape);
  RETURN_IF_NOT_OK(ValidShape(&input_shape, is_hwc));

  *out = std::make_shared<DeviceTensorAscend910B>(input_shape, type, device_context, stream_id, is_hwc);

  // create new device address for data copy
  void *device_address = device_context->device_res_manager_->AllocateMemory((*out)->GetShape().NumOfElements() *
                                                                             (*out)->GetType().SizeInBytes());
  if (device_address == nullptr) {
    RETURN_STATUS_UNEXPECTED("Allocate dynamic workspace memory failed");
  }

  (*out)->SetDeviceAddress(device_address);

  // create the stride
  std::vector<int64_t> strides((*out)->GetShape().Rank(), 1);
  for (int64_t i = (*out)->GetShape().Rank() - 2; i >= 0; i--) {
    strides[i] = (*out)->GetShape()[i + 1] * strides[i + 1];
  }

  // create aclTensor, here we use void* hold it.
  void *device_tensor = nullptr;
  auto ret = AclAdapter::GetInstance().CreateAclTensor((*out)->GetShape().AsVector().data(), (*out)->GetShape().Rank(),
                                                       DETypeToMSType((*out)->GetType()), strides.data(), 0,
                                                       (*out)->GetShape().AsVector().data(), (*out)->GetShape().Rank(),
                                                       (*out)->GetDeviceAddress(), is_hwc, &device_tensor);
  if (ret != APP_ERR_OK) {
    std::string error = "Create acl tensor failed.";
    RETURN_STATUS_UNEXPECTED(error);
  }
  CHECK_FAIL_RETURN_UNEXPECTED(device_tensor != nullptr, "Create device tensor failed.");

  (*out)->SetDeviceTensor(device_tensor);

  return Status::OK();
}

// create device_tensor by host tensor
Status DeviceTensorAscend910B::CreateDeviceTensor(std::shared_ptr<Tensor> tensor, device::DeviceContext *device_context,
                                                  const size_t &stream_id, std::shared_ptr<DeviceTensorAscend910B> *out,
                                                  bool is_hwc) {
  RETURN_UNEXPECTED_IF_NULL(tensor);
  RETURN_UNEXPECTED_IF_NULL(device_context);
  RETURN_UNEXPECTED_IF_NULL(out);

  RETURN_IF_NOT_OK(DeviceTensorAscend910B::CreateDeviceTensor(tensor->shape(), tensor->type(), device_context,
                                                              stream_id, out, is_hwc));

  CHECK_FAIL_RETURN_UNEXPECTED(
    tensor->SizeInBytes() == (*out)->GetShape().NumOfElements() * (*out)->GetType().SizeInBytes(),
    "The device SizeInBytes is not equal the input tensor.");

  // copy the host data to device
  (void)(*out)->GetDeviceContext()->device_res_manager_->SwapIn(
    reinterpret_cast<void *>(tensor->GetMutableBuffer()), (*out)->GetDeviceAddress(), tensor->SizeInBytes(), nullptr);

  return Status::OK();
}

Status DeviceTensorAscend910B::ToHostTensor(std::shared_ptr<Tensor> *host_tensor) {
  CHECK_FAIL_RETURN_UNEXPECTED(host_tensor != nullptr, "The host tensor is nullptr pointer.");
  CHECK_FAIL_RETURN_UNEXPECTED(device_address_ != nullptr, "The device tensor is nullptr pointer.");

  if (!device_context_->device_res_manager_->SyncStream(stream_id_)) {
    std::string err_msg = "SyncStream stream id: " + std::to_string(stream_id_) + " failed.";
    RETURN_STATUS_UNEXPECTED(err_msg);
  }

  RETURN_IF_NOT_OK(Tensor::CreateEmpty(GetShape(), GetType(), host_tensor));

  // copy the device data to host
  (void)device_context_->device_res_manager_->SwapOut(device_address_,
                                                      reinterpret_cast<void *>((*host_tensor)->GetMutableBuffer()),
                                                      (*host_tensor)->SizeInBytes(), nullptr);

  // release the device memory
  (void)device_context_->device_res_manager_->FreeMemory(device_address_);

  (*host_tensor)->Squeeze();

  return Status::OK();
}
}  // namespace dataset
}  // namespace mindspore
