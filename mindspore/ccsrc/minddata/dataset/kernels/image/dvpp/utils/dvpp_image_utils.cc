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
#include "minddata/dataset/kernels/image/dvpp/utils/dvpp_image_utils.h"

#include <opencv2/imgproc/types_c.h>
#include <algorithm>
#include <fstream>
#include <limits>
#include <string>
#include <vector>
#include <stdexcept>
#include <opencv2/imgcodecs.hpp>
#include "utils/file_utils.h"
#include "utils/ms_utils.h"
#include "minddata/dataset/core/cv_tensor.h"
#include "minddata/dataset/core/tensor.h"
#include "minddata/dataset/core/tensor_shape.h"
#include "minddata/dataset/core/type_id.h"
#include "minddata/dataset/include/dataset/constants.h"
#include "minddata/dataset/kernels/image/affine_op.h"
#include "minddata/dataset/kernels/image/auto_contrast_op.h"
#include "minddata/dataset/kernels/image/invert_op.h"
#include "minddata/dataset/kernels/image/image_utils.h"
#include "minddata/dataset/kernels/image/math_utils.h"
#include "minddata/dataset/kernels/image/posterize_op.h"
#include "minddata/dataset/kernels/image/resize_cubic_op.h"
#include "minddata/dataset/kernels/image/sharpness_op.h"
#include "minddata/dataset/kernels/image/solarize_op.h"
#include "minddata/dataset/kernels/data/data_utils.h"

#include "dvpp/acldvppop/acldvpp_adjust_brightness.h"
#include "dvpp/acldvppop/acldvpp_adjust_contrast.h"
#include "dvpp/acldvppop/acldvpp_adjust_hue.h"
#include "dvpp/acldvppop/acldvpp_adjust_saturation.h"
#include "dvpp/acldvppop/acldvpp_decode_jpeg.h"
#include "dvpp/acldvppop/acldvpp_normalize.h"
#include "dvpp/acldvppop/acldvpp_resize.h"
#include "plugin/device/ascend/hal/device/ascend_stream_manager.h"

namespace mindspore {
namespace dataset {
const auto kChannelIndexNHWC = 3;
const auto kNHWCImageRank = 4;
const auto kWidthIndexNHWC = 2;

APP_ERROR DvppResize(const std::shared_ptr<DeviceTensorAscend910B> &input,
                     std::shared_ptr<DeviceTensorAscend910B> *output, int32_t output_height, int32_t output_width,
                     double fx, double fy, InterpolationMode mode) {
  MS_LOG(DEBUG) << "Begin execute dvpp resize.";
  if (input == nullptr || output == nullptr) {
    MS_LOG(ERROR) << "The input or output is nullptr.";
    return APP_ERR_DVPP_RESIZE_FAIL;
  }

  // the input should be HWC
  if (input->GetShape().Rank() != kNHWCImageRank) {
    MS_LOG(ERROR) << "The input data's dims is not 4.";  // NHWC
    return APP_ERR_DVPP_RESIZE_FAIL;
  }
  if (input->GetShape().AsVector()[kChannelIndexNHWC] != kDefaultImageChannel &&
      input->GetShape().AsVector()[kChannelIndexNHWC] != 1) {
    MS_LOG(ERROR) << "The input data's channel is not 3 or 1.";  // C == 3 or 1
    return APP_ERR_DVPP_RESIZE_FAIL;
  }

  const uint32_t kResizeShapeLimits = 1000;
  // resize image too large or too small, 1000 is arbitrarily chosen here to prevent open cv from segmentation fault
  if ((std::numeric_limits<int>::max() / kResizeShapeLimits) <= input->GetShape().AsVector()[1]) {
    MS_LOG(ERROR) << "DvppResize: in_image rows out of bounds.";
    return APP_ERR_DVPP_RESIZE_FAIL;
  }
  if ((std::numeric_limits<int>::max() / kResizeShapeLimits) <= input->GetShape().AsVector()[kWidthIndexNHWC]) {
    MS_LOG(ERROR) << "DvppResize: in_image cols out of bounds.";
    return APP_ERR_DVPP_RESIZE_FAIL;
  }
  if (output_height > input->GetShape().AsVector()[1] * kResizeShapeLimits ||
      output_width > input->GetShape().AsVector()[kWidthIndexNHWC] * kResizeShapeLimits) {
    std::string err_msg =
      "DvppResize: the resizing width or height is too big, it's 1000 times bigger than the original image, got output "
      "height: " +
      std::to_string(output_height) + ", width: " + std::to_string(output_width) +
      ", and original image size:" + std::to_string(input->GetShape().AsVector()[1]) + ", " +
      std::to_string(input->GetShape().AsVector()[kWidthIndexNHWC]);
    MS_LOG(ERROR) << err_msg;
    return APP_ERR_DVPP_RESIZE_FAIL;
  }
  if (output_height == 0 || output_width == 0) {
    std::string err_msg = "DvppResize: the input value of 'resize' is invalid, width or height is zero.";
    MS_LOG(ERROR) << err_msg;
    return APP_ERR_DVPP_RESIZE_FAIL;
  }

  // create the output shape and type, it's HWC
  TensorShape shape{static_cast<int>(input->GetShape()[0]), output_height, output_width,
                    static_cast<int>(input->GetShape()[kChannelIndexHWC + 1])};
  DataType type = input->GetType();

  // create output DeviceTensorAscend910B
  std::shared_ptr<DeviceTensorAscend910B> device_tensor = nullptr;
  if (DeviceTensorAscend910B::CreateDeviceTensor(shape, type, input->GetDeviceContext(), input->GetStreamID(),
                                                 &device_tensor) != Status::OK()) {
    MS_LOG(ERROR) << "Create output device tensor failed.";
    return APP_ERR_DVPP_RESIZE_FAIL;
  }

  // convert InterpolationMode mode to DVPP mode
  auto dvpp_interpolation_mode = GetDVPPInterpolationMode(mode);
  if (dvpp_interpolation_mode == kInvalidInterpolationMode) {
    std::string err_msg =
      "The current InterpolationMode is not supported by DVPP. It is " + std::to_string(static_cast<int>(mode));
    MS_LOG(ERROR) << err_msg;
    return APP_ERR_DVPP_RESIZE_FAIL;
  }

  // call DVPP step1
  uint64_t workspace_size = 0;
  aclOpExecutor *executor;
  auto ret = acldvppResizeGetWorkspaceSize(
    reinterpret_cast<aclTensor *>(input->GetDeviceTensor()), dvpp_interpolation_mode,
    reinterpret_cast<aclTensor *>(device_tensor->GetDeviceTensor()), &workspace_size, &executor);
  if (ret != ACL_SUCCESS) {
    MS_LOG(ERROR) << "Call acldvppResizeGetWorkspaceSize failed, error code: " + std::to_string(ret) + ".";
    return APP_ERR_DVPP_RESIZE_FAIL;
  }

  // call DVPP step2
  void *workspace_addr = nullptr;
  if (workspace_size > 0) {
    // create new device address for data copy
    workspace_addr = input->GetDeviceContext()->device_res_manager_->AllocateMemory(workspace_size);
    if (workspace_addr == nullptr) {
      MS_LOG(ERROR) << "Allocate dynamic workspace memory failed";
      return APP_ERR_DVPP_RESIZE_FAIL;
    }

    // call DVPP step3
    ret = acldvppResize(
      workspace_addr, workspace_size, executor,
      static_cast<aclrtStream>(input->GetDeviceContext()->device_res_manager_->GetStream(input->GetStreamID())));

    if (!input->GetDeviceContext()->device_res_manager_->SyncStream(input->GetStreamID())) {
      MS_LOG(ERROR) << "SyncStream stream id: " << std::to_string(input->GetStreamID()) << " failed.";
      return APP_ERR_DVPP_RESIZE_FAIL;
    }

    // release workspace_addr
    (void)input->GetDeviceContext()->device_res_manager_->FreeMemory(workspace_addr);
  } else {
    // call DVPP step3
    ret = acldvppResize(
      nullptr, workspace_size, executor,
      static_cast<aclrtStream>(input->GetDeviceContext()->device_res_manager_->GetStream(input->GetStreamID())));
  }

  if (ret != ACL_SUCCESS) {
    MS_LOG(ERROR) << "Call aclvisionResize failed, error code: " + std::to_string(ret) + ".";
    return APP_ERR_DVPP_RESIZE_FAIL;
  }

  *output = std::move(device_tensor);  // currently the data is still in device
  return APP_ERR_OK;
}

APP_ERROR DvppDecode(const std::shared_ptr<DeviceTensorAscend910B> &input,
                     std::shared_ptr<DeviceTensorAscend910B> *output) {
  MS_LOG(DEBUG) << "Begin execute dvpp decode.";
  if (input == nullptr || output == nullptr) {
    MS_LOG(ERROR) << "The input or output is nullptr.";
    return APP_ERR_DVPP_JPEG_DECODE_FAIL;
  }

  // the output DeviceTensorAscend910B had been created in npu_map_job.cc,
  // because we need get image height and width from the JPEG header
  std::shared_ptr<DeviceTensorAscend910B> device_tensor = *output;

  // call DVPP step1
  uint64_t workspace_size = 0;
  aclOpExecutor *executor;
  // decode output C is 3
  // don't recovery truncate
  auto ret = acldvppDecodeJpegGetWorkspaceSize(reinterpret_cast<aclTensor *>(input->GetDeviceTensor()), 3, true,
                                               reinterpret_cast<aclTensor *>(device_tensor->GetDeviceTensor()),
                                               &workspace_size, &executor);
  if (ret != ACL_SUCCESS) {
    MS_LOG(ERROR) << "Call acldvppDecodeJpegGetWorkspaceSize failed, error code: " + std::to_string(ret) + ".";
    return APP_ERR_DVPP_JPEG_DECODE_FAIL;
  }

  // call DVPP step2
  void *workspace_addr = nullptr;
  if (workspace_size > 0) {
    // create new device address for data copy
    workspace_addr = input->GetDeviceContext()->device_res_manager_->AllocateMemory(workspace_size);
    if (workspace_addr == nullptr) {
      MS_LOG(ERROR) << "Allocate dynamic workspace memory failed";
      return APP_ERR_DVPP_JPEG_DECODE_FAIL;
    }

    // call DVPP step3
    ret = acldvppDecodeJpeg(
      workspace_addr, workspace_size, executor,
      static_cast<aclrtStream>(input->GetDeviceContext()->device_res_manager_->GetStream(input->GetStreamID())));

    if (!input->GetDeviceContext()->device_res_manager_->SyncStream(input->GetStreamID())) {
      MS_LOG(ERROR) << "SyncStream stream id: " << std::to_string(input->GetStreamID()) << " failed.";
      return APP_ERR_DVPP_JPEG_DECODE_FAIL;
    }

    // release workspace_addr
    (void)input->GetDeviceContext()->device_res_manager_->FreeMemory(workspace_addr);
  } else {
    // call DVPP step3
    ret = acldvppDecodeJpeg(
      nullptr, workspace_size, executor,
      static_cast<aclrtStream>(input->GetDeviceContext()->device_res_manager_->GetStream(input->GetStreamID())));
  }

  if (ret != ACL_SUCCESS) {
    MS_LOG(ERROR) << "Call acldvppDecodeJpeg failed, error code: " + std::to_string(ret) + ".";
    return APP_ERR_DVPP_JPEG_DECODE_FAIL;
  }

  *output = std::move(device_tensor);  // currently the data is still in device
  return APP_ERR_OK;
}

APP_ERROR DvppNormalize(const std::shared_ptr<DeviceTensorAscend910B> &input,
                        std::shared_ptr<DeviceTensorAscend910B> *output, std::vector<float> mean,
                        std::vector<float> std, bool is_hwc) {
  MS_LOG(DEBUG) << "Begin execute dvpp normalize.";
  if (input == nullptr || output == nullptr) {
    MS_LOG(ERROR) << "The input or output is nullptr.";
    return APP_ERR_DVPP_NORMALIZE_FAIL;
  }

  // the input should be 1HWC or 1CHW
  if (input->GetShape().Rank() != kNHWCImageRank) {
    MS_LOG(ERROR) << "The input data's dims is not 4.";  // NHWC
    return APP_ERR_DVPP_NORMALIZE_FAIL;
  }

  if (!is_hwc) {
    if (input->GetShape().AsVector()[1] != kDefaultImageChannel && input->GetShape().AsVector()[1] != 1) {
      MS_LOG(ERROR) << "The input data's channel is not 3 or 1.";  // C == 3 or 1
      return APP_ERR_DVPP_NORMALIZE_FAIL;
    }

    // the channel should be equal to the size of mean
    if (mean.size() != std.size() || std.size() != input->GetShape().AsVector()[1]) {
      MS_LOG(ERROR) << "The channel is not equal to the size of mean or std.";
      return APP_ERR_DVPP_NORMALIZE_FAIL;
    }
  } else {
    if (input->GetShape().AsVector()[kChannelIndexNHWC] != kDefaultImageChannel &&
        input->GetShape().AsVector()[kChannelIndexNHWC] != 1) {
      MS_LOG(ERROR) << "The input data's channel is not 3 or 1.";  // C == 3 or 1
      return APP_ERR_DVPP_NORMALIZE_FAIL;
    }

    // the channel should be equal to the size of mean
    if (mean.size() != std.size() || std.size() != input->GetShape().AsVector()[kChannelIndexNHWC]) {
      MS_LOG(ERROR) << "The channel is not equal to the size of mean or std.";
      return APP_ERR_DVPP_NORMALIZE_FAIL;
    }
  }

  if (input->GetShape().AsVector()[0] != 1) {
    MS_LOG(ERROR) << "The input data is not 1HWC or 1CHW.";  // N == 1
    return APP_ERR_DVPP_NORMALIZE_FAIL;
  }

  // the type is uint8 / float
  if (input->GetType() != DataType::DE_UINT8 && input->GetType() != DataType::DE_FLOAT32) {
    MS_LOG(ERROR) << "The input data is not uint8 or float32";
    return APP_ERR_DVPP_NORMALIZE_FAIL;
  }

  // create the output shape and type, it's 1HWC or 1CHW
  TensorShape shape = input->GetShape();
  DataType type(DataType::DE_FLOAT32);

  // create output DeviceTensorAscend910B
  std::shared_ptr<DeviceTensorAscend910B> device_tensor = nullptr;
  if (DeviceTensorAscend910B::CreateDeviceTensor(shape, type, input->GetDeviceContext(), input->GetStreamID(),
                                                 &device_tensor, is_hwc) != Status::OK()) {
    MS_LOG(ERROR) << "Create output device tensor failed.";
    return APP_ERR_DVPP_NORMALIZE_FAIL;
  }

  // create aclFloatArray for mean
  aclFloatArray *acl_mean = aclCreateFloatArray(mean.data(), mean.size());

  // create aclFloatArray for std
  aclFloatArray *acl_std = aclCreateFloatArray(std.data(), std.size());

  // call DVPP step1
  uint64_t workspace_size = 0;
  aclOpExecutor *executor;
  auto ret = acldvppNormalizeGetWorkspaceSize(reinterpret_cast<aclTensor *>(input->GetDeviceTensor()), acl_mean,
                                              acl_std, reinterpret_cast<aclTensor *>(device_tensor->GetDeviceTensor()),
                                              &workspace_size, &executor);
  if (ret != ACL_SUCCESS) {
    MS_LOG(ERROR) << "Call aclvisionNormalizeGetWorkspaceSize failed, error code: " + std::to_string(ret) + ".";
    return APP_ERR_DVPP_NORMALIZE_FAIL;
  }

  // call DVPP step2
  void *workspace_addr = nullptr;
  if (workspace_size > 0) {
    // create new device address for data copy
    workspace_addr = input->GetDeviceContext()->device_res_manager_->AllocateMemory(workspace_size);
    if (workspace_addr == nullptr) {
      MS_LOG(ERROR) << "Allocate dynamic workspace memory failed";
      return APP_ERR_DVPP_NORMALIZE_FAIL;
    }

    // call DVPP step3
    ret = acldvppNormalize(
      workspace_addr, workspace_size, executor,
      static_cast<aclrtStream>(input->GetDeviceContext()->device_res_manager_->GetStream(input->GetStreamID())));

    if (!input->GetDeviceContext()->device_res_manager_->SyncStream(input->GetStreamID())) {
      MS_LOG(ERROR) << "SyncStream stream id: " << std::to_string(input->GetStreamID()) << " failed.";
      return APP_ERR_DVPP_NORMALIZE_FAIL;
    }

    // release workspace_addr
    (void)input->GetDeviceContext()->device_res_manager_->FreeMemory(workspace_addr);
  } else {
    // call DVPP step3
    ret = acldvppNormalize(
      nullptr, workspace_size, executor,
      static_cast<aclrtStream>(input->GetDeviceContext()->device_res_manager_->GetStream(input->GetStreamID())));
  }

  if (ret != ACL_SUCCESS) {
    MS_LOG(ERROR) << "Call acldvppNormalize failed, error code: " + std::to_string(ret) + ".";
    return APP_ERR_DVPP_NORMALIZE_FAIL;
  }

  *output = std::move(device_tensor);  // currently the data is still in device
  return APP_ERR_OK;
}

APP_ERROR DvppAdjustBrightness(const std::shared_ptr<DeviceTensorAscend910B> &input,
                               std::shared_ptr<DeviceTensorAscend910B> *output, float factor) {
  MS_LOG(DEBUG) << "Begin execute adjust brightness.";
  if (input == nullptr || output == nullptr) {
    MS_LOG(ERROR) << "The input or output is nullptr.";
    return APP_ERR_DVPP_ADJUST_BRIGHTNESS_FAIL;
  }

  // the input should be 1HWC or 1CHW
  if (input->GetShape().Rank() != kNHWCImageRank) {
    MS_LOG(ERROR) << "The input data's dims is not 4.";  // NHWC
    return APP_ERR_DVPP_ADJUST_BRIGHTNESS_FAIL;
  }

  // the channel should be equal to 3 or 1
  if (input->GetShape().AsVector()[kChannelIndexNHWC] != kDefaultImageChannel &&
      input->GetShape().AsVector()[kChannelIndexNHWC] != 1) {
    MS_LOG(ERROR) << "The input data's channel is not 3 or 1.";
    return APP_ERR_DVPP_ADJUST_BRIGHTNESS_FAIL;
  }

  if (input->GetShape().AsVector()[0] != 1) {
    MS_LOG(ERROR) << "The input data is not 1HWC or 1CHW.";  // N == 1
    return APP_ERR_DVPP_ADJUST_BRIGHTNESS_FAIL;
  }

  // the type is uint8 / float
  if (input->GetType() != DataType::DE_UINT8 && input->GetType() != DataType::DE_FLOAT32) {
    MS_LOG(ERROR) << "The input data is not uint8 or float32";
    return APP_ERR_DVPP_ADJUST_BRIGHTNESS_FAIL;
  }

  // create the output shape and type, it's 1HWC or 1CHW
  TensorShape shape = input->GetShape();
  DataType type = input->GetType();

  // create output DeviceTensorAscend910B
  std::shared_ptr<DeviceTensorAscend910B> device_tensor = nullptr;
  if (DeviceTensorAscend910B::CreateDeviceTensor(shape, type, input->GetDeviceContext(), input->GetStreamID(),
                                                 &device_tensor, true) != Status::OK()) {
    MS_LOG(ERROR) << "Create output device tensor failed.";
    return APP_ERR_DVPP_ADJUST_BRIGHTNESS_FAIL;
  }

  // call DVPP step1
  uint64_t workspace_size = 0;
  aclOpExecutor *executor;
  auto ret = acldvppAdjustBrightnessGetWorkspaceSize(reinterpret_cast<aclTensor *>(input->GetDeviceTensor()), factor,
                                                     reinterpret_cast<aclTensor *>(device_tensor->GetDeviceTensor()),
                                                     &workspace_size, &executor);
  if (ret != ACL_SUCCESS) {
    MS_LOG(ERROR) << "Call acldvppAdjustBrightnessGetWorkspaceSize failed, error code: " + std::to_string(ret) + ".";
    return APP_ERR_DVPP_ADJUST_BRIGHTNESS_FAIL;
  }

  // call DVPP step2
  void *workspace_addr = nullptr;
  if (workspace_size > 0) {
    // create new device address for data copy
    workspace_addr = input->GetDeviceContext()->device_res_manager_->AllocateMemory(workspace_size);
    if (workspace_addr == nullptr) {
      MS_LOG(ERROR) << "Allocate dynamic workspace memory failed";
      return APP_ERR_DVPP_ADJUST_BRIGHTNESS_FAIL;
    }

    // call DVPP step3
    ret = acldvppAdjustBrightness(
      workspace_addr, workspace_size, executor,
      static_cast<aclrtStream>(input->GetDeviceContext()->device_res_manager_->GetStream(input->GetStreamID())));

    if (!input->GetDeviceContext()->device_res_manager_->SyncStream(input->GetStreamID())) {
      MS_LOG(ERROR) << "SyncStream stream id: " << std::to_string(input->GetStreamID()) << " failed.";
      return APP_ERR_DVPP_ADJUST_BRIGHTNESS_FAIL;
    }

    // release workspace_addr
    (void)input->GetDeviceContext()->device_res_manager_->FreeMemory(workspace_addr);
  } else {
    // call DVPP step3
    ret = acldvppAdjustBrightness(
      nullptr, workspace_size, executor,
      static_cast<aclrtStream>(input->GetDeviceContext()->device_res_manager_->GetStream(input->GetStreamID())));
  }

  if (ret != ACL_SUCCESS) {
    MS_LOG(ERROR) << "Call acldvppAdjustBrightness failed, error code: " + std::to_string(ret) + ".";
    return APP_ERR_DVPP_ADJUST_BRIGHTNESS_FAIL;
  }

  *output = std::move(device_tensor);  // currently the data is still in device
  return APP_ERR_OK;
}

APP_ERROR DvppAdjustContrast(const std::shared_ptr<DeviceTensorAscend910B> &input,
                             std::shared_ptr<DeviceTensorAscend910B> *output, float factor) {
  MS_LOG(WARNING) << "Begin execute adjust contrast.";
  if (input == nullptr || output == nullptr) {
    MS_LOG(ERROR) << "The input or output is nullptr.";
    return APP_ERR_DVPP_ADJUST_CONTRAST_FAIL;
  }

  // the input should be 1HWC or 1CHW
  if (input->GetShape().Rank() != kNHWCImageRank) {
    MS_LOG(ERROR) << "The input data's dims is not 4.";  // NHWC
    return APP_ERR_DVPP_ADJUST_CONTRAST_FAIL;
  }

  // the channel should be equal to 3
  if (input->GetShape().AsVector()[kChannelIndexNHWC] != kDefaultImageChannel) {
    MS_LOG(ERROR) << "The input data's channel is not 3.";
    return APP_ERR_DVPP_ADJUST_CONTRAST_FAIL;
  }

  if (input->GetShape().AsVector()[0] != 1) {
    MS_LOG(ERROR) << "The input data is not 1HWC or 1CHW.";  // N == 1
    return APP_ERR_DVPP_ADJUST_CONTRAST_FAIL;
  }

  // the type is uint8 / float
  if (input->GetType() != DataType::DE_UINT8 && input->GetType() != DataType::DE_FLOAT32) {
    MS_LOG(ERROR) << "The input data is not uint8 or float32";
    return APP_ERR_DVPP_ADJUST_CONTRAST_FAIL;
  }

  // create the output shape and type, it's 1HWC or 1CHW
  TensorShape shape = input->GetShape();
  DataType type = input->GetType();

  // create output DeviceTensorAscend910B
  std::shared_ptr<DeviceTensorAscend910B> device_tensor = nullptr;
  if (DeviceTensorAscend910B::CreateDeviceTensor(shape, type, input->GetDeviceContext(), input->GetStreamID(),
                                                 &device_tensor, true) != Status::OK()) {
    MS_LOG(ERROR) << "Create output device tensor failed.";
    return APP_ERR_DVPP_ADJUST_CONTRAST_FAIL;
  }

  // call DVPP step1
  uint64_t workspace_size = 0;
  aclOpExecutor *executor;
  auto ret = acldvppAdjustContrastGetWorkspaceSize(reinterpret_cast<aclTensor *>(input->GetDeviceTensor()), factor,
                                                   reinterpret_cast<aclTensor *>(device_tensor->GetDeviceTensor()),
                                                   &workspace_size, &executor);
  if (ret != ACL_SUCCESS) {
    MS_LOG(ERROR) << "Call acldvppAdjustContrastGetWorkspaceSize failed, error code: " + std::to_string(ret) + ".";
    return APP_ERR_DVPP_ADJUST_CONTRAST_FAIL;
  }

  // call DVPP step2
  void *workspace_addr = nullptr;
  if (workspace_size > 0) {
    // create new device address for data copy
    workspace_addr = input->GetDeviceContext()->device_res_manager_->AllocateMemory(workspace_size);
    if (workspace_addr == nullptr) {
      MS_LOG(ERROR) << "Allocate dynamic workspace memory failed";
      return APP_ERR_DVPP_ADJUST_CONTRAST_FAIL;
    }

    // call DVPP step3
    ret = acldvppAdjustContrast(
      workspace_addr, workspace_size, executor,
      static_cast<aclrtStream>(input->GetDeviceContext()->device_res_manager_->GetStream(input->GetStreamID())));

    if (!input->GetDeviceContext()->device_res_manager_->SyncStream(input->GetStreamID())) {
      MS_LOG(ERROR) << "SyncStream stream id: " << std::to_string(input->GetStreamID()) << " failed.";
      return APP_ERR_DVPP_ADJUST_CONTRAST_FAIL;
    }

    // release workspace_addr
    (void)input->GetDeviceContext()->device_res_manager_->FreeMemory(workspace_addr);
  } else {
    // call DVPP step3
    ret = acldvppAdjustContrast(
      nullptr, workspace_size, executor,
      static_cast<aclrtStream>(input->GetDeviceContext()->device_res_manager_->GetStream(input->GetStreamID())));
  }

  if (ret != ACL_SUCCESS) {
    MS_LOG(ERROR) << "Call acldvppAdjustContrast failed, error code: " + std::to_string(ret) + ".";
    return APP_ERR_DVPP_ADJUST_CONTRAST_FAIL;
  }

  *output = std::move(device_tensor);  // currently the data is still in device
  return APP_ERR_OK;
}

APP_ERROR DvppAdjustHue(const std::shared_ptr<DeviceTensorAscend910B> &input,
                        std::shared_ptr<DeviceTensorAscend910B> *output, float factor) {
  MS_LOG(WARNING) << "Begin execute adjust hue.";
  if (input == nullptr || output == nullptr) {
    MS_LOG(ERROR) << "The input or output is nullptr.";
    return APP_ERR_DVPP_ADJUST_HUE_FAIL;
  }

  // the input should be 1HWC or 1CHW
  if (input->GetShape().Rank() != kNHWCImageRank) {
    MS_LOG(ERROR) << "The input data's dims is not 4.";  // NHWC
    return APP_ERR_DVPP_ADJUST_HUE_FAIL;
  }

  // the channel should be equal to 3 or 1
  if (input->GetShape().AsVector()[3] != 3 && input->GetShape().AsVector()[3] != 1) {
    MS_LOG(ERROR) << "The input data's channel is not 3 or 1.";
    return APP_ERR_DVPP_ADJUST_HUE_FAIL;
  }

  if (input->GetShape().AsVector()[0] != 1) {
    MS_LOG(ERROR) << "The input data is not 1HWC or 1CHW.";  // N == 1
    return APP_ERR_DVPP_ADJUST_HUE_FAIL;
  }

  // the type is uint8 / float
  if (input->GetType() != DataType::DE_UINT8 && input->GetType() != DataType::DE_FLOAT32) {
    MS_LOG(ERROR) << "The input data is not uint8 or float32";
    return APP_ERR_DVPP_ADJUST_HUE_FAIL;
  }

  // create the output shape and type, it's 1HWC or 1CHW
  TensorShape shape = input->GetShape();
  DataType type = input->GetType();

  // create output DeviceTensorAscend910B
  std::shared_ptr<DeviceTensorAscend910B> device_tensor = nullptr;
  if (DeviceTensorAscend910B::CreateDeviceTensor(shape, type, input->GetDeviceContext(), input->GetStreamID(),
                                                 &device_tensor, true) != Status::OK()) {
    MS_LOG(ERROR) << "Create output device tensor failed.";
    return APP_ERR_DVPP_ADJUST_HUE_FAIL;
  }

  // call DVPP step1
  uint64_t workspace_size = 0;
  aclOpExecutor *executor;
  auto ret = acldvppAdjustHueGetWorkspaceSize(reinterpret_cast<aclTensor *>(input->GetDeviceTensor()), factor,
                                              reinterpret_cast<aclTensor *>(device_tensor->GetDeviceTensor()),
                                              &workspace_size, &executor);
  if (ret != ACL_SUCCESS) {
    MS_LOG(ERROR) << "Call acldvppAdjustHueGetWorkspaceSize failed, error code: " + std::to_string(ret) + ".";
    return APP_ERR_DVPP_ADJUST_HUE_FAIL;
  }

  // call DVPP step2
  void *workspace_addr = nullptr;
  if (workspace_size > 0) {
    // create new device address for data copy
    workspace_addr = input->GetDeviceContext()->device_res_manager_->AllocateMemory(workspace_size);
    if (workspace_addr == nullptr) {
      MS_LOG(ERROR) << "Allocate dynamic workspace memory failed";
      return APP_ERR_DVPP_ADJUST_HUE_FAIL;
    }

    // call DVPP step3
    ret = acldvppAdjustHue(
      workspace_addr, workspace_size, executor,
      static_cast<aclrtStream>(input->GetDeviceContext()->device_res_manager_->GetStream(input->GetStreamID())));

    if (!input->GetDeviceContext()->device_res_manager_->SyncStream(input->GetStreamID())) {
      MS_LOG(ERROR) << "SyncStream stream id: " << std::to_string(input->GetStreamID()) << " failed.";
      return APP_ERR_DVPP_ADJUST_HUE_FAIL;
    }

    // release workspace_addr
    (void)input->GetDeviceContext()->device_res_manager_->FreeMemory(workspace_addr);
  } else {
    // call DVPP step3
    ret = acldvppAdjustHue(
      nullptr, workspace_size, executor,
      static_cast<aclrtStream>(input->GetDeviceContext()->device_res_manager_->GetStream(input->GetStreamID())));
  }

  if (ret != ACL_SUCCESS) {
    MS_LOG(ERROR) << "Call acldvppAdjustHue failed, error code: " + std::to_string(ret) + ".";
    return APP_ERR_DVPP_ADJUST_HUE_FAIL;
  }

  *output = std::move(device_tensor);  // currently the data is still in device
  return APP_ERR_OK;
}

APP_ERROR DvppAdjustSaturation(const std::shared_ptr<DeviceTensorAscend910B> &input,
                               std::shared_ptr<DeviceTensorAscend910B> *output, float factor) {
  MS_LOG(WARNING) << "Begin execute adjust saturation.";
  if (input == nullptr || output == nullptr) {
    MS_LOG(ERROR) << "The input or output is nullptr.";
    return APP_ERR_DVPP_ADJUST_SATURATION_FAIL;
  }

  // the input should be 1HWC or 1CHW
  if (input->GetShape().Rank() != kNHWCImageRank) {
    MS_LOG(ERROR) << "The input data's dims is not 4.";  // NHWC
    return APP_ERR_DVPP_ADJUST_SATURATION_FAIL;
  }

  // the channel should be equal to 3 or 1
  if (input->GetShape().AsVector()[3] != 3 && input->GetShape().AsVector()[3] != 1) {
    MS_LOG(ERROR) << "The input data's channel is not 3 or 1.";
    return APP_ERR_DVPP_ADJUST_SATURATION_FAIL;
  }

  if (input->GetShape().AsVector()[0] != 1) {
    MS_LOG(ERROR) << "The input data is not 1HWC or 1CHW.";  // N == 1
    return APP_ERR_DVPP_ADJUST_SATURATION_FAIL;
  }

  // the type is uint8 / float
  if (input->GetType() != DataType::DE_UINT8 && input->GetType() != DataType::DE_FLOAT32) {
    MS_LOG(ERROR) << "The input data is not uint8 or float32";
    return APP_ERR_DVPP_ADJUST_SATURATION_FAIL;
  }

  // create the output shape and type, it's 1HWC or 1CHW
  TensorShape shape = input->GetShape();
  DataType type = input->GetType();

  // create output DeviceTensorAscend910B
  std::shared_ptr<DeviceTensorAscend910B> device_tensor = nullptr;
  if (DeviceTensorAscend910B::CreateDeviceTensor(shape, type, input->GetDeviceContext(), input->GetStreamID(),
                                                 &device_tensor, true) != Status::OK()) {
    MS_LOG(ERROR) << "Create output device tensor failed.";
    return APP_ERR_DVPP_ADJUST_SATURATION_FAIL;
  }

  // call DVPP step1
  uint64_t workspace_size = 0;
  aclOpExecutor *executor;
  auto ret = acldvppAdjustSaturationGetWorkspaceSize(reinterpret_cast<aclTensor *>(input->GetDeviceTensor()), factor,
                                                     reinterpret_cast<aclTensor *>(device_tensor->GetDeviceTensor()),
                                                     &workspace_size, &executor);
  if (ret != ACL_SUCCESS) {
    MS_LOG(ERROR) << "Call acldvppAdjustSaturationGetWorkspaceSize failed, error code: " + std::to_string(ret) + ".";
    return APP_ERR_DVPP_ADJUST_SATURATION_FAIL;
  }

  // call DVPP step2
  void *workspace_addr = nullptr;
  if (workspace_size > 0) {
    // create new device address for data copy
    workspace_addr = input->GetDeviceContext()->device_res_manager_->AllocateMemory(workspace_size);
    if (workspace_addr == nullptr) {
      MS_LOG(ERROR) << "Allocate dynamic workspace memory failed";
      return APP_ERR_DVPP_ADJUST_SATURATION_FAIL;
    }

    // call DVPP step3
    ret = acldvppAdjustSaturation(
      workspace_addr, workspace_size, executor,
      static_cast<aclrtStream>(input->GetDeviceContext()->device_res_manager_->GetStream(input->GetStreamID())));

    if (!input->GetDeviceContext()->device_res_manager_->SyncStream(input->GetStreamID())) {
      MS_LOG(ERROR) << "SyncStream stream id: " << std::to_string(input->GetStreamID()) << " failed.";
      return APP_ERR_DVPP_ADJUST_SATURATION_FAIL;
    }

    // release workspace_addr
    (void)input->GetDeviceContext()->device_res_manager_->FreeMemory(workspace_addr);
  } else {
    // call DVPP step3
    ret = acldvppAdjustSaturation(
      nullptr, workspace_size, executor,
      static_cast<aclrtStream>(input->GetDeviceContext()->device_res_manager_->GetStream(input->GetStreamID())));
  }

  if (ret != ACL_SUCCESS) {
    MS_LOG(ERROR) << "Call acldvppAdjustSaturation failed, error code: " + std::to_string(ret) + ".";
    return APP_ERR_DVPP_ADJUST_SATURATION_FAIL;
  }

  *output = std::move(device_tensor);  // currently the data is still in device
  return APP_ERR_OK;
}

APP_ERROR GetSocName(std::string *soc_name) {
  const char *soc_name_c = aclrtGetSocName();
  if (soc_name_c == nullptr) {
    *soc_name = "";
  }
  *soc_name = std::string(soc_name_c);
  return APP_ERR_OK;
}

APP_ERROR CreateAclTensor(const int64_t *view_dims, uint64_t view_dims_num, mindspore::TypeId data_type,
                          const int64_t *stride, int64_t offset, const int64_t *storage_dims, uint64_t storage_dims_num,
                          void *tensor_data, bool is_hwc, void **acl_tensor) {
  if (view_dims == nullptr) {
    MS_LOG(ERROR) << "Input view_dims is null.";
    return APP_ERR_COMM_FAILURE;
  }
  if (stride == nullptr) {
    MS_LOG(ERROR) << "Input stride is null.";
    return APP_ERR_COMM_FAILURE;
  }
  if (storage_dims == nullptr) {
    MS_LOG(ERROR) << "Input storage_dims is null.";
    return APP_ERR_COMM_FAILURE;
  }
  if (tensor_data == nullptr) {
    MS_LOG(ERROR) << "Input tensor_data is null.";
    return APP_ERR_COMM_FAILURE;
  }
  if (acl_tensor == nullptr) {
    MS_LOG(ERROR) << "Input acl_tensor is null.";
    return APP_ERR_COMM_FAILURE;
  }

  aclDataType acl_data_type = aclDataType::ACL_DT_UNDEFINED;

  switch (data_type) {
    case kNumberTypeBool:
      acl_data_type = aclDataType::ACL_BOOL;
      break;
    case kNumberTypeInt8:
      acl_data_type = aclDataType::ACL_INT8;
      break;
    case kNumberTypeUInt8:
      acl_data_type = aclDataType::ACL_UINT8;
      break;
    case kNumberTypeInt16:
      acl_data_type = aclDataType::ACL_INT16;
      break;
    case kNumberTypeUInt16:
      acl_data_type = aclDataType::ACL_UINT16;
      break;
    case kNumberTypeInt32:
      acl_data_type = aclDataType::ACL_INT32;
      break;
    case kNumberTypeUInt32:
      acl_data_type = aclDataType::ACL_UINT32;
      break;
    case kNumberTypeInt64:
      acl_data_type = aclDataType::ACL_INT64;
      break;
    case kNumberTypeUInt64:
      acl_data_type = aclDataType::ACL_UINT64;
      break;
    case kNumberTypeFloat16:
      acl_data_type = aclDataType::ACL_FLOAT16;
      break;
    case kNumberTypeFloat32:
      acl_data_type = aclDataType::ACL_FLOAT;
      break;
    case kNumberTypeFloat64:
      acl_data_type = aclDataType::ACL_DOUBLE;
      break;
    case kObjectTypeString:
      acl_data_type = aclDataType::ACL_STRING;
      break;
    default:
      acl_data_type = aclDataType::ACL_DT_UNDEFINED;
      break;
  }

  if (acl_data_type == aclDataType::ACL_DT_UNDEFINED) {
    MS_LOG(ERROR) << "Invalid data type: " << data_type << ", which couldn't be converted to aclDataType.";
    return APP_ERR_COMM_FAILURE;
  }

  if (is_hwc) {
    *acl_tensor = reinterpret_cast<void *>(aclCreateTensor(view_dims, view_dims_num, acl_data_type, stride, offset,
                                                           aclFormat::ACL_FORMAT_NHWC, storage_dims, storage_dims_num,
                                                           tensor_data));
  } else {
    *acl_tensor = reinterpret_cast<void *>(aclCreateTensor(view_dims, view_dims_num, acl_data_type, stride, offset,
                                                           aclFormat::ACL_FORMAT_NCHW, storage_dims, storage_dims_num,
                                                           tensor_data));
  }
  return APP_ERR_OK;
}
}  // namespace dataset
}  // namespace mindspore
