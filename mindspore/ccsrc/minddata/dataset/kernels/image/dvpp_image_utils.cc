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
#include "minddata/dataset/kernels/image/dvpp_image_utils.h"

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

#include "acldvppop/acldvpp_resize.h"
#include "plugin/device/ascend/hal/device/ascend_stream_manager.h"

namespace mindspore {
namespace dataset {

const int kInvalidInterpolationMode = 100;

inline int GetDVPPInterpolationMode(InterpolationMode mode) {
  switch (mode) {
    case InterpolationMode::kLinear:
      return 0;  // dvpp BILINEAR
    case InterpolationMode::kCubic:
      return 2;  // dvpp BICUBIC
    case InterpolationMode::kArea:
      return kInvalidInterpolationMode;
    case InterpolationMode::kNearestNeighbour:
      return 1;  // dvpp NEAREST
    default:
      return kInvalidInterpolationMode;
  }
}

Status DvppResize(const std::shared_ptr<DeviceTensorAscend910B> &input, std::shared_ptr<DeviceTensorAscend910B> *output,
                  int32_t output_height, int32_t output_width, double fx, double fy, InterpolationMode mode) {
  IO_CHECK(input, output);
  // the input should be reconstruct NHWC in pre steps
  CHECK_FAIL_RETURN_UNEXPECTED(input->GetShape().Rank() == 4, "The input data's dims is not 3.");  // NHWC
  CHECK_FAIL_RETURN_UNEXPECTED(input->GetShape().AsVector()[3] == 3 || input->GetShape().AsVector()[3] == 1,
                               "The input data's channel is not 3 or 1.");  // C == 3 or 1

  const uint32_t kResizeShapeLimits = 1000;
  // resize image too large or too small, 1000 is arbitrarily chosen here to prevent open cv from segmentation fault
  CHECK_FAIL_RETURN_UNEXPECTED((std::numeric_limits<int>::max() / kResizeShapeLimits) > input->GetShape().AsVector()[1],
                               "DvppResize: in_image rows out of bounds.");
  CHECK_FAIL_RETURN_UNEXPECTED((std::numeric_limits<int>::max() / kResizeShapeLimits) > input->GetShape().AsVector()[2],
                               "DvppResize: in_image cols out of bounds.");
  if (output_height > input->GetShape().AsVector()[1] * kResizeShapeLimits ||
      output_width > input->GetShape().AsVector()[2] * kResizeShapeLimits) {
    std::string err_msg =
      "DvppResize: the resizing width or height is too big, it's 1000 times bigger than the original image, got output "
      "height: " +
      std::to_string(output_height) + ", width: " + std::to_string(output_width) +
      ", and original image size:" + std::to_string(input->GetShape().AsVector()[1]) + ", " +
      std::to_string(input->GetShape().AsVector()[2]);
    return Status(StatusCode::kMDShapeMisMatch, err_msg);
  }
  if (output_height == 0 || output_width == 0) {
    std::string err_msg = "DvppResize: the input value of 'resize' is invalid, width or height is zero.";
    return Status(StatusCode::kMDShapeMisMatch, err_msg);
  }

  try {
    // create the output shape and type, it's NHWC
    TensorShape shape{static_cast<int>(input->GetShape()[0]), output_height, output_width,
                      static_cast<int>(input->GetShape()[kChannelIndexHWC + 1])};
    DataType type = input->GetType();

    // create output DeviceTensorAscend910B
    std::shared_ptr<DeviceTensorAscend910B> device_tensor = nullptr;
    RETURN_IF_NOT_OK(DeviceTensorAscend910B::CreateDeviceTensor(shape, type, input->GetDeviceContext(),
                                                                input->GetStreamID(), &device_tensor));

    // convert InterpolationMode mode to DVPP mode
    auto dvpp_interpolation_mode = GetDVPPInterpolationMode(mode);
    if (dvpp_interpolation_mode == kInvalidInterpolationMode) {
      RETURN_STATUS_UNEXPECTED("The InterpolationMode is not supported by DVPP. It is " +
                               std::to_string(static_cast<int>(mode)));
    }

    // call DVPP step1
    uint64_t workspace_size = 0;
    aclOpExecutor *executor;
    auto ret = acldvppResizeGetWorkspaceSize(input->GetDeviceTensor(), dvpp_interpolation_mode,
                                             device_tensor->GetDeviceTensor(), &workspace_size, &executor);
    CHECK_FAIL_RETURN_UNEXPECTED(ret == ACL_SUCCESS,
                                 "Call acldvppResizeGetWorkspaceSize failed, error code: " + std::to_string(ret) + ".");

    // call DVPP step2
    device::DeviceAddressPtr workspace_addr = nullptr;
    if (workspace_size > 0) {
      // create new device address for data copy
      workspace_addr = input->GetDeviceContext()->device_res_manager_->CreateDeviceAddress(
        nullptr, workspace_size, "", DETypeToMSType(type), shape.AsVector());
      CHECK_FAIL_RETURN_UNEXPECTED(workspace_addr->GetPtr(), "Call CreateDeviceAddress failed.");
      CHECK_FAIL_RETURN_UNEXPECTED(input->GetDeviceContext()->device_res_manager_->AllocateMemory(workspace_addr.get()),
                                   "Allocate dynamic workspace memory failed.");

      // call DVPP step3
      ret = acldvppResize(workspace_addr->GetMutablePtr(), workspace_size, executor,
                          (aclrtStream)input->GetDeviceContext()->device_res_manager_->GetStream(input->GetStreamID()));
    } else {
      // call DVPP step3
      ret = acldvppResize(nullptr, workspace_size, executor,
                          (aclrtStream)input->GetDeviceContext()->device_res_manager_->GetStream(input->GetStreamID()));
    }

    CHECK_FAIL_RETURN_UNEXPECTED(ret == ACL_SUCCESS,
                                 "Call aclvisionResize failed, error code: " + std::to_string(ret) + ".");

    *output = std::move(device_tensor);  // currently the data is still in device
    return Status::OK();
  } catch (const cv::Exception &e) {
    RETURN_STATUS_UNEXPECTED("DvppResize: " + std::string(e.what()));
  }
}
}  // namespace dataset
}  // namespace mindspore
