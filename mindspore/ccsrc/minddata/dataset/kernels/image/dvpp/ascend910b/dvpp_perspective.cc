/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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

#include "minddata/dataset/kernels/image/dvpp/ascend910b/dvpp_perspective.h"

#ifndef ENABLE_ANDROID
#include "minddata/dataset/kernels/image/image_utils.h"
#include "minddata/dataset/kernels/image/dvpp/utils/dvpp_image_utils.h"
#include "minddata/dataset/kernels/image/dvpp/acl_adapter.h"
#include "minddata/dataset/kernels/image/dvpp/utils/ErrorCode.h"
#else
#include "minddata/dataset/kernels/image/lite_image_utils.h"
#endif
#include "minddata/dataset/util/status.h"

namespace mindspore {
namespace dataset {
Status DvppPerspectiveOp::Compute(const std::shared_ptr<DeviceTensorAscend910B> &input,
                                  std::shared_ptr<DeviceTensorAscend910B> *output) {
  IO_CHECK(input, output);
  // check the input tensor shape
  const auto kNHWCImageRank = 4;
  if (input->GetShape().Rank() != kNHWCImageRank) {
    RETURN_STATUS_UNEXPECTED("DvppPerspective: invalid input shape, only support NHWC input, got rank: " +
                             std::to_string(input->GetShape().Rank()));
  }

  // the channel should be 3 or 1
  const auto kChannelIndexNHWC = 3;
  CHECK_FAIL_RETURN_UNEXPECTED(input->GetShape().AsVector()[kChannelIndexNHWC] == kMinImageChannel ||
                                 input->GetShape().AsVector()[kChannelIndexNHWC] == kDefaultImageChannel,
                               "DvppPerspective: the channel of the input is not 1 or 3.");

  // the type should be uint8 or float
  CHECK_FAIL_RETURN_UNEXPECTED(input->GetType() == DataType::DE_UINT8 || input->GetType() == DataType::DE_FLOAT32,
                               "DvppPerspective: the type of the input is not uint8 or float.");

  APP_ERROR ret = AclAdapter::GetInstance().DvppPerspective(input, output, start_points_, end_points_, interpolation_);
  if (ret != APP_ERR_OK) {
    std::string error = "DvppPerspective: Error in dvpp processing: " + std::to_string(ret);
    RETURN_STATUS_UNEXPECTED(error);
  }

  return Status::OK();
}
}  // namespace dataset
}  // namespace mindspore
