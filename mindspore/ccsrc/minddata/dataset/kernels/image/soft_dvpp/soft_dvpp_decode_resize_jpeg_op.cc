/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
#include "minddata/dataset/kernels/image/soft_dvpp/soft_dvpp_decode_resize_jpeg_op.h"
#include <string>

#include "./utils/external_soft_dp.h"
#include "opencv2/opencv.hpp"

#include "minddata/dataset/core/cv_tensor.h"
#include "minddata/dataset/kernels/image/image_utils.h"

namespace mindspore {
namespace dataset {
Status SoftDvppDecodeResizeJpegOp::Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) {
  IO_CHECK(input, output);
  if (!IsNonEmptyJPEG(input)) {
    RETURN_STATUS_UNEXPECTED("SoftDvppDecodeReiszeJpegOp only support process jpeg image.");
  }
  try {
    unsigned char *buffer = const_cast<unsigned char *>(input->GetBuffer());
    CHECK_FAIL_RETURN_UNEXPECTED(buffer != nullptr, "The input image buffer is empty.");
    SoftDpProcsessInfo info;
    info.input_buffer = static_cast<uint8_t *>(buffer);
    info.input_buffer_size = input->SizeInBytes();
    info.output_width = target_width_;
    info.output_height = target_height_;

    SoftDpCropInfo crop_info{0, 0, 0, 0};

    cv::Mat out_rgb_img(target_height_, target_width_, CV_8UC3);
    info.output_buffer = out_rgb_img.data;
    info.output_buffer_size = target_width_ * target_height_ * 3;
    info.is_v_before_u = true;
    int ret = DecodeAndResizeJpeg(&info);
    std::string error_info("Soft dvpp DecodeAndResizeJpeg failed with return code: ");
    error_info += std::to_string(ret);
    CHECK_FAIL_RETURN_UNEXPECTED(ret == 0, error_info);
    std::shared_ptr<CVTensor> cv_tensor = nullptr;
    RETURN_IF_NOT_OK(CVTensor::CreateFromMat(out_rgb_img, &cv_tensor));
    *output = std::static_pointer_cast<Tensor>(cv_tensor);
  } catch (const cv::Exception &e) {
    RETURN_STATUS_UNEXPECTED("Error in soft dvpp image decode and resize.");
  }
  return Status::OK();
}
}  // namespace dataset
}  // namespace mindspore
