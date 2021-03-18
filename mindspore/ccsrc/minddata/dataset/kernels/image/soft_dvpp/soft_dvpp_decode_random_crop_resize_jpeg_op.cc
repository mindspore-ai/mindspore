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
#include "minddata/dataset/kernels/image/soft_dvpp/soft_dvpp_decode_random_crop_resize_jpeg_op.h"
#include <string>

#include "opencv2/opencv.hpp"

#include "minddata/dataset/core/cv_tensor.h"
#include "minddata/dataset/kernels/image/image_utils.h"
#include "minddata/dataset/util/random.h"

namespace mindspore {
namespace dataset {
SoftDvppDecodeRandomCropResizeJpegOp::SoftDvppDecodeRandomCropResizeJpegOp(int32_t target_height, int32_t target_width,
                                                                           float scale_lb, float scale_ub,
                                                                           float aspect_lb, float aspect_ub,
                                                                           int32_t max_iter)
    : RandomCropAndResizeOp(target_height, target_width, scale_lb, scale_ub, aspect_lb, aspect_ub,
                            InterpolationMode::kLinear, max_iter) {}

Status SoftDvppDecodeRandomCropResizeJpegOp::GetCropInfo(const std::shared_ptr<Tensor> &input,
                                                         SoftDpCropInfo *crop_info) {
  int img_width = 0;
  int img_height = 0;
  RETURN_IF_NOT_OK(GetJpegImageInfo(input, &img_width, &img_height));
  int x = 0;
  int y = 0;
  int crop_heigh = 0;
  int crop_widht = 0;
  RETURN_IF_NOT_OK(GetCropBox(img_height, img_width, &x, &y, &crop_heigh, &crop_widht));
  crop_info->left = x;
  crop_info->up = y;
  crop_info->right = crop_info->left + crop_widht - 1;
  crop_info->down = crop_info->up + crop_heigh - 1;
  return Status::OK();
}

Status SoftDvppDecodeRandomCropResizeJpegOp::Compute(const std::shared_ptr<Tensor> &input,
                                                     std::shared_ptr<Tensor> *output) {
  IO_CHECK(input, output);
  if (!IsNonEmptyJPEG(input)) {
    RETURN_STATUS_UNEXPECTED("SoftDvppDecodeRandomCropResizeJpeg: only support processing raw jpeg image.");
  }
  SoftDpCropInfo crop_info;
  RETURN_IF_NOT_OK(GetCropInfo(input, &crop_info));
  try {
    unsigned char *buffer = const_cast<unsigned char *>(input->GetBuffer());
    CHECK_FAIL_RETURN_UNEXPECTED(buffer != nullptr,
                                 "SoftDvppDecodeRandomCropResizeJpeg: the input image buffer is empty.");
    SoftDpProcsessInfo info;
    info.input_buffer = static_cast<uint8_t *>(buffer);
    info.input_buffer_size = input->SizeInBytes();
    info.output_width = target_width_;
    info.output_height = target_height_;
    cv::Mat out_rgb_img(target_height_, target_width_, CV_8UC3);
    info.output_buffer = out_rgb_img.data;
    info.output_buffer_size = target_width_ * target_height_ * 3;
    info.is_v_before_u = true;
    int ret = DecodeAndCropAndResizeJpeg(&info, crop_info);
    std::string error_info("SoftDvppDecodeRandomCropResizeJpeg: failed with return code: ");
    error_info += std::to_string(ret) + ", please check the log information for more details.";
    CHECK_FAIL_RETURN_UNEXPECTED(ret == 0, error_info);
    std::shared_ptr<CVTensor> cv_tensor = nullptr;
    RETURN_IF_NOT_OK(CVTensor::CreateFromMat(out_rgb_img, &cv_tensor));
    *output = std::static_pointer_cast<Tensor>(cv_tensor);
  } catch (const cv::Exception &e) {
    std::string error = "SoftDvppDecodeRandomCropResizeJpeg:" + std::string(e.what());
    RETURN_STATUS_UNEXPECTED(error);
  }
  return Status::OK();
}
}  // namespace dataset
}  // namespace mindspore
