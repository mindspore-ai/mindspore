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
#include "minddata/dataset/kernels/image/soft_dvpp/soft_dvpp_decode_resize_jpeg_op.h"
#include <string>
#include <vector>

#include "./utils/external_soft_dp.h"
#include "opencv2/opencv.hpp"

#include "minddata/dataset/core/cv_tensor.h"
#include "minddata/dataset/kernels/image/image_utils.h"

namespace mindspore {
namespace dataset {
Status SoftDvppDecodeResizeJpegOp::Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) {
  IO_CHECK(input, output);
  if (!IsNonEmptyJPEG(input)) {
    RETURN_STATUS_UNEXPECTED("SoftDvppDecodeReiszeJpeg: only support processing raw jpeg image.");
  }
  try {
    unsigned char *buffer = const_cast<unsigned char *>(input->GetBuffer());
    CHECK_FAIL_RETURN_UNEXPECTED(buffer != nullptr, "SoftDvppDecodeReiszeJpeg: the input image buffer is empty.");
    SoftDpProcsessInfo info;
    info.input_buffer = static_cast<uint8_t *>(buffer);
    info.input_buffer_size = input->SizeInBytes();

    int input_w = 0;
    int input_h = 0;
    RETURN_IF_NOT_OK(GetJpegImageInfo(input, &input_w, &input_h));

    if (target_width_ == 0) {
      if (input_h < input_w) {
        CHECK_FAIL_RETURN_UNEXPECTED(input_h != 0, "SoftDvppDecodeReiszeJpeg: the input height is 0.");
        info.output_height = target_height_;
        info.output_width = static_cast<int>(std::lround(static_cast<float>(input_w) / input_h * info.output_height));
      } else {
        CHECK_FAIL_RETURN_UNEXPECTED(input_w != 0, "SoftDvppDecodeReiszeJpeg: the input width is 0.");
        info.output_width = target_height_;
        info.output_height = static_cast<int>(std::lround(static_cast<float>(input_h) / input_w * info.output_width));
      }
    } else {
      info.output_height = target_height_;
      info.output_width = target_width_;
    }

    cv::Mat out_rgb_img(info.output_height, info.output_width, CV_8UC3);
    info.output_buffer = out_rgb_img.data;
    info.output_buffer_size = info.output_height * info.output_width * 3;

    info.is_v_before_u = true;
    int ret = DecodeAndResizeJpeg(&info);
    std::string error_info("SoftDvppDecodeReiszeJpeg: failed with return code: ");
    error_info += std::to_string(ret) + ", please check the log information for more details.";
    CHECK_FAIL_RETURN_UNEXPECTED(ret == 0, error_info);
    std::shared_ptr<CVTensor> cv_tensor = nullptr;
    RETURN_IF_NOT_OK(CVTensor::CreateFromMat(out_rgb_img, &cv_tensor));
    *output = std::static_pointer_cast<Tensor>(cv_tensor);
  } catch (const cv::Exception &e) {
    std::string error = "SoftDvppDecodeResizeJpeg:" + std::string(e.what());
    RETURN_STATUS_UNEXPECTED(error);
  }
  return Status::OK();
}

Status SoftDvppDecodeResizeJpegOp::OutputShape(const std::vector<TensorShape> &inputs,
                                               std::vector<TensorShape> &outputs) {
  RETURN_IF_NOT_OK(TensorOp::OutputShape(inputs, outputs));
  outputs.clear();
  TensorShape out({-1, -1, 3});  // we don't know what is output image size, but we know it should be 3 channels
  if (inputs[0].Rank() == 1) outputs.emplace_back(out);
  if (!outputs.empty()) return Status::OK();
  return Status(StatusCode::kMDUnexpectedError, "SoftDvppDecodeReiszeJpeg: input has a wrong shape.");
}

}  // namespace dataset
}  // namespace mindspore
