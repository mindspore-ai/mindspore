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
#include "minddata/dataset/kernels/image/random_crop_decode_resize_op.h"
#include <random>
#include "minddata/dataset/kernels/image/image_utils.h"
#include "minddata/dataset/core/config_manager.h"
#include "minddata/dataset/kernels/image/decode_op.h"

namespace mindspore {
namespace dataset {
RandomCropDecodeResizeOp::RandomCropDecodeResizeOp(int32_t target_height, int32_t target_width, float scale_lb,
                                                   float scale_ub, float aspect_lb, float aspect_ub,
                                                   InterpolationMode interpolation, int32_t max_attempts)
    : RandomCropAndResizeOp(target_height, target_width, scale_lb, scale_ub, aspect_lb, aspect_ub, interpolation,
                            max_attempts) {}

Status RandomCropDecodeResizeOp::Compute(const TensorRow &input, TensorRow *output) {
  IO_CHECK_VECTOR(input, output);
  const int output_count = input.size();
  output->resize(output_count);
  int x = 0;
  int y = 0;
  int crop_height = 0;
  int crop_width = 0;
  TensorRow decoded;
  decoded.resize(output_count);
  for (size_t i = 0; i < input.size(); i++) {
    if (input[i] == nullptr) {
      RETURN_STATUS_UNEXPECTED("RandomCropDecodeResize: input image is empty since got nullptr.");
    }
    if (!IsNonEmptyJPEG(input[i])) {
      DecodeOp op(true);
      RETURN_IF_NOT_OK(op.Compute(input[i], &decoded[i]));
      RETURN_IF_NOT_OK(RandomCropAndResizeOp::Compute(decoded, output));
    } else {
      int h_in = 0;
      int w_in = 0;
      RETURN_IF_NOT_OK(GetJpegImageInfo(input[i], &w_in, &h_in));
      if (i == 0) {
        RETURN_IF_NOT_OK(GetCropBox(h_in, w_in, &x, &y, &crop_height, &crop_width));
      }
      std::shared_ptr<Tensor> decoded_tensor = nullptr;
      RETURN_IF_NOT_OK(JpegCropAndDecode(input[i], &decoded_tensor, x, y, crop_width, crop_height));
      RETURN_IF_NOT_OK(Resize(decoded_tensor, &(*output)[i], target_height_, target_width_, 0.0, 0.0, interpolation_));
    }
  }
  return Status::OK();
}
}  // namespace dataset
}  // namespace mindspore
