/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_KERNELS_IMAGE_IMAGE_UTILS_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_KERNELS_IMAGE_IMAGE_UTILS_H_

#include <setjmp.h>

#include <memory>
#include <random>
#include <string>
#include <vector>
#if defined(_WIN32) || defined(_WIN64)
#undef HAVE_STDDEF_H
#undef HAVE_STDLIB_H
#endif
#include "./jpeglib.h"
#include "./jerror.h"
#include "minddata/dataset/core/tensor.h"
#include "minddata/dataset/kernels/tensor_op.h"
#include "minddata/dataset/kernels/image/lite_cv/image_process.h"
#include "minddata/dataset/util/status.h"

namespace mindspore {
namespace dataset {
void JpegErrorExitCustom(j_common_ptr cinfo);

struct JpegErrorManagerCustom {
  // "public" fields
  struct jpeg_error_mgr pub;
  // for return to caller
  jmp_buf setjmp_buffer;
};

bool IsNonEmptyJPEG(const std::shared_ptr<Tensor> &input);

void JpegSetSource(j_decompress_ptr c_info, const void *data, int64_t data_size);

Status JpegCropAndDecode(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output, int x = 0, int y = 0,
                         int w = 0, int h = 0);

/// \brief Returns cropped ROI of an image
/// \param input: Tensor of shape <H,W,C> or <H,W> and any OpenCv compatible type, see CVTensor.
/// \param x: starting horizontal position of ROI
/// \param y: starting vertical position of ROI
/// \param w: width of the ROI
/// \param h: height of the ROI
/// \param output: Cropped image Tensor of shape <h,w,C> or <h,w> and same input type.
Status Crop(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output, int x, int y, int w, int h);

/// \brief Returns Decoded image
/// Supported images:
///  BMP JPEG JPG PNG TIFF
/// supported by opencv, if user need more image analysis capabilities, please compile opencv particularlly.
/// \param input: CVTensor containing the not decoded image 1D bytes
/// \param output: Decoded image Tensor of shape <H,W,C> and type DE_UINT8. Pixel order is RGB
Status Decode(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output);

/// \brief Get jpeg image width and height
/// \param input: CVTensor containing the not decoded image 1D bytes
/// \param img_width: the jpeg image width
/// \param img_height: the jpeg image height
Status GetJpegImageInfo(const std::shared_ptr<Tensor> &input, int *img_width, int *img_height);

/// \brief Returns Normalized image
/// \param input: Tensor of shape <H,W,C> in RGB order and any OpenCv compatible type, see CVTensor.
/// \param mean: Tensor of shape <3> and type DE_FLOAT32 which are mean of each channel in RGB order
/// \param std:  Tensor of shape <3> and type DE_FLOAT32 which are std of each channel in RGB order
/// \param output: Normalized image Tensor of same input shape and type DE_FLOAT32
Status Normalize(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output,
                 const std::shared_ptr<Tensor> &mean, const std::shared_ptr<Tensor> &std);

/// \brief  Returns Resized image.
/// \param input/output: Tensor of shape <H,W,C> or <H,W> and any OpenCv compatible type, see CVTensor.
/// \param output_height: height of output
/// \param output_width: width of output
/// \param fx: horizontal scale
/// \param fy: vertical scale
/// \param InterpolationMode: the interpolation mode
/// \param output: Resized image of shape <outputHeight,outputWidth,C> or <outputHeight,outputWidth>
///                and same type as input
Status Resize(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output, int32_t output_height,
              int32_t output_width, double fx = 0.0, double fy = 0.0,
              InterpolationMode mode = InterpolationMode::kLinear);

/// \brief Pads the input image and puts the padded image in the output
/// \param input: input Tensor
/// \param output: padded Tensor
/// \param pad_top: amount of padding done in top
/// \param pad_bottom: amount of padding done in bottom
/// \param pad_left: amount of padding done in left
/// \param pad_right: amount of padding done in right
/// \param border_types: the interpolation to be done in the border
/// \param fill_r: red fill value for pad
/// \param fill_g: green fill value for pad
/// \param fill_b: blue fill value for pad.
Status Pad(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output, const int32_t &pad_top,
           const int32_t &pad_bottom, const int32_t &pad_left, const int32_t &pad_right, const BorderType &border_types,
           uint8_t fill_r = 0, uint8_t fill_g = 0, uint8_t fill_b = 0);

/// \brief Rotate the input image by orientation
/// \param input: input Tensor
/// \param output: padded Tensor
/// \param orientation:  the orientation of EXIF
Status Rotate(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output, const uint64_t orientation);

}  // namespace dataset
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_KERNELS_IMAGE_IMAGE_UTILS_H_
