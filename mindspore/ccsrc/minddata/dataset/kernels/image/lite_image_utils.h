/**
 * Copyright 2019-2021 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_KERNELS_IMAGE_LITE_IMAGE_UTILS_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_KERNELS_IMAGE_LITE_IMAGE_UTILS_H_

#include <csetjmp>
#include <memory>
#include <random>
#include <set>
#include <string>
#include <vector>

#if defined(_WIN32) || defined(_WIN64)
#undef HAVE_STDDEF_H
#undef HAVE_STDLIB_H
#endif

#include "./jpeglib.h"
#include "./jerror.h"
#include "minddata/dataset/core/tensor.h"
#include "minddata/dataset/kernels/image/lite_cv/image_process.h"
#include "minddata/dataset/kernels/tensor_op.h"
#include "minddata/dataset/util/status.h"
#include "minddata/dataset/util/validators.h"

namespace mindspore {
namespace dataset {
constexpr dsize_t kChannelIndexHWC = 2;      // images are hwc, so index 2 represents number of channels
constexpr dsize_t kChannelIndexCHW = 0;      // images are chw, so index 0 represents number of channels
constexpr int32_t kMaxBitValue = 255;        // max bit value after decode is 256
constexpr dsize_t kMinImageChannel = 1;      // image ops support minimum of 1 channel
constexpr dsize_t kDefaultImageChannel = 3;  // images are 3 channels in general
constexpr dsize_t kMaxImageChannel = 4;      // image ops support maximum of 4 channel
constexpr float kHalf = 0.5;                 // to get the half of a value
constexpr dsize_t kMinJpegQuality = 1;       // the minimum quality for JPEG
constexpr dsize_t kMaxJpegQuality = 100;     // the maximum quality for JPEG
constexpr dsize_t kDefaultImageRank = 3;     // images are hwc channels in general
constexpr dsize_t kMinImageRank = 2;         // images have at least 2 dimensions
constexpr int32_t kMaxPixelValue = 255;
constexpr dsize_t kHeightIndex = 0;  // index of height of HWC images
constexpr dsize_t kWidthIndex = 1;   // index of width of HWC images
constexpr dsize_t kRIndex = 0;       // index of red channel in RGB format
constexpr dsize_t kGIndex = 1;       // index of green channel in RGB format
constexpr dsize_t kBIndex = 2;       // index of blue channel in RGB format

void JpegErrorExitCustom(j_common_ptr cinfo);

struct JpegErrorManagerCustom {
  // "public" fields
  struct jpeg_error_mgr pub;
  // for return to caller
  jmp_buf setjmp_buffer;
};

#if defined(ENABLE_CLOUD_FUSION_INFERENCE)
/// \brief Returns Rescaled image
/// \param input: Tensor of shape <H,W,C> or <H,W> and any OpenCv compatible type, see CVTensor.
/// \param rescale: rescale parameter
/// \param shift: shift parameter
/// \param output: Rescaled image Tensor of same input shape and type DE_FLOAT32
Status Rescale(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output, float rescale, float shift);

/// \brief Swap the red and blue pixels (RGB <-> BGR)
/// \param input: Tensor of shape <H,W,3> and any OpenCv compatible type, see CVTensor.
/// \param output: Swapped image of same shape and type
Status SwapRedAndBlue(std::shared_ptr<Tensor> input, std::shared_ptr<Tensor> *output);
#endif

bool IsNonEmptyJPEG(const std::shared_ptr<Tensor> &input);

void JpegSetSource(j_decompress_ptr c_info, const void *data, int64_t data_size);

Status JpegCropAndDecode(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output, int x = 0, int y = 0,
                         int w = 0, int h = 0);

/// \brief Returns cropped ROI of an image
/// \param[in] input: Tensor of shape <H,W,C> or <H,W> and any OpenCv compatible type, see CVTensor.
/// \param[in] x Starting horizontal position of ROI
/// \param[in] y Starting vertical position of ROI
/// \param[in] w Width of the ROI
/// \param[in] h Height of the ROI
/// \param[out] output: Cropped image Tensor of shape <h,w,C> or <h,w> and same input type.
Status Crop(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output, int x, int y, int w, int h);

/// \brief Returns Decoded image
/// Supported images:
///  BMP JPEG JPG PNG TIFF
/// supported by opencv, if user need more image analysis capabilities, please compile opencv particularlly.
/// \param[in] input CVTensor containing the not decoded image 1D bytes
/// \param[out] output Decoded image Tensor of shape <H,W,C> and type DE_UINT8. Pixel order is RGB
Status Decode(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output);

/// \brief Get jpeg image width and height
/// \param[in] input CVTensor containing the not decoded image 1D bytes
/// \param[in] img_width The jpeg image width
/// \param[in] img_height The jpeg image height
Status GetJpegImageInfo(const std::shared_ptr<Tensor> &input, int *img_width, int *img_height);

/// \brief Returns Normalized image
/// \param[in] input Tensor of shape <H,W,C> in RGB order and any OpenCv compatible type, see CVTensor.
/// \param[in] mean Tensor of shape <3> and type DE_FLOAT32 which are mean of each channel in RGB order
/// \param[in] std  Tensor of shape <3> and type DE_FLOAT32 which are std of each channel in RGB order
/// \param[out] output Normalized image Tensor of same input shape and type DE_FLOAT32
Status Normalize(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output,
                 const std::vector<float> &vec_mean, const std::vector<float> &vec_std);

/// \brief  Returns Resized image.
/// \param[in] input
/// \param[in] output_height Height of output
/// \param[in] output_width Width of output
/// \param[in] fx Horizontal scale
/// \param[in] fy Vertical scale
/// \param[in] InterpolationMode The interpolation mode
/// \param[out] output Resized image of shape <outputHeight,outputWidth,C> or <outputHeight,outputWidth>
///                and same type as input
Status Resize(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output, int32_t output_height,
              int32_t output_width, double fx = 0.0, double fy = 0.0,
              InterpolationMode mode = InterpolationMode::kLinear);

/// \brief  Returns Resized image.
/// \param[in] inputs input TensorRow
/// \param[in] height Height of output
/// \param[in] width Width of output
/// \param[in] img_orientation Angle method of image rotation
/// \param[out] outputs Resized image of shape <height,width,C> and same type as input
Status ResizePreserve(const TensorRow &inputs, int32_t height, int32_t width, int32_t img_orientation,
                      TensorRow *outputs);

/// \brief Take in a 3 channel image in RBG to BGR
/// \param[in] input The input image
/// \param[out] output The output image
/// \return Status code
Status RgbToBgr(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output);

/// \brief Take in a 3 channel image in RBG to GRAY
/// \param[in] input The input image
/// \param[out] output The output image
/// \return Status code
Status RgbToGray(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output);

/// \brief Pads the input image and puts the padded image in the output
/// \param[in] input: input Tensor
/// \param[out] output: padded Tensor
/// \param[in] pad_top Amount of padding done in top
/// \param[in] pad_bottom Amount of padding done in bottom
/// \param[in] pad_left Amount of padding done in left
/// \param[in] pad_right Amount of padding done in right
/// \param[in] border_types The interpolation to be done in the border
/// \param[in] fill_r Red fill value for pad
/// \param[in] fill_g Green fill value for pad
/// \param[in] fill_b Blue fill value for pad
Status Pad(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output, const int32_t &pad_top,
           const int32_t &pad_bottom, const int32_t &pad_left, const int32_t &pad_right, const BorderType &border_types,
           uint8_t fill_r = 0, uint8_t fill_g = 0, uint8_t fill_b = 0);

/// \brief Rotate the input image by orientation
/// \param[in] input Input Tensor
/// \param[out] output Rotated Tensor
/// \param[in] orientation The orientation of EXIF
Status Rotate(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output, const uint64_t orientation);

/// \brief Get an affine matrix that applies affine transformation
/// \param[in] input Input Tensor
/// \param[in] matrix The transformation matrix
/// \param[in] degrees Range of the rotation degrees
/// \param[in] translation The horizontal and vertical translations
/// \param[in] scale The scaling factor
/// \param[in] shear The shear angle
Status GetAffineMatrix(const std::shared_ptr<Tensor> &input, std::vector<float_t> *matrix, float_t degrees,
                       const std::vector<float_t> &translation, float_t scale, const std::vector<float_t> &shear);

/// \brief Geometrically transform the input image
/// \param[in] input Input Tensor
/// \param[out] output Transformed Tensor
/// \param[in] degrees Range of the rotation degrees
/// \param[in] translation The horizontal and vertical translations
/// \param[in] scale The scaling factor
/// \param[in] shear The shear angle
/// \param[in] interpolation The interpolation mode
/// \param[in] fill_value Fill value for pad
Status Affine(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output, float_t degrees,
              const std::vector<float_t> &translation, float_t scale, const std::vector<float_t> &shear,
              InterpolationMode interpolation, const std::vector<uint8_t> &fill_value);

/// \brief Filter the input image with a Gaussian kernel
/// \param[in] input Input Tensor
/// \param[out] output Transformed Tensor
/// \param[in] kernel_size_x Gaussian kernel size of width
/// \param[in] kernel_size_y Gaussian kernel size of height
/// \param[in] sigma_x Gaussian kernel standard deviation of width
/// \param[in] sigma_y Gaussian kernel standard deviation of height
Status GaussianBlur(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output, int32_t kernel_size_x,
                    int32_t kernel_size_y, float sigma_x, float sigma_y);

/// \brief Get the size of input image.
/// \param[in] image Tensor of the image.
/// \param[out] size Size of the image as [height, width].
/// \return The status code.
Status ImageSize(const std::shared_ptr<Tensor> &image, std::vector<dsize_t> *size);

/// \brief Validate image Dtype, rank and channel.
/// \param[in] image Image tensor to be validated.
/// \param[in] op_name operator name.
/// \param[in] valid_dtype Valid date type of the image tensor. Default: {}, means not to check date type.
/// \param[in] valid_rank Valid dimension of the image tensor. Default: {}, means not to check dimension.
/// \param[in] valid_channel Valid channel of the image tensor. Default: {}, means not to check channel.
Status ValidateImage(const std::shared_ptr<Tensor> &image, const std::string &op_name,
                     const std::set<uint8_t> &valid_dtype = {}, const std::set<dsize_t> &valid_rank = {},
                     const std::set<dsize_t> &valid_channel = {});

/// \brief Validate image rank.
/// \param[in] op_name operator name.
/// \param[in] rank refers to the rank of input image shape.
Status ValidateImageRank(const std::string &op_name, int32_t rank);

/// \brief Swaps the channels in the image, i.e. converts HWC to CHW
/// \param input: Tensor of shape <H,W,C> or <H,W> and any OpenCv compatible type, see CVTensor.
/// \param output: Tensor of shape <C,H,W> or <H,W> and same input type.
Status HwcToChw(std::shared_ptr<Tensor> input, std::shared_ptr<Tensor> *output);
}  // namespace dataset
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_KERNELS_IMAGE_LITE_IMAGE_UTILS_H_
