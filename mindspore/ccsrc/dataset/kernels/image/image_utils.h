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
#ifndef DATASET_KERNELS_IMAGE_IMAGE_UTILS_H_
#define DATASET_KERNELS_IMAGE_IMAGE_UTILS_H_

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
#include <opencv2/imgproc/imgproc.hpp>
#include "dataset/core/tensor.h"
#include "dataset/kernels/tensor_op.h"
#include "dataset/util/status.h"

namespace mindspore {
namespace dataset {
enum class InterpolationMode { kLinear = 0, kNearestNeighbour = 1, kCubic = 2, kArea = 3 };

enum class BorderType { kConstant = 0, kEdge = 1, kReflect = 2, kSymmetric = 3 };

void JpegErrorExitCustom(j_common_ptr cinfo);

struct JpegErrorManagerCustom {
  // "public" fields
  struct jpeg_error_mgr pub;
  // for return to caller
  jmp_buf setjmp_buffer;
};

// Returns the interpolation mode in openCV format
// @param mode: interpolation mode in DE format
int GetCVInterpolationMode(InterpolationMode mode);

// Returns the openCV equivalent of the border type used for padding.
// @param type
// @return
int GetCVBorderType(BorderType type);

// Returns flipped image
// @param input/output: Tensor of shape <H,W,C> or <H,W> and any OpenCv compatible type, see CVTensor.
// @param flip_code: 1 for Horizontal (around y-axis), 0 for Vertical (around x-axis), -1 for both
// The flipping happens in place.
Status Flip(std::shared_ptr<Tensor> input, std::shared_ptr<Tensor> *output, int flip_code);

// Returns Horizontally flipped image
// @param input/output: Tensor of shape <H,W,C> or <H,W> and any OpenCv compatible type, see CVTensor.
// The flipping happens in place.
Status HorizontalFlip(std::shared_ptr<Tensor> input, std::shared_ptr<Tensor> *output);

// Returns Vertically flipped image
// @param input/output: Tensor of shape <H,W,C> or <H,W> and any OpenCv compatible type, see CVTensor.
// The flipping happens in place.
Status VerticalFlip(std::shared_ptr<Tensor> input, std::shared_ptr<Tensor> *output);

// Returns Resized image.
// @param input/output: Tensor of shape <H,W,C> or <H,W> and any OpenCv compatible type, see CVTensor.
// @param output_height: height of output
// @param output_width: width of output
// @param fx: horizontal scale
// @param fy: vertical scale
// @param InterpolationMode: the interpolation mode
// @param output: Resized image of shape <outputHeight,outputWidth,C> or <outputHeight,outputWidth>
//                and same type as input
Status Resize(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output, int32_t output_height,
              int32_t output_width, double fx = 0.0, double fy = 0.0,
              InterpolationMode mode = InterpolationMode::kLinear);

// Returns Decoded image
// Supported images:
//  BMP JPEG JPG PNG TIFF
// supported by opencv, if user need more image analysis capabilities, please compile opencv particularlly.
// @param input: CVTensor containing the not decoded image 1D bytes
// @param output: Decoded image Tensor of shape <H,W,C> and type DE_UINT8. Pixel order is RGB
Status Decode(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output);

Status DecodeCv(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output);

bool HasJpegMagic(const std::shared_ptr<Tensor> &input);

void JpegSetSource(j_decompress_ptr c_info, const void *data, int64_t data_size);

Status JpegCropAndDecode(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output, int x = 0, int y = 0,
                         int w = 0, int h = 0);
// Returns Rescaled image
// @param input: Tensor of shape <H,W,C> or <H,W> and any OpenCv compatible type, see CVTensor.
// @param rescale: rescale parameter
// @param shift: shift parameter
// @param output: Rescaled image Tensor of same input shape and type DE_FLOAT32
Status Rescale(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output, float rescale, float shift);

// Returns cropped ROI of an image
// @param input: Tensor of shape <H,W,C> or <H,W> and any OpenCv compatible type, see CVTensor.
// @param x: starting horizontal position of ROI
// @param y: starting vertical position of ROI
// @param w: width of the ROI
// @param h: height of the ROI
// @param output: Cropped image Tensor of shape <h,w,C> or <h,w> and same input type.
Status Crop(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output, int x, int y, int w, int h);

// Swaps the channels in the image, i.e. converts HWC to CHW
// @param input: Tensor of shape <H,W,C> or <H,W> and any OpenCv compatible type, see CVTensor.
// @param output: Tensor of shape <C,H,W> or <H,W> and same input type.
Status HwcToChw(std::shared_ptr<Tensor> input, std::shared_ptr<Tensor> *output);

// Swap the red and blue pixels (RGB <-> BGR)
// @param input: Tensor of shape <H,W,3> and any OpenCv compatible type, see CVTensor.
// @param output: Swapped image of same shape and type
Status SwapRedAndBlue(std::shared_ptr<Tensor> input, std::shared_ptr<Tensor> *output);

// Crops and resizes the image
// @param input: Tensor of shape <H,W,C> or <H,W> and any OpenCv compatible type, see CVTensor.
// @param x: horizontal start point
// @param y: vertical start point
// @param crop_height: height of the cropped ROI
// @param crop_width: width of the cropped ROI
// @param target_width: width of the final resized image
// @param target_height: height of the final resized image
// @param InterpolationMode: the interpolation used in resize operation
// @param output: Tensor of shape <targetHeight,targetWidth,C> or <targetHeight,targetWidth>
//                and same type as input
Status CropAndResize(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output, int x, int y,
                     int crop_height, int crop_width, int target_height, int target_width, InterpolationMode mode);

// Returns rotated image
// @param input: Tensor of shape <H,W,C> or <H,W> and any OpenCv compatible type, see CVTensor.
// @param fx: rotation center x coordinate
// @param fy: rotation center y coordinate
// @param degree: degree to rotate
// @param expand: if reshape is necessary
// @param output: rotated image of same input type.
Status Rotate(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output, float fx, float fy, float degree,
              InterpolationMode interpolation = InterpolationMode::kNearestNeighbour, bool expand = false,
              uint8_t fill_r = 0, uint8_t fill_g = 0, uint8_t fill_b = 0);

// Returns Normalized image
// @param input: Tensor of shape <H,W,C> in RGB order and any OpenCv compatible type, see CVTensor.
// @param mean: Tensor of shape <3> and type DE_FLOAT32 which are mean of each channel in RGB order
// @param std:  Tensor of shape <3> and type DE_FLOAT32 which are std of each channel in RGB order
// @param output: Normalized image Tensor of same input shape and type DE_FLOAT32
Status Normalize(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output,
                 const std::shared_ptr<Tensor> &mean, const std::shared_ptr<Tensor> &std);

// Returns image with adjusted brightness.
// @param input: Tensor of shape <H,W,3> in RGB order and any OpenCv compatible type, see CVTensor.
// @param alpha: Alpha value to adjust brightness by. Should be a positive number.
//               If user input one value in python, the range is [1 - value, 1 + value].
//               This will output original image multiplied by alpha. 0 gives a black image, 1 gives the
//               original image while 2 increases the brightness by a factor of 2.
// @param output: Adjusted image of same shape and type.
Status AdjustBrightness(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output, const float &alpha);

// Returns image with adjusted contrast.
// @param input: Tensor of shape <H,W,3> in RGB order and any OpenCv compatible type, see CVTensor.
// @param alpha: Alpha value to adjust contrast by. Should be a positive number.
//               If user input one value in python, the range is [1 - value, 1 + value].
//               0 gives a solid gray image, 1 gives the original image while 2 increases
//               the contrast by a factor of 2.
// @param output: Adjusted image of same shape and type.
Status AdjustContrast(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output, const float &alpha);

// Returns image with adjusted saturation.
// @param input: Tensor of shape <H,W,3> in RGB order and any OpenCv compatible type, see CVTensor.
// @param alpha: Alpha value to adjust saturation by. Should be a positive number.
//               If user input one value in python, the range is [1 - value, 1 + value].
//               0 will give a black and white image, 1 will give the original image while
//               2 will enhance the saturation by a factor of 2.
// @param output: Adjusted image of same shape and type.
Status AdjustSaturation(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output, const float &alpha);

// Returns image with adjusted hue.
// @param input: Tensor of shape <H,W,3> in RGB order and any OpenCv compatible type, see CVTensor.
// @param hue: Hue value to adjust by, should be within range [-0.5, 0.5]. 0.5 and - 0.5 will reverse the hue channel
//             completely.
//             If user input one value in python, the range is [-value, value].
// @param output: Adjusted image of same shape and type.
Status AdjustHue(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output, const float &hue);

// Masks out a random section from the image with set dimension
// @param input: input Tensor
// @param output: cutOut Tensor
// @param box_height: height of the cropped box
// @param box_width: width of the cropped box
// @param num_patches: number of boxes to cut out from the image
// @param bounded: boolean flag to toggle between random erasing and cutout
// @param random_color: whether or not random fill value should be used
// @param fill_r: red fill value for erase
// @param fill_g: green fill value for erase
// @param fill_b: blue fill value for erase.
Status Erase(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output, int32_t box_height,
             int32_t box_width, int32_t num_patches, bool bounded, bool random_color, std::mt19937 *rnd,
             uint8_t fill_r = 0, uint8_t fill_g = 0, uint8_t fill_b = 0);

// Pads the input image and puts the padded image in the output
// @param input: input Tensor
// @param output: padded Tensor
// @param pad_top: amount of padding done in top
// @param pad_bottom: amount of padding done in bottom
// @param pad_left: amount of padding done in left
// @param pad_right: amount of padding done in right
// @param border_types: the interpolation to be done in the border
// @param fill_r: red fill value for pad
// @param fill_g: green fill value for pad
// @param fill_b: blue fill value for pad.
Status Pad(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output, const int32_t &pad_top,
           const int32_t &pad_bottom, const int32_t &pad_left, const int32_t &pad_right, const BorderType &border_types,
           uint8_t fill_r = 0, uint8_t fill_g = 0, uint8_t fill_b = 0);
}  // namespace dataset
}  // namespace mindspore

#endif  // DATASET_KERNELS_IMAGE_IMAGE_UTILS_H_
