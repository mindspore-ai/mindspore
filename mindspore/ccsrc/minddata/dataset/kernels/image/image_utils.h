/**
 * Copyright 2019-2022 Huawei Technologies Co., Ltd
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
#include <set>
#include <string>
#include <utility>
#include <vector>
#if defined(_WIN32) || defined(_WIN64)
#undef HAVE_STDDEF_H
#undef HAVE_STDLIB_H
#elif __APPLE__
#include <sys/param.h>
#include <sys/mount.h>
#endif
#include "./jpeglib.h"
#include "./jerror.h"
#include <opencv2/imgproc/imgproc.hpp>
#include "minddata/dataset/core/tensor.h"
#include "minddata/dataset/kernels/tensor_op.h"
#include "minddata/dataset/util/status.h"
#include "minddata/dataset/util/validators.h"

namespace mindspore {
namespace dataset {
constexpr dsize_t kChannelIndexHWC = 2;      // images are hwc, so index 2 represents number of channels
constexpr dsize_t kChannelIndexCHW = 0;      // images are chw, so index 0 represents number of channels
constexpr dsize_t kMinImageRank = 2;         // images have at least 2 dimensions
constexpr dsize_t kDefaultImageRank = 3;     // images are hwc channels in general
constexpr int32_t kMaxBitValue = 255;        // max bit value after decode is 256
constexpr dsize_t kMinImageChannel = 1;      // image ops support minimum of 1 channel
constexpr dsize_t kDefaultImageChannel = 3;  // images are 3 channels in general
constexpr dsize_t kMaxImageChannel = 4;      // image ops support maximum of 4 channel
constexpr float kHalf = 0.5;                 // to get the half of a value
constexpr dsize_t kRIndex = 0;               // index of red channel in RGB format
constexpr dsize_t kGIndex = 1;               // index of green channel in RGB format
constexpr dsize_t kBIndex = 2;               // index of blue channel in RGB format
constexpr dsize_t kHeightIndex = 0;          // index of height of HWC images
constexpr dsize_t kWidthIndex = 1;           // index of width of HWC images
constexpr dsize_t kMinJpegQuality = 1;       // the minimum quality for JPEG
constexpr dsize_t kMaxJpegQuality = 100;     // the maximum quality for JPEG

void JpegErrorExitCustom(j_common_ptr cinfo);

struct JpegErrorManagerCustom {
  // "public" fields
  struct jpeg_error_mgr pub;
  // for return to caller
  jmp_buf setjmp_buffer;
};

/// \brief Returns the interpolation mode in openCV format
/// \param[in] mode Interpolation mode in DE format
int GetCVInterpolationMode(InterpolationMode mode);

/// \brief Returns the openCV equivalent of the border type used for padding.
/// \param type
/// \return Status code
int GetCVBorderType(BorderType type);

/// \brief Get the number of input image channels.
/// \param[in] image Tensor of the image.
/// \param[out] channels Channels of the image.
/// \return The status code.
Status ImageNumChannels(const std::shared_ptr<Tensor> &image, dsize_t *channels);

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

/// \brief Validate image dtype.
/// \param[in] op_name operator name.
/// \param[in] dtype Date type of the image tensor.
Status ValidateImageDtype(const std::string &op_name, DataType dtype);

/// \brief Validate image rank.
/// \param[in] op_name operator name.
/// \param[in] rank refers to the rank of input image shape.
Status ValidateImageRank(const std::string &op_name, int32_t rank);

/// \brief Returns the check result of tensor rank and tensor shape
/// \param[in] tensor: The input tensor need to check
/// \param[in] channel: The channel index of tensor shape.
/// \param[out] return true if channel of tensor shape is 1 or 3.
bool CheckTensorShape(const std::shared_ptr<Tensor> &tensor, const int &channel);

/// \brief Returns flipped image
/// \param[in] input/output: Tensor of shape <H,W,C> or <H,W> and any OpenCv compatible type, see CVTensor.
/// \param flip_code: 1 for Horizontal (around y-axis), 0 for Vertical (around x-axis), -1 for both
///     The flipping happens in place.
Status Flip(std::shared_ptr<Tensor> input, std::shared_ptr<Tensor> *output, int flip_code);

/// \brief Returns Horizontally flipped image
/// \param input/output: Tensor of shape <H,W,C> or <H,W> and any OpenCv compatible type, see CVTensor.
/// The flipping happens in place.
Status HorizontalFlip(std::shared_ptr<Tensor> input, std::shared_ptr<Tensor> *output);

/// \brief Returns Vertically flipped image
/// \param input/output: Tensor of shape <H,W,C> or <H,W> and any OpenCv compatible type, see CVTensor.
/// \note The flipping happens in place.
Status VerticalFlip(std::shared_ptr<Tensor> input, std::shared_ptr<Tensor> *output);

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

/// \brief Returns Decoded image
/// Supported images:
///  BMP JPEG JPG PNG TIFF
/// supported by opencv, if user need more image analysis capabilities, please compile opencv particularlly.
/// \param input: CVTensor containing the not decoded image 1D bytes
/// \param output: Decoded image Tensor of shape <H,W,C> and type DE_UINT8. Pixel order is RGB
Status Decode(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output);

Status DecodeCv(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output);

DATASET_API bool IsNonEmptyJPEG(const std::shared_ptr<Tensor> &input);

bool IsNonEmptyPNG(const std::shared_ptr<Tensor> &input);

void JpegSetSource(j_decompress_ptr c_info, const void *data, int64_t data_size);

Status JpegCropAndDecode(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output, int x = 0, int y = 0,
                         int w = 0, int h = 0);

/// \brief Returns Rescaled image
/// \param input: Tensor of shape <H,W,C> or <H,W> and any OpenCv compatible type, see CVTensor.
/// \param rescale: rescale parameter
/// \param shift: shift parameter
/// \param output: Rescaled image Tensor of same input shape and type DE_FLOAT32
Status Rescale(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output, float rescale, float shift);

/// \brief Returns cropped ROI of an image
/// \param input: Tensor of shape <H,W,C> or <H,W> and any OpenCv compatible type, see CVTensor.
/// \param x: starting horizontal position of ROI
/// \param y: starting vertical position of ROI
/// \param w: width of the ROI
/// \param h: height of the ROI
/// \param output: Cropped image Tensor of shape <h,w,C> or <h,w> and same input type.
Status Crop(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output, int x, int y, int w, int h);

/// \brief Change the color space of the image.
/// \param input: The input image.
/// \param output: The output image.
/// \param convert_mode: The mode of image channel conversion.
Status ConvertColor(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output, ConvertMode convert_mode);

/// \brief Swaps the channels in the image, i.e. converts HWC to CHW
/// \param input: Tensor of shape <H,W,C> or <H,W> and any OpenCv compatible type, see CVTensor.
/// \param output: Tensor of shape <C,H,W> or <H,W> and same input type.
Status HwcToChw(std::shared_ptr<Tensor> input, std::shared_ptr<Tensor> *output);

/// \brief Masks the given part of the input image with a another image (sub_mat)
/// \param[in] sub_mat The image we want to mask with
/// \param[in] input The pointer to the image we want to mask
/// \param[in] x The horizontal coordinate of left side of crop box
/// \param[in] y The vertical coordinate of the top side of crop box
/// \param[in] width The width of the mask box
/// \param[in] height The height of the mask box
/// \param[in] image_format The format of the image (CHW or HWC)
/// \param[out] input Masks the input image in-place and returns it
/// @return Status ok/error
Status MaskWithTensor(const std::shared_ptr<Tensor> &sub_mat, std::shared_ptr<Tensor> *input, int x, int y, int width,
                      int height, ImageFormat image_format);

/// \brief Copies a value from a source tensor into a destination tensor
/// \note This is meant for images and therefore only works if tensor is uint8 or float32
/// \param[in] source_tensor The tensor we take the value from
/// \param[in] dest_tensor The pointer to the tensor we want to copy the value to
/// \param[in] source_indx index of the value in the source tensor
/// \param[in] dest_indx index of the value in the destination tensor
/// \param[out] dest_tensor Copies the value to the given dest_tensor and returns it
/// @return Status ok/error
Status CopyTensorValue(const std::shared_ptr<Tensor> &source_tensor, std::shared_ptr<Tensor> *dest_tensor,
                       const std::vector<int64_t> &source_indx, const std::vector<int64_t> &dest_indx);

/// \brief Swap the red and blue pixels (RGB <-> BGR)
/// \param input: Tensor of shape <H,W,3> and any OpenCv compatible type, see CVTensor.
/// \param output: Swapped image of same shape and type
Status SwapRedAndBlue(std::shared_ptr<Tensor> input, std::shared_ptr<Tensor> *output);

/// \brief Crops and resizes the image
/// \param input: Tensor of shape <H,W,C> or <H,W> and any OpenCv compatible type, see CVTensor.
/// \param x: horizontal start point
/// \param y: vertical start point
/// \param crop_height: height of the cropped ROI
/// \param crop_width: width of the cropped ROI
/// \param target_width: width of the final resized image
/// \param target_height: height of the final resized image
/// \param InterpolationMode: the interpolation used in resize operation
/// \param output: Tensor of shape <targetHeight,targetWidth,C> or <targetHeight,targetWidth>
///     and same type as input
Status CropAndResize(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output, int x, int y,
                     int crop_height, int crop_width, int target_height, int target_width, InterpolationMode mode);

/// \brief Returns rotated image
/// \param input: Tensor of shape <H,W,C> or <H,W> and any OpenCv compatible type, see CVTensor.
/// \param center: rotation center
/// \param degree: degree to rotate
/// \param expand: if reshape is necessary
/// \param output: rotated image of same input type.
Status Rotate(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output, std::vector<float> center,
              float degree, InterpolationMode interpolation = InterpolationMode::kNearestNeighbour, bool expand = false,
              uint8_t fill_r = 0, uint8_t fill_g = 0, uint8_t fill_b = 0);

/// \brief Returns Normalized image
/// \param input: Tensor of shape <H,W,C> in RGB order and any OpenCv compatible type, see CVTensor.
/// \param mean: Tensor of shape <3> and type DE_FLOAT32 which are mean of each channel in RGB order
/// \param std:  Tensor of shape <3> and type DE_FLOAT32 which are std of each channel in RGB order
/// \param is_hwc: Check if input is HWC/CHW format
/// \param output: Normalized image Tensor of same input shape and type DE_FLOAT32
Status Normalize(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output, std::vector<float> mean,
                 std::vector<float> std, bool is_hwc);

/// \brief Returns Normalized and padded image
/// \param input: Tensor of shape <H,W,C> in RGB order and any OpenCv compatible type, see CVTensor.
/// \param mean: vector of float values which are mean of each channel
/// \param std:  vector of float values which are std of each channel
/// \param dtype: output dtype
/// \param is_hwc: Check if input is HWC/CHW format
/// \param output: Normalized image Tensor and pad an extra channel, return a dtype Tensor
Status NormalizePad(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output, std::vector<float> mean,
                    std::vector<float> std, const std::string &dtype, bool is_hwc);

/// \brief Returns image with adjusted brightness.
/// \param input: Tensor of shape <H,W,3> in RGB order and any OpenCv compatible type, see CVTensor.
/// \param alpha: Alpha value to adjust brightness by. Should be a positive number.
///               If user input one value in python, the range is [1 - value, 1 + value].
///               This will output original image multiplied by alpha. 0 gives a black image, 1 gives the
///               original image while 2 increases the brightness by a factor of 2.
/// \param output: Adjusted image of same shape and type.
Status AdjustBrightness(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output, float alpha);

/// \brief Returns image with adjusted contrast.
/// \param input: Tensor of shape <H,W,3> in RGB order and any OpenCv compatible type, see CVTensor.
/// \param alpha: Alpha value to adjust contrast by. Should be a positive number.
///               If user input one value in python, the range is [1 - value, 1 + value].
///               0 gives a solid gray image, 1 gives the original image while 2 increases
///               the contrast by a factor of 2.
/// \param output: Adjusted image of same shape and type.
Status AdjustContrast(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output, float alpha);

/// \brief Returns image with contrast maximized.
/// \param input: Tensor of shape <H,W,3>/<H,W,1>/<H,W> in RGB/Grayscale and any OpenCv compatible type, see CVTensor.
/// \param cutoff: Cutoff percentage of how many pixels are to be removed (high pixels change to 255 and low change
///     to 0) from the high and low ends of the histogram.
/// \param ignore: Pixel values to be ignored in the algorithm.
Status AutoContrast(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output, float cutoff,
                    const std::vector<uint32_t> &ignore);

/// \brief Returns image with gamma correction.
/// \param[in] input: Tensor of shape <H,W,3>/<H,W,1>/<H,W> in RGB/Grayscale and any OpenCV compatible type,
///     see CVTensor.
/// \param[in] gamma: Non negative real number, same as gamma in the equation. gamma larger than 1 make the shadows
///     darker, while gamma smaller than 1 make dark regions lighter.
/// \param[in] gain: The constant multiplier.
/// \param[out] output: Adjusted image of same shape and type.
Status AdjustGamma(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output, float gamma, float gain);

/// \brief Returns image with adjusted saturation.
/// \param input: Tensor of shape <H,W,3> in RGB order and any OpenCv compatible type, see CVTensor.
/// \param alpha: Alpha value to adjust saturation by. Should be a positive number.
///               If user input one value in python, the range is [1 - value, 1 + value].
///               0 will give a black and white image, 1 will give the original image while
///               2 will enhance the saturation by a factor of 2.
/// \param output: Adjusted image of same shape and type.
Status AdjustSaturation(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output, float alpha);

/// \brief Returns image with adjusted hue.
/// \param input: Tensor of shape <H,W,3> in RGB order and any OpenCv compatible type, see CVTensor.
/// \param hue: Hue value to adjust by, should be within range [-0.5, 0.5]. 0.5 and - 0.5 will reverse the hue channel
///             completely.
///             If user input one value in python, the range is [-value, value].
/// \param output: Adjusted image of same shape and type.
Status AdjustHue(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output, float hue);

/// \brief Returns image with equalized histogram.
/// \param[in] input: Tensor of shape <H,W,3>/<H,W,1>/<H,W> in RGB/Grayscale and
///                   any OpenCv compatible type, see CVTensor.
/// \param[out] output: Equalized image of same shape and type.
Status Equalize(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output);

/// \brief Masks out a random section from the image with set dimension
/// \param input: input Tensor
/// \param output: cutOut Tensor
/// \param box_height: height of the cropped box
/// \param box_width: width of the cropped box
/// \param num_patches: number of boxes to cut out from the image
/// \param bounded: boolean flag to toggle between random erasing and cutout
/// \param random_color: whether or not random fill value should be used
/// \param fill_colors: vector of color fill values for erase
/// \param is_hwc: Check if input is HWC/CHW format
Status CutOut(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output, int32_t box_height,
              int32_t box_width, int32_t num_patches, bool bounded, bool random_color, std::mt19937 *rnd,
              std::vector<uint8_t> fill_colors = {}, bool is_hwc = true);

/// \brief Erase the input image with given value
/// \param input: input Tensor
/// \param output: erase Tensor
/// \param top: top of the cropped box
/// \param left: left of the cropped box
/// \param height: height of the cropped box
/// \param width: width of the cropped box
/// \param value: fill value for erase
/// \param inplace: whether to apply erasing inplace
Status Erase(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output, int32_t top, int32_t left,
             int32_t height, int32_t width, const std::vector<uint8_t> &value, bool inplace);

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

/// \brief Add AlexNet-style PCA-based noise to an image.
/// \param[in] input The input image.
/// \param[out] output The output image.
/// \param[in] rnd_r Random weight for red channel.
/// \param[in] rnd_g Random weight for green channel.
/// \param[in] rnd_b Random weight for blue channel.
/// \return Status code.
Status RandomLighting(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output, float rnd_r, float rnd_g,
                      float rnd_b);

/// \brief Take in a 4 channel image in RBGA to RGB
/// \param[in] input The input image
/// \param[out] output The output image
/// \return Status code
Status RgbaToRgb(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output);

/// \brief Take in a 4 channel image in RBGA to BGR
/// \param[in] input The input image
/// \param[out] output The output image
/// \return Status code
Status RgbaToBgr(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output);

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

/// \brief Get jpeg image width and height
/// \param input: CVTensor containing the not decoded image 1D bytes
/// \param img_width: the jpeg image width
/// \param img_height: the jpeg image height
Status GetJpegImageInfo(const std::shared_ptr<Tensor> &input, int *img_width, int *img_height);

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

/// \brief Apply perspective transformation on input image.
/// \param[in] input Input Tensor.
/// \param[out] output Transformed Tensor.
/// \param[in] start_points List containing four lists of two integers corresponding to four
///     corners [top-left, top-right, bottom-right, bottom-left] of the original image.
/// \param[in] end_points List containing four lists of two integers corresponding to four
///     corners [top-left, top-right, bottom-right, bottom-left] of the transformed image.
/// \param[in] interpolation Method of interpolation.
Status Perspective(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output,
                   const std::vector<std::vector<int32_t>> &start_points,
                   const std::vector<std::vector<int32_t>> &end_points, InterpolationMode interpolation);

/// \brief Slice tensor to multiple patches.
/// \param[in] input Input Tensor
/// \param[out] output Vector of Output Tensor
/// \param[in] num_height Number of patches in vertical direction.
/// \param[in] num_width Number of patches in horizontal direction.
/// \param[in] slice_mode Mode represents padding or drop.
/// \param[in] fill_value The value of filled pixel in right and bottom border when padding.
Status SlicePatches(const std::shared_ptr<Tensor> &input, std::vector<std::shared_ptr<Tensor>> *output,
                    int32_t num_height, int32_t num_width, SliceMode slice_mode, uint8_t fill_value);

/// \brief Solarize the image by inverting all pixel values within the threshold.
/// \param[in] input Input Tensor
/// \param[out] output Output Tensor
/// \param[in] threshold Pixel value range to be inverted.
Status Solarize(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output,
                const std::vector<float> &threshold);

/// \brief Compute patch height and width.
/// \param[in] input Input CVTensor
/// \param[out] patch_size Size of patch
/// \param[in] num_height Number of patches in vertical direction.
/// \param[in] num_width Number of patches in horizontal direction.
/// \param[in] slice_mode Mode represents padding or drop.
Status ComputePatchSize(const std::shared_ptr<CVTensor> &input_cv,
                        std::shared_ptr<std::pair<int32_t, int32_t>> *patch_size, int32_t num_height, int32_t num_width,
                        SliceMode slice_mode);

/// \brief Rescale and convert HWC to CHW format.
/// \param[in] input The input image
/// \param[in] data_type The output data type
/// \param[out] output The output image
/// \return Status code
Status ToTensor(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output, const DataType &data_type);

/// \brief Generate a vector that contains n numbers between start and end with evenly interval.
/// \param[in] start Start number.
/// \param[in] end End number.
/// \param[in] n Count of numbers.
/// \param[in] scale Zoom scale.
/// \param[in] offset Bias.
/// \param[in] round Round input to the nearest integer.
std::vector<float> Linspace(float start, float end, int32_t n, float scale = 1.0, float offset = 0, bool round = false);

/// \brief Round input to the nearest integer. Note that this function implements the "round half to even" to break
///     ties when a number is equidistant from two integers.
/// \param[in] value Input value.
float Round(float value);

/// \brief Perform the selected augment.
/// \param[in] input The input tensor.
/// \param[in] output The output tensor.
/// \param[in] op_name The selected op.
/// \param[in] magnitude The magnitude value.
/// \param[in] interpolation Possible options for interpolation method.
/// \param[in] fill_value Values used to fill.
Status ApplyAugment(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output, const std::string &op_name,
                    float magnitude, InterpolationMode interpolation, const std::vector<uint8_t> &fill_value);

/// \brief Encode the image as JPEG data.
/// \param[in] image The image to be encoded.
/// \param[out] output The Tensor data.
/// \param[in] quality The quality for the output tensor from 1 to 100. Default: 75.
/// \return The status code.
Status EncodeJpeg(const std::shared_ptr<Tensor> &image, std::shared_ptr<Tensor> *output, int quality = 75);

/// \brief Write the one dimension uint8 data into a file using binary mode.
/// \param[in] filename The path to the file to be written.
/// \param[in] data The tensor data.
/// \return The status code.
Status WriteFile(const std::string &filename, const std::shared_ptr<Tensor> &data);
}  // namespace dataset
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_KERNELS_IMAGE_IMAGE_UTILS_H_
