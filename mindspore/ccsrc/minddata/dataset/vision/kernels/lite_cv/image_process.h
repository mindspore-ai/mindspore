/**
 * Copyright 2020-2025 Huawei Technologies Co., Ltd
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

#ifndef IMAGE_PROCESS_H_
#define IMAGE_PROCESS_H_

#include <algorithm>
#include <vector>

#include "minddata/dataset/util/visible.h"
#include "minddata/dataset/vision/kernels/lite_cv/lite_mat.h"

namespace mindspore::dataset {
#define CV_PI 3.1415926535897932384626433832795
#define IM_TOOL_EXIF_ORIENTATION_0_DEG 1
#define IM_TOOL_EXIF_ORIENTATION_0_DEG_MIRROR 2
#define IM_TOOL_EXIF_ORIENTATION_180_DEG 3
#define IM_TOOL_EXIF_ORIENTATION_180_DEG_MIRROR 4
#define IM_TOOL_EXIF_ORIENTATION_90_DEG_MIRROR 5
#define IM_TOOL_EXIF_ORIENTATION_90_DEG 6
#define IM_TOOL_EXIF_ORIENTATION_270_DEG_MIRROR 7
#define IM_TOOL_EXIF_ORIENTATION_270_DEG 8
#define NUM_OF_RGB_CHANNELS 9
#define IM_TOOL_DATA_TYPE_FLOAT (1)
#define IM_TOOL_DATA_TYPE_UINT8 (2)
#define IM_TOOL_RETURN_STATUS_SUCCESS (0)
#define IM_TOOL_RETURN_STATUS_INVALID_INPUT (1)
#define IM_TOOL_RETURN_STATUS_FAILED (2)

#define INT16_CAST(X) \
  static_cast<int16_t>(::std::min(::std::max(static_cast<int>(X + (X >= 0.f ? 0.5f : -0.5f)), -32768), 32767))

enum PaddBorderType {
  PADD_BORDER_CONSTANT = 0,                     /**< Fills the border with constant values. */
  PADD_BORDER_REPLICATE = 1,                    /**< Fills the border with replicate mode. */
  PADD_BORDER_REFLECT_101 = 4,                  /**< Fills the border with reflect 101 mode. */
  PADD_BORDER_DEFAULT = PADD_BORDER_REFLECT_101 /**< Default pad mode, use reflect 101 mode. */
};

struct BoxesConfig {
 public:
  std::vector<size_t> img_shape;
  std::vector<int> num_default;
  std::vector<int> feature_size;
  float min_scale;
  float max_scale;
  std::vector<std::vector<float>> aspect_rations;
  std::vector<int> steps;
  std::vector<float> prior_scaling;
};

/// \brief resizing image by bilinear algorithm, the data type of currently only supports is uint8,
///          the channel of currently supports is 3 and 1.
/// \param[in] src Input image data.
/// \param[in] dst Output image data.
/// \param[in] dst_w The width of the output image.
/// \param[in] dst_h The length of the output image.
/// \par Example
/// \code
///     /* Assume p_rgb is a pointer that points to an image with shape (width, height, channel) */
///     LiteMat lite_mat_src(width, height, channel, (void *)p_rgb, LDataType::UINT8);
///     LiteMat lite_mat_dst;
///
///     /* Resize to (256, 256, 3) */
///     ResizeBilinear(lite_mat_src, lite_mat_dst, 256, 256);
///     std::cout << lite_mat_dst.width_ << " " << lite_mat_dst.height_ << " " << lite_mat_dst.channel_ << std::endl;
/// \endcode
/// \return Return true if transform successfully.
bool DATASET_API ResizeBilinear(const LiteMat &src, LiteMat &dst, int dst_w, int dst_h);

/// \brief Init Lite Mat from pixel, the conversion of currently supports is rbgaTorgb and rgbaTobgr.
/// \note The length of the pointer must be the same as that of the multiplication of w and h.
/// \param[in] data Input image data.
/// \param[in] pixel_type The type of pixel (refer to enum LPixelType).
///   - LPixelType.BGR, pixel in BGR type.
///   - LPixelType.RGB, pixel in RGB type.
///   - LPixelType.RGBA, pixel in RGBA type.
///   - LPixelType.RGBA2GRAY, convert image from RGBA to GRAY.
///   - LPixelType.RGBA2BGR, convert image from RGBA to BGR.
///   - LPixelType.RGBA2RGB, convert image from RGBA to RGB.
///   - LPixelType.NV212BGR, convert image from NV21 to BGR.
///   - LPixelType.NV122BGR, convert image from NV12 to BGR.
/// \param[in] data_type The type of data (refer to LDataType class).
/// \param[in] w The width of the output image.
/// \param[in] h The length of the output image.
/// \param[in] m Used to store image data.
/// \par Example
/// \code
///     /* Assume p_rgb is a pointer that points to an image with shape (width, height, channel) */
///     LiteMat lite_mat_dst;
///     InitFromPixel(p_rgb, LPixelType::RGB, LDataType::UINT8, width, height, lite_mat_dst);
///
///     std::cout << lite_mat_dst.width_ << " " << lite_mat_dst.height_ << " " << lite_mat_dst.channel_ << std::endl;
/// \endcode
/// \return Return true if transform successfully.
bool DATASET_API InitFromPixel(const unsigned char *data, LPixelType pixel_type, LDataType data_type, int w, int h,
                               LiteMat &m);

/// \brief convert the data type, the conversion of currently supports is uint8 to float.
/// \param[in] src Input image data.
/// \param[in] dst Output image data.
/// \param[in] scale Scale pixel value(default:1.0).
/// \par Example
/// \code
///     /* Assume p_rgb is a pointer that points to an image with shape (width, height, channel) */
///     LiteMat lite_mat_src;
///     InitFromPixel(p_rgb, LPixelType::RGB, LDataType::UINT8, width, height, lite_mat_dst);
///
///     LiteMat lite_mat_dst;
///     ConvertTo(lite_mat_src, lite_mat_dst);
///     std::cout << lite_mat_dst.width_ << " " << lite_mat_dst.height_ << " " << lite_mat_dst.channel_ << std::endl;
/// \endcode
/// \return Return true if transform successfully.
bool DATASET_API ConvertTo(const LiteMat &src, LiteMat &dst, double scale = 1.0);

/// \brief crop image, the channel supports is 3 and 1.
/// \param[in] src Input image data.
/// \param[in] dst Output image data.
/// \param[in] x The x coordinate value of the starting point of the screenshot.
/// \param[in] y The y coordinate value of the starting point of the screenshot.
/// \param[in] w The width of the screenshot.
/// \param[in] h The height of the screenshot.
/// \par Example
/// \code
///     /* Assume p_rgb is a pointer that points to an image with shape (width, height, channel) */
///     LiteMat lite_mat_src(width, height, channel, (void *)p_rgb, LDataType::UINT8);
///     LiteMat lite_mat_dst;
///
///     /* Crop to (32, 32, 3) */
///     Crop(lite_mat_src, lite_mat_dst, 0, 0, 32, 32);
///     std::cout << lite_mat_dst.width_ << " " << lite_mat_dst.height_ << " " << lite_mat_dst.channel_ << std::endl;
/// \endcode
/// \return Return true if transform successfully.
bool DATASET_API Crop(const LiteMat &src, LiteMat &dst, int x, int y, int w, int h);

/// \brief normalize image, currently the supports data type is float.
/// \param[in] src Input image data.
/// \param[in] dst Output image data.
/// \param[in] mean Mean of the data set.
/// \param[in] std Norm of the data set.
/// \par Example
/// \code
///     /* Assume p_rgb is a pointer that points to an image with shape (width, height, channel) */
///     LiteMat lite_mat_src(width, height, channel, (void *)p_rgb, LDataType::UINT8);
///     LiteMat lite_mat_src2;
///     ConvertTo(lite_mat_src, lite_mat_src2);
///     LiteMat lite_mat_dst;
///
///     /* Normalize */
///     std::vector<float> means = {0.485, 0.456, 0.406};
///     std::vector<float> stds = {0.229, 0.224, 0.225};
///     SubStractMeanNormalize(lite_mat_src2, lite_mat_dst, means, stds);
///     std::cout << lite_mat_dst.width_ << " " << lite_mat_dst.height_ << " " << lite_mat_dst.channel_ << std::endl;
/// \endcode
/// \return Return true if transform successfully.
bool DATASET_API SubStractMeanNormalize(const LiteMat &src, LiteMat &dst, const std::vector<float> &mean,
                                        const std::vector<float> &std);

/// \brief padd image, the channel supports is 3 and 1.
/// \param[in] src Input image data.
/// \param[in] dst Output image data.
/// \param[in] top The length of top.
/// \param[in] bottom The length of bottom.
/// \param[in] left The length of left.
/// \param[in] right he length of right.
/// \param[in] pad_type The type of pad.
///   - PaddBorderType.PADD_BORDER_CONSTANT, fills the border with constant values.
///   - PaddBorderType.PADD_BORDER_REPLICATE, fills the border with replicate mode.
///   - PaddBorderType.PADD_BORDER_REFLECT_101, fills the border with reflect 101 mode.
///   - PaddBorderType.PADD_BORDER_DEFAULT, default pad mode, use reflect 101 mode.
/// \param[in] fill_b_or_gray B or GRAY.
/// \param[in] fill_g G.
/// \param[in] fill_r R.
/// \par Example
/// \code
///     /* Assume p_rgb is a pointer that points to an image with shape (width, height, channel) */
///     LiteMat lite_mat_src(width, height, channel, (void *)p_rgb, LDataType::FLOAT32);
///     LiteMat lite_mat_dst;
///
///     /* Pad image with 4 pixels */
///     Pad(lite_mat_src, lite_mat_dst, 4, 4, 4, 4, PaddBorderType::PADD_BORDER_CONSTANT);
///     std::cout << lite_mat_dst.width_ << " " << lite_mat_dst.height_ << " " << lite_mat_dst.channel_ << std::endl;
/// \endcode
/// \return Return true if transform successfully.
bool DATASET_API Pad(const LiteMat &src, LiteMat &dst, int top, int bottom, int left, int right,
                     PaddBorderType pad_type, uint8_t fill_b_or_gray = 0, uint8_t fill_g = 0, uint8_t fill_r = 0);

/// \brief Extract image channel by index.
/// \param[in] src Input image data.
/// \param[in] dst Output image data.
/// \param[in] col The serial number of the channel.
/// \par Example
/// \code
///     /* Assume p_rgb is a pointer that points to an image with shape (width, height, channel) */
///     LiteMat lite_mat_src(width, height, channel, (void *)p_rgb, LDataType::UINT8);
///     LiteMat lite_mat_dst;
///
///     /* Extract the first channel of image */
///     ExtractChannel(lite_mat_src, lite_mat_dst, 0);
///     std::cout << lite_mat_dst.width_ << " " << lite_mat_dst.height_ << " " << lite_mat_dst.channel_ << std::endl;
/// \endcode
/// \return Return true if transform successfully.
bool DATASET_API ExtractChannel(LiteMat &src, LiteMat &dst, int col);

/// \brief Split image channels.
/// \param[in] src Input image data.
/// \param[in] mv Vector of LiteMat containing all channels.
/// \par Example
/// \code
///     /* Assume p_rgb is a pointer that points to an image with shape (width, height, channel) */
///     LiteMat lite_mat_src(width, height, channel, (void *)p_rgb, LDataType::UINT8);
///     std::vector<LiteMat> lite_mat_dst;
///
///     /* Extract all channels of image */
///     Split(lite_mat_src, lite_mat_dst);
/// \endcode
/// \return Return true if transform successfully.
bool DATASET_API Split(const LiteMat &src, std::vector<LiteMat> &mv);

/// \brief Create a multi-channel image out of several single-channel arrays.
/// \param[in] mv Single channel data.
/// \param[in] dst Output image data.
/// \par Example
/// \code
///     /* Assume p_rgb is a pointer that points to an image with shape (width, height, channel) */
///     LiteMat lite_mat_src(width, height, channel, (void *)p_rgb, LDataType::UINT8);
///     std::vector<LiteMat> lite_mat_dst;
///
///     /* Extract all channels of image */
///     Split(lite_mat_src, lite_mat_dst);
///
///     /* Merge all channels to an image */
///     LiteMat lite_mat_dst2;
///     Merge(lite_mat_dst, lite_mat_dst2);
/// \endcode
/// \return Return true if transform successfully.
bool DATASET_API Merge(const std::vector<LiteMat> &mv, LiteMat &dst);

/// \brief Apply affine transformation for 1 channel image.
/// \param[in] src Input image data.
/// \param[in] out_img Output image data.
/// \param[in] M[6] Affine transformation matrix.
/// \param[in] dsize The size of the output image.
/// \param[in] borderValue The pixel value is used for filing after the image is captured.
/// \par Example
/// \code
///     /* Assume p_rgb is a pointer that points to an image with shape (width, height) */
///     LiteMat lite_mat_src(width, height, (void *)p_rgb, LDataType::UINT8);
///     LiteMat lite_mat_src2;
///     ConvertRgbToGray(lite_mat_src, LDataType::UINT8, width, height, lite_mat_src2);
///
///     /* Define Affine matrix and apply */
///     LiteMat lite_mat_dst;
///     double M[6] = {1, 0, 0,
///                    0, 1, 0};
///     Affine(lite_mat_src2, lite_mat_dst, M, {width, height}, UINT8_C1(0));
///     std::cout << lite_mat_dst.width_ << " " << lite_mat_dst.height_ << " " << lite_mat_dst.channel_ << std::endl;
/// \endcode
/// \return Return true if transform successfully.
bool DATASET_API Affine(LiteMat &src, LiteMat &out_img, const double M[6], std::vector<size_t> dsize,
                        UINT8_C1 borderValue);

/// \brief Apply affine transformation for 3 channel image.
/// \param[in] src Input image data.
/// \param[in] out_img Output image data.
/// \param[in] M[6] Affine transformation matrix.
/// \param[in] dsize The size of the output image.
/// \param[in] borderValue The pixel value is used for filing after the image is captured.
/// \par Example
/// \code
///     /* Assume p_rgb is a pointer that points to an image with shape (width, height, channel) */
///     LiteMat lite_mat_src(width, height, channel, (void *)p_rgb, LDataType::UINT8);
///     LiteMat lite_mat_dst;
///
///     /* Define Affine matrix and apply */
///     double M[6] = {1, 0, 20,
///                    0, 1, 20};
///     Affine(lite_mat_src, lite_mat_dst, M, {image.cols, image.rows}, UINT8_C3(0, 0, 0));
///     std::cout << lite_mat_dst.width_ << " " << lite_mat_dst.height_ << " " << lite_mat_dst.channel_ << std::endl;
/// \endcode
/// \return Return true if transform successfully.
bool DATASET_API Affine(LiteMat &src, LiteMat &out_img, const double M[6], std::vector<size_t> dsize,
                        UINT8_C3 borderValue);

/// \brief Get default anchor boxes for Faster R-CNN, SSD, YOLO etc.
/// \param[in] config Objects of BoxesConfig structure.
std::vector<std::vector<float>> DATASET_API GetDefaultBoxes(const BoxesConfig &config);

/// \brief Convert the prediction boxes to the actual boxes of (y, x, h, w).
/// \param[in] boxes Actual size box.
/// \param[in] default_boxes Default box.
/// \param[in] config Objects of BoxesConfig structure.
void DATASET_API ConvertBoxes(std::vector<std::vector<float>> &boxes,
                              const std::vector<std::vector<float>> &default_boxes, const BoxesConfig &config);

/// \brief Apply Non-Maximum Suppression.
/// \param[in] all_boxes All input boxes.
/// \param[in] all_scores Score after all boxes are executed through the network.
/// \param[in] thres Pre-value of IOU.
/// \param[in] max_boxes Maximum value of output box.
/// \par Example
/// \code
///     /* Apply NMS on bboxes */
///     std::vector<std::vector<float>> all_boxes = {{1, 1, 2, 2}, {3, 3, 4, 4}, {5, 5, 6, 6}, {5, 5, 6, 6}};
///     std::vector<float> all_scores = {0.6, 0.5, 0.4, 0.9};
///     std::vector<int> keep = ApplyNms(all_boxes, all_scores, 0.5, 10);
/// \endcode
/// \return Remaining bounding boxes.
std::vector<int> DATASET_API ApplyNms(const std::vector<std::vector<float>> &all_boxes, std::vector<float> &all_scores,
                                      float thres, int max_boxes);

/// \brief affine image by linear.
/// \param[in] src Input image data.
/// \param[in] dst Output image data.
/// \param[in] M Transformation matrix
/// \param[in] dst_w The width of the output image.
/// \param[in] dst_h The height of the output image.
/// \param[in] borderType Edge processing type.
///   - PaddBorderType.PADD_BORDER_CONSTANT, fills the border with constant values.
///   - PaddBorderType.PADD_BORDER_REPLICATE, fills the border with replicate mode.
///   - PaddBorderType.PADD_BORDER_REFLECT_101, fills the border with reflect 101 mode.
///   - PaddBorderType.PADD_BORDER_DEFAULT, default pad mode, use reflect 101 mode.
/// \param[in] borderValue Boundary fill value.
/// \par Example
/// \code
///     /* Assume p_rgb is a pointer that points to an image with shape (width, height, channel) */
///     LiteMat lite_mat_src(width, height, channel, (void *)p_rgb, LDataType::UINT8);
///     LiteMat lite_mat_dst;
///
///     /* Define Affine matrix and apply */
///     double M[6] = {1, 0, 20,
///                    0, 1, 20};
///     LiteMat Matrix(3, 2, M, LDataType::DOUBLE);
///     std::vector<uint8_t> border_value = {0, 0, 0};
///     WarpAffineBilinear(lite_mat_src, lite_mat_dst, Matrix, width, height,
///                        PaddBorderType::PADD_BORDER_CONSTANT, border_value);
///     std::cout << lite_mat_dst.width_ << " " << lite_mat_dst.height_ << " " << lite_mat_dst.channel_ << std::endl;
/// \endcode
/// \return Return true if transform successfully.
bool DATASET_API WarpAffineBilinear(const LiteMat &src, LiteMat &dst, const LiteMat &M, int dst_w, int dst_h,
                                    PaddBorderType borderType, std::vector<uint8_t> &borderValue);

/// \brief affine image by linear.
/// \param[in] src Input image data.
/// \param[in] dst Output image data.
/// \param[in] M Transformation matrix
/// \param[in] dst_w The width of the output image.
/// \param[in] dst_h The height of the output image.
/// \param[in] borderType Edge processing type.
///   - PaddBorderType.PADD_BORDER_CONSTANT, fills the border with constant values.
///   - PaddBorderType.PADD_BORDER_REPLICATE, fills the border with replicate mode.
///   - PaddBorderType.PADD_BORDER_REFLECT_101, fills the border with reflect 101 mode.
///   - PaddBorderType.PADD_BORDER_DEFAULT, default pad mode, use reflect 101 mode.
/// \param[in] borderValue Boundary fill value.
/// \par Example
/// \code
///     /* Assume p_rgb is a pointer that points to an image with shape (width, height, channel) */
///     LiteMat lite_mat_src(width, height, channel, (void *)p_rgb, LDataType::UINT8);
///     LiteMat lite_mat_dst;
///
///     /* Get Perspective matrix and apply */
///     std::vector<Point> src = {Point(165, 270), Point(835, 270), Point(360, 125), Point(615, 125)};
///     std::vector<Point> dst = {Point(165, 270), Point(835, 270), Point(100, 100), Point(500, 30)};
///     LiteMat M;
///     GetPerspectiveTransform(src, dst, M);
///     std::vector<uint8_t> border_value = {0, 0, 0};
///     WarpPerspectiveBilinear(lite_mat_src, lite_mat_dst, M, width, height,
///                             PaddBorderType::PADD_BORDER_CONSTANT, border_value);
///     std::cout << lite_mat_dst.width_ << " " << lite_mat_dst.height_ << " " << lite_mat_dst.channel_ << std::endl;
/// \endcode
/// \return Return true if transform successfully.
bool DATASET_API WarpPerspectiveBilinear(const LiteMat &src, LiteMat &dst, const LiteMat &M, int dst_w, int dst_h,
                                         PaddBorderType borderType, std::vector<uint8_t> &borderValue);

/// \brief Matrix rotation.
/// \param[in] x The value of the x-axis of the coordinate rotation point.
/// \param[in] y The value of the y-axis of the coordinate rotation point.
/// \param[in] angle Rotation angle.
/// \param[in] scale Scaling ratio.
/// \param[in] M Output transformation matrix.
/// \par Example
/// \code
///     /* Get Rotation matrix */
///     double angle = 60.0;
///     double scale = 0.5;
///     LiteMat M;
///     GetRotationMatrix2D(1.0f, 2.0f, angle, scale, M);
///     std::cout << M.width_ << " " << M.height_ << " " << M.channel_ << std::endl;
/// \endcode
/// \return Return true if transform successfully.
bool DATASET_API GetRotationMatrix2D(float x, float y, double angle, double scale, LiteMat &M);

/// \brief Perspective transformation.
/// \param[in] src_point Input coordinate point.
/// \param[in] dst_point Output coordinate point.
/// \param[in] M Output matrix.
/// \par Example
/// \code
///     /* Get Perspective matrix */
///     std::vector<Point> src = {Point(165, 270), Point(835, 270), Point(360, 125), Point(615, 125)};
///     std::vector<Point> dst = {Point(165, 270), Point(835, 270), Point(100, 100), Point(500, 30)};
///     LiteMat M;
///     GetPerspectiveTransform(src, dst, M);
///     std::cout << M.width_ << " " << M.height_ << " " << M.channel_ << std::endl;
/// \endcode
/// \return Return true if transform successfully.
bool DATASET_API GetPerspectiveTransform(std::vector<Point> src_point, std::vector<Point> dst_point, LiteMat &M);

/// \brief Affine transformation.
/// \param[in] src_point Input coordinate point.
/// \param[in] dst_point Output coordinate point.
/// \param[in] M Output matrix.
/// \par Example
/// \code
///     /* Get Affine matrix */
///     std::vector<Point> src = {Point(50, 50), Point(200, 50), Point(50, 200)};
///     std::vector<Point> dst = {Point(40, 40), Point(100, 40), Point(50, 90)};
///     LiteMat M;
///     GetAffineTransform(src, dst, M);
///     std::cout << M.width_ << " " << M.height_ << " " << M.channel_ << std::endl;
/// \endcode
/// \return Return true if transform successfully.
bool DATASET_API GetAffineTransform(std::vector<Point> src_point, std::vector<Point> dst_point, LiteMat &M);

/// \brief Matrix transpose.
/// \param[in] src Input matrix.
/// \param[in] dst Output matrix.
/// \par Example
/// \code
///     /* Assume p_rgb is a pointer that points to an image with shape (width, height, channel) */
///     LiteMat lite_mat_src(width, height, channel, (void *)p_rgb, LDataType::UINT8);
///     LiteMat lite_mat_src2;
///     ConvertTo(lite_mat_src, lite_mat_src2);
///     LiteMat lite_mat_dst;
///
///     /* Transpose image */
///     Transpose(lite_mat_src2, lite_mat_dst);
///     std::cout << lite_mat_dst.width_ << " " << lite_mat_dst.height_ << " " << lite_mat_dst.channel_ << std::endl;
/// \endcode
/// \return Return true if transform successfully.
bool DATASET_API Transpose(const LiteMat &src, LiteMat &dst);

/// \brief Filter the image by a Gaussian kernel
/// \param[in] src LiteMat image to be processed. Only LiteMat of type UINT8 is supported now.
/// \param[in] dst LiteMat image after processing.
/// \param[in] ksize The size of Gaussian kernel. It should be a vector of size 2 as {kernel_x, kernel_y}, both value of
///     which should be positive and odd.
/// \param[in] sigmaX The Gaussian kernel standard deviation of width. It should be a positive value.
/// \param[in] sigmaY The Gaussian kernel standard deviation of height (default=0.f). It should be a positive value,
///     or will use the value of sigmaX.
/// \param[in] pad_type The padding type used while filtering (default=PaddBorderType::PADD_BORDER_DEFAULT).
///   - PaddBorderType.PADD_BORDER_CONSTANT, fills the border with constant values.
///   - PaddBorderType.PADD_BORDER_REPLICATE, fills the border with replicate mode.
///   - PaddBorderType.PADD_BORDER_REFLECT_101, fills the border with reflect 101 mode.
///   - PaddBorderType.PADD_BORDER_DEFAULT, default pad mode, use reflect 101 mode.
/// \par Example
/// \code
///     /* Assume p_rgb is a pointer that points to an image with shape (width, height, channel) */
///     LiteMat lite_mat_src(width, height, channel, (void *)p_rgb, LDataType::UINT8);
///     LiteMat lite_mat_dst;
///
///     /* Blur image */
///     GaussianBlur(lite_mat_src, lite_mat_dst, {3, 5}, 3, 3);
///     std::cout << lite_mat_dst.width_ << " " << lite_mat_dst.height_ << " " << lite_mat_dst.channel_ << std::endl;
/// \endcode
/// \return Return true if transform successfully.
bool DATASET_API GaussianBlur(const LiteMat &src, LiteMat &dst, const std::vector<int> &ksize, double sigmaX,
                              double sigmaY = 0.f, PaddBorderType pad_type = PaddBorderType::PADD_BORDER_DEFAULT);

/// \brief Detect edges in an image
/// \param[in] src LiteMat image to be processed. Only single channel LiteMat of type UINT8 is supported now.
/// \param[in] dst LiteMat image after processing.
/// \param[in] low_thresh The lower bound of the edge. Pixel with value below it will not be considered as a boundary.
///     It should be a nonnegative value.
//// \param[in] high_thresh The higher bound of the edge. Pixel with value over it will
/// be absolutely considered as a boundary. It should be a nonnegative value and no less than low_thresh.
/// \param[in] ksize The size of Sobel kernel (default=3). It can only be 3, 5 or 7.
/// \param[in] L2gradient Whether to use L2 distance while calculating gradient (default=false).
/// \par Example
/// \code
///     /* Assume p_rgb is a pointer that points to an image with shape (width, height, channel) */
///     LiteMat lite_mat_src;
///     InitFromPixel(p_rgb, LPixelType::RGB, LDataType::UINT8, width, height, lite_mat_src);
///     LiteMat lite_mat_src2;
///     ConvertRgbToGray(lite_mat_src, LDataType::UINT8, image.cols, image.rows, lite_mat_src2);
///
///     LiteMat lite_mat_dst;
///     Canny(lite_mat_src2, lite_mat_dst, 200, 300, 5);
///     std::cout << lite_mat_dst.width_ << " " << lite_mat_dst.height_ << " " << lite_mat_dst.channel_ << std::endl;
/// \endcode
/// \return Return true if transform successfully.
bool DATASET_API Canny(const LiteMat &src, LiteMat &dst, double low_thresh, double high_thresh, int ksize = 3,
                       bool L2gradient = false);

/// \brief Apply a 2D convolution over the image.
/// \param[in] src LiteMat image to be processed. Only LiteMat of type UINT8 and FLOAT32 is supported now.
/// \param[in] kernel LiteMat 2D convolution kernel. Only LiteMat of type FLOAT32 is supported now.
/// \param[in] dst LiteMat image after processing.
/// \param[in] dst_type Output data type of dst.
/// \param[in] pad_type The padding type used while filtering (default=PaddBorderType::PADD_BORDER_DEFAULT).
/// \par Example
/// \code
///     /* Assume p_rgb is a pointer that points to an image with shape (width, height, channel) */
///     LiteMat lite_mat_src(width, height, channel, (void *)p_rgb, LDataType::UINT8);
///     LiteMat lite_mat_dst;
///
///     LiteMat kernel;
///     kernel.Init(3, 3, 1, LDataType::FLOAT32);
///     float *kernel_ptr = kernel;
///     for (int i = 0; i < 9; i++) {
///         kernel_ptr[i] = i % 2;
///     }
///     Conv2D(lite_mat_src, kernel, lite_mat_dst, LDataType::UINT8);
///     std::cout << lite_mat_dst.width_ << " " << lite_mat_dst.height_ << " " << lite_mat_dst.channel_ << std::endl;
/// \endcode
/// \return Return true if transform successfully.
bool DATASET_API Conv2D(const LiteMat &src, const LiteMat &kernel, LiteMat &dst, LDataType dst_type,
                        PaddBorderType pad_type = PaddBorderType::PADD_BORDER_DEFAULT);

/// \brief Applies a separable linear convolution over the image
/// \param[in] src LiteMat image to be processed. Only LiteMat of type UINT8 and FLOAT32 is supported now.
/// \param[in] kx LiteMat 1D convolution kernel. Only LiteMat of type FLOAT32 is supported now.
/// \param[in] ky LiteMat 1D convolution kernel. Only LiteMat of type FLOAT32 is supported now.
/// \param[in] dst LiteMat image after processing.
/// \param[in] dst_type Output data type of dst.
/// \param[in] pad_type The padding type used while filtering (default=PaddBorderType::PADD_BORDER_DEFAULT).
bool DATASET_API ConvRowCol(const LiteMat &src, const LiteMat &kx, const LiteMat &ky, LiteMat &dst, LDataType dst_type,
                            PaddBorderType pad_type = PaddBorderType::PADD_BORDER_DEFAULT);

/// \brief Filter the image by a Sobel kernel
/// \param[in] src LiteMat image to be processed. Only LiteMat of type UINT8 is supported now.
/// \param[in] dst LiteMat image after processing.
/// \param[in] flag_x Order of the derivative x. It should be a nonnegative value and can not be equal to 0 at the same
///     time with flag_y.
/// \param[in] flag_y Order of the derivative y. It should be a nonnegative value and can not be equal
///     to 0 at the same time with flag_x.
/// \param[in] ksize The size of Sobel kernel (default=3). It can only be 1, 3, 5 or 7.
/// \param[in] scale The scale factor for the computed derivative values (default=1.0).
/// \param[in] pad_type The padding type used while filtering (default=PaddBorderType::PADD_BORDER_DEFAULT).
///   - PaddBorderType.PADD_BORDER_CONSTANT, fills the border with constant values.
///   - PaddBorderType.PADD_BORDER_REPLICATE, fills the border with replicate mode.
///   - PaddBorderType.PADD_BORDER_REFLECT_101, fills the border with reflect 101 mode.
///   - PaddBorderType.PADD_BORDER_DEFAULT, default pad mode, use reflect 101 mode.
/// \par Example
/// \code
///     /* Assume p_rgb is a pointer that points to an image with shape (width, height, channel) */
///     LiteMat lite_mat_src;
///     InitFromPixel(p_rgb, LPixelType::RGB, LDataType::UINT8, width, height, lite_mat_src);
///     LiteMat lite_mat_src2;
///     ConvertRgbToGray(lite_mat_src, LDataType::UINT8, image.cols, image.rows, lite_mat_src2);
///
///     LiteMat lite_mat_dst;
///     Sobel(lite_mat_src2, lite_mat_dst, 1, 0, 3, 1, PaddBorderType::PADD_BORDER_REPLICATE);
///     std::cout << lite_mat_dst.width_ << " " << lite_mat_dst.height_ << " " << lite_mat_dst.channel_ << std::endl;
/// \endcode
/// \return Return true if transform successfully.
bool DATASET_API Sobel(const LiteMat &src, LiteMat &dst, int flag_x, int flag_y, int ksize = 3, double scale = 1.0,
                       PaddBorderType pad_type = PaddBorderType::PADD_BORDER_DEFAULT);

/// \brief Convert RGB image or color image to BGR image.
/// \param[in] src Input image data.
/// \param[in] data_type The type of data (refer to LDataType class).
/// \param[in] w The width of output image.
/// \param[in] h The height of output image.
/// \param[in] mat Output image data.
/// \par Example
/// \code
///     /* Assume p_rgb is a pointer that points to an image with shape (width, height, channel) */
///     LiteMat lite_mat_src;
///     lite_mat_src.Init(width, height, channel, p_rgb, LDataType::UINT8);
///     LiteMat lite_mat_dst;
///
///     ConvertRgbToBgr(lite_mat_src, LDataType::UINT8, width, height, lite_mat_dst);
///     std::cout << lite_mat_dst.width_ << " " << lite_mat_dst.height_ << " " << lite_mat_dst.channel_ << std::endl;
/// \endcode
/// \return Return true if transform successfully.
bool DATASET_API ConvertRgbToBgr(const LiteMat &src, const LDataType &data_type, int w, int h, LiteMat &mat);

/// \brief Convert RGB image or color image to grayscale image.
/// \param[in] src Input image data.
/// \param[in] data_type The type of data (refer to LDataType class).
/// \param[in] w The width of output image.
/// \param[in] h The height of output image.
/// \param[in] mat Output image data.
/// \par Example
/// \code
///     /* Assume p_rgb is a pointer that points to an image with shape (width, height, channel) */
///     LiteMat lite_mat_src;
///     lite_mat_src.Init(width, height, channel, p_rgb, LDataType::UINT8);
///     LiteMat lite_mat_dst;
///
///     ConvertRgbToGray(lite_mat_src, LDataType::UINT8, width, height, lite_mat_dst);
///     std::cout << lite_mat_dst.width_ << " " << lite_mat_dst.height_ << " " << lite_mat_dst.channel_ << std::endl;
/// \endcode
/// \return Return true if transform successfully.
bool DATASET_API ConvertRgbToGray(const LiteMat &src, LDataType data_type, int w, int h, LiteMat &mat);

/// \brief Resize preserve AR with filler.
/// \param[in] src Input image data.
/// \param[in] dst Output image data.
/// \param[in] h The height of output image.
/// \param[in] w The width of output image.
/// \param[in] ratioShiftWShiftH Array that records the ratio, width shift, and height shift.
/// \param[in] invM Fixed direction array.
/// \param[in] img_orientation Way of export direction.
/// \par Example
/// \code
///     /* Assume p_rgb is a pointer that points to an image with shape (width, height, channel) */
///     LiteMat lite_mat_src;
///     lite_mat_src.Init(width, height, channel, p_rgb, LDataType::UINT8);
///     LiteMat lite_mat_dst;
///
///     float ratioShiftWShiftH[3] = {0};
///     float invM[2][3] = {{0, 0, 0}, {0, 0, 0}};
///     int h = 1000;
///     int w = 1000;
///     ResizePreserveARWithFiller(lite_mat_src, lite_mat_dst, h, w, &ratioShiftWShiftH, &invM, 0);
///     std::cout << lite_mat_dst.width_ << " " << lite_mat_dst.height_ << " " << lite_mat_dst.channel_ << std::endl;
/// \endcode
/// \return Return true if transform successfully.
bool DATASET_API ResizePreserveARWithFiller(LiteMat &src, LiteMat &dst, int h, int w, float (*ratioShiftWShiftH)[3],
                                            float (*invM)[2][3], int img_orientation);

/// \brief Transpose the input image; shape (H, W, C) to shape (C, H, W).
/// \param[in] src Input image data.
/// \param[in] dst Output image data.
/// \par Example
/// \code
///     /* Assume p_rgb is a pointer that points to an image with shape (width, height, channel) */
///     LiteMat lite_mat_src;
///     lite_mat_src.Init(width, height, channel, p_rgb, LDataType::UINT8);
///     LiteMat lite_mat_dst;
///
///     HWC2CHW(lite_mat_src, lite_mat_dst);
///     std::cout << lite_mat_dst.width_ << " " << lite_mat_dst.height_ << " " << lite_mat_dst.channel_ << std::endl;
/// \endcode
/// \return Return true if transform successfully.
bool DATASET_API HWC2CHW(LiteMat &src, LiteMat &dst);
}  // namespace mindspore::dataset
#endif  // IMAGE_PROCESS_H_
