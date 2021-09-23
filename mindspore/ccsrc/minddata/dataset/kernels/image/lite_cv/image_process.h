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

#ifndef IMAGE_PROCESS_H_
#define IMAGE_PROCESS_H_

#include <math.h>
#include <vector>
#include <algorithm>
#include <iostream>

#include "lite_cv/lite_mat.h"

namespace mindspore {
namespace dataset {

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
  static_cast<int16_t>(::std::min(::std::max(static_cast<int>(X + (X >= 0.f ? 0.5f : -0.5f)), -32768), 32767));

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
bool ResizeBilinear(const LiteMat &src, LiteMat &dst, int dst_w, int dst_h);

/// \brief Init Lite Mat from pixel, the conversion of currently supports is rbgaTorgb and rgbaTobgr.
/// \note The length of the pointer must be the same as that of the multiplication of w and h.
/// \param[in] data Input image data.
/// \param[in] pixel_type The type of pixel_type.
/// \param[in] data_type The type of data_type.
/// \param[in] w The width of the output image.
/// \param[in] h The length of the output image.
/// \param[in] m Used to store image data.
bool InitFromPixel(const unsigned char *data, LPixelType pixel_type, LDataType data_type, int w, int h, LiteMat &m);

/// \brief convert the data type, the conversion of currently supports is uint8 to float.
/// \param[in] src Input image data.
/// \param[in] dst Output image data.
/// \param[in] scale Scale pixel value(default:1.0).
bool ConvertTo(const LiteMat &src, LiteMat &dst, double scale = 1.0);

/// \brief crop image, the channel supports is 3 and 1.
/// \param[in] src Input image data.
/// \param[in] dst Output image data.
/// \param[in] x The x coordinate value of the starting point of the screenshot.
/// \param[in] y The y coordinate value of the starting point of the screenshot.
/// \param[in] w The width of the screenshot.
/// \param[in] h The height of the screenshot.
bool Crop(const LiteMat &src, LiteMat &dst, int x, int y, int w, int h);

/// \brief normalize image, currently the supports data type is float.
/// \param[in] src Input image data.
/// \param[in] dst Output image data.
/// \param[in] mean Mean of the data set.
/// \param[in] std Norm of the data set.
bool SubStractMeanNormalize(const LiteMat &src, LiteMat &dst, const std::vector<float> &mean,
                            const std::vector<float> &std);

/// \brief padd image, the channel supports is 3 and 1.
/// \param[in] src Input image data.
/// \param[in] dst Output image data.
/// \param[in] top The length of top.
/// \param[in] bottom The length of bottom.
/// \param[in] left The length of left.
/// \param[in] right he length of right.
/// \param[in] pad_type The type of pad.
/// \param[in] fill_b_or_gray B or GRAY.
/// \param[in] fill_g G.
/// \param[in] fill_r R.
bool Pad(const LiteMat &src, LiteMat &dst, int top, int bottom, int left, int right, PaddBorderType pad_type,
         uint8_t fill_b_or_gray = 0, uint8_t fill_g = 0, uint8_t fill_r = 0);

/// \brief Extract image channel by index.
/// \param[in] src Input image data.
/// \param[in] dst Output image data.
/// \param[in] col The serial number of the channel.
bool ExtractChannel(LiteMat &src, LiteMat &dst, int col);

/// \brief Split image channels to single channel.
/// \param[in] src Input image data.
/// \param[in] mv Single channel data.
bool Split(const LiteMat &src, std::vector<LiteMat> &mv);

/// \brief Create a multi-channel image out of several single-channel arrays.
/// \param[in] mv Single channel data.
/// \param[in] dst Output image data.
bool Merge(const std::vector<LiteMat> &mv, LiteMat &dst);

/// \brief Apply affine transformation for 1 channel image.
/// \param[in] src Input image data.
/// \param[in] out_img Output image data.
/// \param[in] M[6] Affine transformation matrix.
/// \param[in] dsize The size of the output image.
/// \param[in] borderValue The pixel value is used for filing after the image is captured.
bool Affine(LiteMat &src, LiteMat &out_img, const double M[6], std::vector<size_t> dsize, UINT8_C1 borderValue);

/// \brief Apply affine transformation for 3 channel image.
/// \param[in] src Input image data.
/// \param[in] out_img Output image data.
/// \param[in] M[6] Affine transformation matrix.
/// \param[in] dsize The size of the output image.
/// \param[in] borderValue The pixel value is used for filing after the image is captured.
bool Affine(LiteMat &src, LiteMat &out_img, const double M[6], std::vector<size_t> dsize, UINT8_C3 borderValue);

/// \brief Get default anchor boxes for Faster R-CNN, SSD, YOLO etc.
/// \param[in] config Objects of BoxesConfig structure.
std::vector<std::vector<float>> GetDefaultBoxes(const BoxesConfig config);

/// \brief Convert the prediction boxes to the actual boxes of (y, x, h, w).
/// \param[in] boxes Actual size box.
/// \param[in] default_boxes Default box.
/// \param[in] config Objects of BoxesConfig structure.
void ConvertBoxes(std::vector<std::vector<float>> &boxes, const std::vector<std::vector<float>> &default_boxes,
                  const BoxesConfig config);

/// \brief Apply Non-Maximum Suppression.
/// \param[in] all_boxes All input boxes.
/// \param[in] all_scores Score after all boxes are executed through the network.
/// \param[in] thres Pre-value of IOU.
/// \param[in] max_boxes Maximum value of output box.
std::vector<int> ApplyNms(const std::vector<std::vector<float>> &all_boxes, std::vector<float> &all_scores, float thres,
                          int max_boxes);

/// \brief affine image by linear.
/// \param[in] src Input image data.
/// \param[in] dst Output image data.
/// \param[in] M Transformation matrix
/// \param[in] dst_w The width of the output image.
/// \param[in] dst_h The height of the output image.
/// \param[in] borderType Edge processing type.
/// \param[in] borderValue Boundary fill value.
bool WarpAffineBilinear(const LiteMat &src, LiteMat &dst, const LiteMat &M, int dst_w, int dst_h,
                        PaddBorderType borderType, std::vector<uint8_t> &borderValue);

/// \brief affine image by linear.
/// \param[in] src Input image data.
/// \param[in] dst Output image data.
/// \param[in] M Transformation matrix
/// \param[in] dst_w The width of the output image.
/// \param[in] dst_h The height of the output image.
/// \param[in] borderType Edge processing type.
/// \param[in] borderValue Boundary fill value.
bool WarpPerspectiveBilinear(const LiteMat &src, LiteMat &dst, const LiteMat &M, int dst_w, int dst_h,
                             PaddBorderType borderType, std::vector<uint8_t> &borderValue);

/// \brief Matrix rotation.
/// \param[in] x The value of the x-axis of the coordinate rotation point.
/// \param[in] y The value of the y-axis of the coordinate rotation point.
/// \param[in] angle Rotation angle.
/// \param[in] scale Scaling ratio.
/// \param[in] M Output transformation matrix.
bool GetRotationMatrix2D(float x, float y, double angle, double scale, LiteMat &M);

/// \brief Perspective transformation.
/// \param[in] src_point Input coordinate point.
/// \param[in] dst_point Output coordinate point.
/// \param[in] M Output matrix.
bool GetPerspectiveTransform(std::vector<Point> src_point, std::vector<Point> dst_point, LiteMat &M);

/// \brief Affine transformation.
/// \param[in] src_point Input coordinate point.
/// \param[in] dst_point Output coordinate point.
/// \param[in] M Output matrix.
bool GetAffineTransform(std::vector<Point> src_point, std::vector<Point> dst_point, LiteMat &M);

/// \brief Matrix transpose.
/// \param[in] src Input matrix.
/// \param[in] dst Output matrix.
bool Transpose(const LiteMat &src, LiteMat &dst);

/// \brief Filter the image by a Gaussian kernel
/// \param[in] src LiteMat image to be processed. Only LiteMat of type UINT8 is supported now.
/// \param[in] dst LiteMat image after processing.
/// \param[in] ksize The size of Gaussian kernel. It should be a vector of size 2 as {kernel_x, kernel_y}, both value of
///     which should be positive and odd.
/// \param[in] sigmaX The Gaussian kernel standard deviation of width. It should be a positive value.
/// \param[in] sigmaY The Gaussian kernel standard deviation of height (default=0.f). It should be a positive value,
///     or will use the value of sigmaX.
/// \param[in] pad_type The padding type used while filtering (default=PaddBorderType::PADD_BORDER_DEFAULT).
bool GaussianBlur(const LiteMat &src, LiteMat &dst, const std::vector<int> &ksize, double sigmaX, double sigmaY = 0.f,
                  PaddBorderType pad_type = PaddBorderType::PADD_BORDER_DEFAULT);

/// \brief Detect edges in an image
/// \param[in] src LiteMat image to be processed. Only single channel LiteMat of type UINT8 is supported now.
/// \param[in] dst LiteMat image after processing.
/// \param[in] low_thresh The lower bound of the edge. Pixel with value below it will not be considered as a boundary.
///     It should be a nonnegative value.
//// \param[in] high_thresh The higher bound of the edge. Pixel with value over it will
/// be absolutely considered as a boundary. It should be a nonnegative value and no less than low_thresh.
/// \param[in] ksize The size of Sobel kernel (default=3). It can only be 3, 5 or 7.
/// \param[in] L2gradient Whether to use L2 distance while calculating gradient (default=false).
bool Canny(const LiteMat &src, LiteMat &dst, double low_thresh, double high_thresh, int ksize = 3,
           bool L2gradient = false);

/// \brief Apply a 2D convolution over the image.
/// \param[in] src LiteMat image to be processed. Only LiteMat of type UINT8 and FLOAT32 is supported now.
/// \param[in] kernel LiteMat 2D convolution kernel. Only LiteMat of type FLOAT32 is supported now.
/// \param[in] dst LiteMat image after processing.
/// \param[in] dst_type Output data type of dst.
/// \param[in] pad_type The padding type used while filtering (default=PaddBorderType::PADD_BORDER_DEFAULT).
bool Conv2D(const LiteMat &src, const LiteMat &kernel, LiteMat &dst, LDataType dst_type,
            PaddBorderType pad_type = PaddBorderType::PADD_BORDER_DEFAULT);

/// \brief Applies a separable linear convolution over the image
/// \param[in] src LiteMat image to be processed. Only LiteMat of type UINT8 and FLOAT32 is supported now.
/// \param[in] kx LiteMat 1D convolution kernel. Only LiteMat of type FLOAT32 is supported now.
/// \param[in] ky LiteMat 1D convolution kernel. Only LiteMat of type FLOAT32 is supported now.
/// \param[in] dst LiteMat image after processing.
/// \param[in] dst_type Output data type of dst.
/// \param[in] pad_type The padding type used while filtering (default=PaddBorderType::PADD_BORDER_DEFAULT).
bool ConvRowCol(const LiteMat &src, const LiteMat &kx, const LiteMat &ky, LiteMat &dst, LDataType dst_type,
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
bool Sobel(const LiteMat &src, LiteMat &dst, int flag_x, int flag_y, int ksize = 3, double scale = 1.0,
           PaddBorderType pad_type = PaddBorderType::PADD_BORDER_DEFAULT);

/// \brief Convert RGB image or color image to BGR image.
/// \param[in] src Input image data.
/// \param[in] data_type The type of data_type.
/// \param[in] w The width of output image.
/// \param[in] h The height of output image.
/// \param[in] mat Output image data.
bool ConvertRgbToBgr(const LiteMat &src, const LDataType &data_type, int w, int h, LiteMat &mat);

/// \brief Convert RGB image or color image to grayscale image.
/// \param[in] src Input image data.
/// \param[in] data_type The type of data_type.
/// \param[in] w The width of output image.
/// \param[in] h The height of output image.
/// \param[in] mat Output image data.
bool ConvertRgbToGray(const LiteMat &src, LDataType data_type, int w, int h, LiteMat &mat);

/// \brief Resize preserve AR with filler.
/// \param[in] src Input image data.
/// \param[in] dst Output image data.
/// \param[in] h The height of output image.
/// \param[in] w The width of output image.
/// \param[in] ratioShiftWShiftH Array that records the ratio, width shift, and height shift.
/// \param[in] invM Fixed direction array.
/// \param[in] img_orientation Way of export direction.
bool ResizePreserveARWithFiller(LiteMat &src, LiteMat &dst, int h, int w, float (*ratioShiftWShiftH)[3],
                                float (*invM)[2][3], int img_orientation);

}  // namespace dataset
}  // namespace mindspore
#endif  // IMAGE_PROCESS_H_
