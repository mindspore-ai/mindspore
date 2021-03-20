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

#define INT16_CAST(X) \
  static_cast<int16_t>(::std::min(::std::max(static_cast<int>(X + (X >= 0.f ? 0.5f : -0.5f)), -32768), 32767));

enum PaddBorderType {
  PADD_BORDER_CONSTANT = 0,
  PADD_BORDER_REPLICATE = 1,
  PADD_BORDER_REFLECT_101 = 4,
  PADD_BORDER_DEFAULT = PADD_BORDER_REFLECT_101
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
///          the channel of currently supports is 3 and 1
bool ResizeBilinear(const LiteMat &src, LiteMat &dst, int dst_w, int dst_h);

/// \brief Init Lite Mat from pixel, the conversion of currently supports is rbgaTorgb and rgbaTobgr
bool InitFromPixel(const unsigned char *data, LPixelType pixel_type, LDataType data_type, int w, int h, LiteMat &m);

/// \brief convert the data type, the conversion of currently supports is uint8 to float
bool ConvertTo(const LiteMat &src, LiteMat &dst, double scale = 1.0);

/// \brief crop image, the channel supports is 3 and 1
bool Crop(const LiteMat &src, LiteMat &dst, int x, int y, int w, int h);

/// \brief normalize image, currently the supports data type is float
bool SubStractMeanNormalize(const LiteMat &src, LiteMat &dst, const std::vector<float> &mean,
                            const std::vector<float> &std);

/// \brief padd image, the channel supports is 3 and 1
bool Pad(const LiteMat &src, LiteMat &dst, int top, int bottom, int left, int right, PaddBorderType pad_type,
         uint8_t fill_b_or_gray = 0, uint8_t fill_g = 0, uint8_t fill_r = 0);

/// \brief Extract image channel by index
bool ExtractChannel(LiteMat &src, LiteMat &dst, int col);

/// \brief Split image channels to single channel
bool Split(const LiteMat &src, std::vector<LiteMat> &mv);

/// \brief Create a multi-channel image out of several single-channel arrays.
bool Merge(const std::vector<LiteMat> &mv, LiteMat &dst);

/// \brief Apply affine transformation for 1 channel image
bool Affine(LiteMat &src, LiteMat &out_img, const double M[6], std::vector<size_t> dsize, UINT8_C1 borderValue);

/// \brief Apply affine transformation for 3 channel image
bool Affine(LiteMat &src, LiteMat &out_img, const double M[6], std::vector<size_t> dsize, UINT8_C3 borderValue);

/// \brief Get default anchor boxes for Faster R-CNN, SSD, YOLO etc
std::vector<std::vector<float>> GetDefaultBoxes(const BoxesConfig config);

/// \brief Convert the prediction boxes to the actual boxes of (y, x, h, w)
void ConvertBoxes(std::vector<std::vector<float>> &boxes, const std::vector<std::vector<float>> &default_boxes,
                  const BoxesConfig config);

/// \brief Apply Non-Maximum Suppression
std::vector<int> ApplyNms(const std::vector<std::vector<float>> &all_boxes, std::vector<float> &all_scores, float thres,
                          int max_boxes);

/// \brief affine image by linear
bool WarpAffineBilinear(const LiteMat &src, LiteMat &dst, const LiteMat &M, int dst_w, int dst_h,
                        PaddBorderType borderType, std::vector<uint8_t> &borderValue);

/// \brief perspective image by linear
bool WarpPerspectiveBilinear(const LiteMat &src, LiteMat &dst, const LiteMat &M, int dst_w, int dst_h,
                             PaddBorderType borderType, std::vector<uint8_t> &borderValue);

/// \brief Matrix rotation
bool GetRotationMatrix2D(float x, float y, double angle, double scale, LiteMat &M);

/// \brief Perspective transformation
bool GetPerspectiveTransform(std::vector<Point> src_point, std::vector<Point> dst_point, LiteMat &M);

/// \brief Affine transformation
bool GetAffineTransform(std::vector<Point> src_point, std::vector<Point> dst_point, LiteMat &M);

/// \brief Matrix transpose
bool Transpose(LiteMat &src, LiteMat &dst);

/// \brief Filter the image by a Gaussian kernel
bool GaussianBlur(const LiteMat &src, LiteMat &dst, const std::vector<int> &ksize, double sigmaX, double sigmaY = 0.f,
                  PaddBorderType pad_type = PaddBorderType::PADD_BORDER_DEFAULT);

/// \brief Detect edges in an image
bool Canny(const LiteMat &src, LiteMat &dst, double low_thresh, double high_thresh, int ksize = 3,
           bool L2gradient = false);

/// \brief Apply a 2D convolution over the image
bool Conv2D(const LiteMat &src, const LiteMat &kernel, LiteMat &dst, LDataType dst_type,
            PaddBorderType pad_type = PaddBorderType::PADD_BORDER_DEFAULT);

/// \brief Applies a separable linear convolution over the image
bool ConvRowCol(const LiteMat &src, const LiteMat &kx, const LiteMat &ky, LiteMat &dst, LDataType dst_type,
                PaddBorderType pad_type = PaddBorderType::PADD_BORDER_DEFAULT);

/// \brief Filter the image by a Sobel kernel
bool Sobel(const LiteMat &src, LiteMat &dst, int flag_x, int flag_y, int ksize = 3, double scale = 1.0,
           PaddBorderType pad_type = PaddBorderType::PADD_BORDER_DEFAULT);

/// \brief Convert RGB image or color image to grayscale image
bool ConvertRgbToGray(const LiteMat &src, LDataType data_type, int w, int h, LiteMat &mat);

/// \brief Resize preserve AR with filler
bool ResizePreserveARWithFiller(LiteMat &src, LiteMat &dst, int h, int w, float (*ratioShiftWShiftH)[3],
                                float (*invM)[2][3], int img_orientation);

}  // namespace dataset
}  // namespace mindspore
#endif  // IMAGE_PROCESS_H_
