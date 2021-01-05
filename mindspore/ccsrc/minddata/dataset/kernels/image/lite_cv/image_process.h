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

#define INT16_CAST(X) \
  static_cast<int16_t>(::std::min(::std::max(static_cast<int>(X + (X >= 0.f ? 0.5f : -0.5f)), -32768), 32767));

enum PaddBorderType { PADD_BORDER_CONSTANT = 0, PADD_BORDER_REPLICATE = 1 };

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
         uint8_t fill_b_or_gray, uint8_t fill_g, uint8_t fill_r);

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

}  // namespace dataset
}  // namespace mindspore
#endif  // IMAGE_PROCESS_H_
