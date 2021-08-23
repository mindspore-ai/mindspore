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

#include "tools/converter/preprocess/image_preprocess.h"
#include "src/common/log_adapter.h"
#include "include/errorcode.h"
namespace mindspore {
namespace lite {
namespace preprocess {
int ReadImage(const std::string &image_path, cv::Mat *image) {
  *image = cv::imread(image_path);
  if (image->empty() || image->data == nullptr) {
    MS_LOG(ERROR) << "missing file, improper permissions, unsupported or invalid format.";
    return RET_ERROR;
  }
  return RET_OK;
}

int ConvertImageFormat(cv::Mat *image, cv::ColorConversionCodes to_format) {
  if (to_format == cv::COLOR_COLORCVT_MAX) {
    MS_LOG(ERROR) << "to_format is cv::COLOR_COLORCVT_MAX";
    return RET_ERROR;
  }
  cv::cvtColor(*image, *image, to_format);
  return RET_OK;
}

int Normalize(cv::Mat *image, const std::vector<float> &mean, const std::vector<float> &std) {
  if (static_cast<int>(mean.size()) != image->channels() && static_cast<int>(std.size()) != image->channels()) {
    MS_LOG(ERROR) << "mean size:" << mean.size() << " != image->dims:" << image->dims << " or scale size:" << std.size()
                  << " !=image->dims:" << image->dims;
    return RET_ERROR;
  }
  std::vector<cv::Mat> channels(std.size());
  cv::split(*image, channels);
  for (size_t i = 0; i < channels.size(); i++) {
    channels[i].convertTo(channels[i], CV_32FC1, 1.0 / std[i], (0.0 - mean[i]) / std[i]);
  }
  cv::merge(channels, *image);
  return RET_OK;
}

int Resize(cv::Mat *image, int width, int height, cv::InterpolationFlags resize_method) {
  if (resize_method == cv::INTER_MAX) {
    MS_LOG(ERROR) << "resize method is cv::INTER_MAX";
    return RET_ERROR;
  }
  cv::resize(*image, *image, cv::Size(width, height), 0, 0, resize_method);
  return RET_OK;
}

int CenterCrop(cv::Mat *image, int width, int height) {
  if (width > image->cols || height > image->rows) {
    MS_LOG(ERROR) << "width:" << width << " > "
                  << "image->cols:" << image->cols << " or"
                  << "height:" << height << " > "
                  << "image->rows:" << image->rows;
    return RET_ERROR;
  }
  const int offsetW = (image->cols - width) / 2;
  const int offsetH = (image->rows - height) / 2;
  const cv::Rect roi(offsetW, offsetH, width, height);
  cv::Mat image_object = *(image);
  *(image) = image_object(roi).clone();
  return RET_OK;
}
}  // namespace preprocess
}  // namespace lite
}  // namespace mindspore
