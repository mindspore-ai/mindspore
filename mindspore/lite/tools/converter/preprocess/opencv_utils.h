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
#ifndef MINDSPORE_LITE_TOOLS_CONVERTER_PREPROCESS_OPENCV_UTILS_H
#define MINDSPORE_LITE_TOOLS_CONVERTER_PREPROCESS_OPENCV_UTILS_H
#include <string>
#include <opencv2/opencv.hpp>
#include "tools/converter/preprocess/preprocess_param.h"

namespace mindspore {
namespace lite {
namespace preprocess {
cv::ColorConversionCodes ConvertColorConversionCodes(const std::string &format);

cv::ColorConversionCodes ConvertColorConversionCodes(preprocess::ImageToFormat format);

cv::InterpolationFlags ConvertResizeMethod(const std::string &method);

int GetMatData(const cv::Mat &mat, void **data, size_t *size);
}  // namespace preprocess
}  // namespace lite
}  // namespace mindspore
#endif  // MINDSPORE_LITE_TOOLS_CONVERTER_PREPROCESS_OPENCV_UTILS_H
