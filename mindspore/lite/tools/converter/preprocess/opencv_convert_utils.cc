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

#include "tools/converter/preprocess/opencv_convert_utils.h"
#include "src/common/log_adapter.h"
#include "include/errorcode.h"
namespace mindspore {
namespace lite {
namespace preprocess {
cv::ColorConversionCodes ConvertColorConversionCodes(const std::string &format) {
  if (format == "RGB") {
    return cv::COLOR_BGR2RGB;
  } else if (format == "GRAY") {
    return cv::COLOR_BGR2GRAY;
  } else {
    MS_LOG(ERROR) << "Unsupported format:" << format;
    return cv::COLOR_COLORCVT_MAX;
  }
}

cv::InterpolationFlags ConvertResizeFlag(const std::string &flag) {
  if (flag == "NEAREST") {
    return cv::INTER_NEAREST;
  } else if (flag == "LINEAR") {
    return cv::INTER_LINEAR;
  } else if (flag == "CUBIC") {
    return cv::INTER_CUBIC;
  } else {
    MS_LOG(ERROR) << "Unsupported resize method:" << flag;
    return cv::INTER_MAX;
  }
}
}  // namespace preprocess
}  // namespace lite
}  // namespace mindspore
