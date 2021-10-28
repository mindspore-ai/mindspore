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

#include "tools/converter/preprocess/opencv_utils.h"
#include <vector>
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

cv::ColorConversionCodes ConvertColorConversionCodes(preprocess::ImageToFormat format) {
  if (format == RGB) {
    return cv::COLOR_BGR2RGB;
  } else if (format == GRAY) {
    return cv::COLOR_BGR2GRAY;
  } else {
    MS_LOG(ERROR) << "Unsupported format:" << format;
    return cv::COLOR_COLORCVT_MAX;
  }
}

cv::InterpolationFlags ConvertResizeMethod(const std::string &method) {
  if (method == "NEAREST") {
    return cv::INTER_NEAREST;
  } else if (method == "LINEAR") {
    return cv::INTER_LINEAR;
  } else if (method == "CUBIC") {
    return cv::INTER_CUBIC;
  } else {
    MS_LOG(ERROR) << "INPUT ILLEGAL: resize_method must be NEAREST|LINEAR|CUBIC.";
    return cv::INTER_MAX;
  }
}

int GetMatData(const cv::Mat &mat, void **data, size_t *size) {
  if (data == nullptr || size == nullptr) {
    MS_LOG(ERROR) << "data or size is nullptr.";
    return RET_NULL_PTR;
  }
  cv::Mat mat_local = mat;
  // if the input Mat's memory is not continuous, copy it to one block of memory
  if (!mat.isContinuous()) {
    mat_local = mat.clone();
  }
  (*size) = 0;
  for (int i = 0; i < mat.rows; ++i) {
    (*size) += static_cast<size_t>(mat.cols) * mat.elemSize();
  }

  (*data) = new char[*size];
  if (memcpy_s(*data, *size, mat_local.data,
               static_cast<size_t>(mat.rows * mat.cols * mat.channels()) * sizeof(float)) != EOK) {
    MS_LOG(ERROR) << "memcpy failed.";
    return RET_ERROR;
  }
  return RET_OK;
}
}  // namespace preprocess
}  // namespace lite
}  // namespace mindspore
