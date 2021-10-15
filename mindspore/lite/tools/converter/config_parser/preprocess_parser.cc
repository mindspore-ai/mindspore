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

#include "tools/converter/config_parser/preprocess_parser.h"
#include <dirent.h>
#include <sys/stat.h>
#include <map>
#include <vector>
#include "tools/converter/preprocess/opencv_utils.h"
#include "src/common/log_adapter.h"
#include "mindspore/lite/tools/common/string_util.h"
#include "src/common/file_utils.h"
#include "include/errorcode.h"

namespace mindspore {
namespace lite {
namespace {
constexpr int kMinSize = 0;
constexpr int kMaxSize = 65535;
}  // namespace
int PreprocessParser::ParseInputType(const std::string &input_type_str, preprocess::InputType *input_type) {
  if (input_type_str == "IMAGE") {
    (*input_type) = preprocess::IMAGE;
  } else if (input_type_str == "BIN") {
    (*input_type) = preprocess::BIN;
  } else {
    (*input_type) = preprocess::INPUT_TYPE_MAX;
    MS_LOG(ERROR) << "INPUT ILLEGAL: input_type must be IMAGE|BIN.";
    return RET_INPUT_PARAM_INVALID;
  }
  return RET_OK;
}

int PreprocessParser::ParsePreprocess(const DataPreProcessString &data_pre_process_str,
                                      preprocess::DataPreProcessParam *data_pre_process) {
  int ret;
  if (!data_pre_process_str.calibrate_path.empty()) {
    ret = ParseCalibratePath(data_pre_process_str.calibrate_path, &data_pre_process->calibrate_path);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "Parse calibrate path failed.";
      return ret;
    }
  }

  if (!data_pre_process_str.calibrate_size.empty()) {
    if (!ConvertIntNum(data_pre_process_str.calibrate_size, &data_pre_process->calibrate_size)) {
      MS_LOG(ERROR) << "calibrate_size should be a valid number.";
      return RET_INPUT_PARAM_INVALID;
    }
    if (data_pre_process->calibrate_size < kMinSize) {
      MS_LOG(ERROR) << "calibrate_size must larger 0.";
      return RET_INPUT_PARAM_INVALID;
    }
  }

  if (!data_pre_process_str.input_type.empty()) {
    ret = ParseInputType(data_pre_process_str.input_type, &data_pre_process->input_type);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "input_type parse failed.";
      return ret;
    }
  }
  if (!data_pre_process_str.image_to_format.empty()) {
    ret =
      ParseImageToFormat(data_pre_process_str.image_to_format, &data_pre_process->image_pre_process.image_to_format);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "image preprocess parse failed.";
      return ret;
    }
    if (data_pre_process->image_pre_process.image_to_format == preprocess::RGB ||
        data_pre_process->image_pre_process.image_to_format == preprocess::GRAY) {
      data_pre_process->image_pre_process.image_to_format_code =
        preprocess::ConvertColorConversionCodes(data_pre_process->image_pre_process.image_to_format);
    }
  }
  ret = ParseImagePreProcess(data_pre_process_str, &data_pre_process->image_pre_process);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "image preprocess parse failed.";
    return ret;
  }

  ret = CollectCalibInputs(data_pre_process->calibrate_path, data_pre_process->calibrate_size,
                           &data_pre_process->calibrate_path_vector);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "collect calibrate inputs failed.";
    return ret;
  }
  return RET_OK;
}

int PreprocessParser::ParseCalibratePath(const std::string &str, std::map<std::string, std::string> *value) {
  if (value == nullptr) {
    MS_LOG(ERROR) << "value is nullptr.";
    return RET_ERROR;
  }
  auto key_values = SplitStringToVector(str, ',');
  for (const auto &key_value : key_values) {
    auto tmp = SplitStringToVector(key_value, ':');
    if (tmp.size() != 2) {
      MS_LOG(ERROR) << "vector need size = 2, size is " << tmp.size();
      return RET_INPUT_PARAM_INVALID;
    }
    auto data_path = RealPath(tmp.at(1).c_str());
    if (data_path.empty()) {
      MS_LOG(ERROR) << "path is invalid.";
      return RET_INPUT_PARAM_INVALID;
    }
    (*value)[tmp.at(0)] = data_path;
  }
  return RET_OK;
}

int PreprocessParser::ParseImagePreProcess(const DataPreProcessString &data_pre_process_str,
                                           preprocess::ImagePreProcessParam *image_pre_process) {
  auto ret = ParseImageNormalize(data_pre_process_str, image_pre_process);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Parse image normalize failed.";
    return ret;
  }
  ret = ParseImageCenterCrop(data_pre_process_str, image_pre_process);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Parse image center crop failed.";
    return ret;
  }
  ret = ParseImageResize(data_pre_process_str, image_pre_process);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Parse image resize failed.";
    return ret;
  }
  return RET_OK;
}

int PreprocessParser::ParseImageToFormat(const std::string &image_to_format_str,
                                         preprocess::ImageToFormat *image_to_format) {
  if (image_to_format_str == "RGB") {
    (*image_to_format) = preprocess::RGB;
  } else if (image_to_format_str == "GRAY") {
    (*image_to_format) = preprocess::GRAY;
  } else if (image_to_format_str == "BGR") {
    (*image_to_format) = preprocess::BGR;
  } else {
    (*image_to_format) = preprocess::IMAGE_TO_FORMAT_MAX;
    MS_LOG(ERROR) << "INPUT ILLEGAL: image_to_format must be RGB|GRAY|BGR.";
    return RET_INPUT_PARAM_INVALID;
  }
  return RET_OK;
}

int PreprocessParser::CollectCalibInputs(const std::map<std::string, std::string> &calibrate_data_path,
                                         size_t limited_count,
                                         std::map<std::string, std::vector<std::string>> *inputs) {
  if (inputs == nullptr) {
    MS_LOG(ERROR) << "inputs is null";
    return RET_ERROR;
  }

  auto AddImage = [&inputs](const std::string &file, const std::string &input_name) {
    struct stat buf {};
    if (stat(file.c_str(), &buf) == 0) {
      (*inputs)[input_name].push_back(file);
    } else {
      MS_LOG(WARNING) << "invalid image file path: " << file;
    }
  };

  for (const auto &image_path : calibrate_data_path) {
    DIR *root = opendir(image_path.second.c_str());
    if (root == nullptr) {
      MS_LOG(ERROR) << "invalid data path: " << image_path;
      return RET_PARAM_INVALID;
    }
    struct dirent *image_dir = readdir(root);
    size_t count = 0;
    while (image_dir != nullptr && count < limited_count) {
      std::string file_name(image_dir->d_name);
      if (file_name != "." && file_name != "..") {
        const std::string file_path = image_path.second + "/" + file_name;
        AddImage(file_path, image_path.first);
        count++;
      }
      image_dir = readdir(root);
    }
    auto ret = closedir(root);
    if (ret != 0) {
      MS_LOG(ERROR) << " close dir failed.";
      return RET_ERROR;
    }
    auto &cur_inputs = inputs->at(image_path.first);
    std::sort(cur_inputs.begin(), cur_inputs.end());
    if (count != limited_count) {
      MS_LOG(ERROR) << " data path: " << image_path.second << " data count:" << count
                    << " < limited_count:" << limited_count;
      return RET_ERROR;
    }
  }
  return RET_OK;
}
int PreprocessParser::ParseImageNormalize(const DataPreProcessString &data_pre_process_str,
                                          preprocess::ImagePreProcessParam *image_pre_process) {
  if (!data_pre_process_str.normalize_mean.empty() &&
      !ConvertDoubleVector(data_pre_process_str.normalize_mean, &image_pre_process->normalize_mean)) {
    MS_LOG(ERROR) << "Convert normalize_mean failed.";
    return RET_INPUT_PARAM_INVALID;
  }

  if (!data_pre_process_str.normalize_std.empty() &&
      !ConvertDoubleVector(data_pre_process_str.normalize_std, &image_pre_process->normalize_std)) {
    MS_LOG(ERROR) << "Convert normalize_std failed.";
    return RET_INPUT_PARAM_INVALID;
  }

  return RET_OK;
}
int PreprocessParser::ParseImageResize(const DataPreProcessString &data_pre_process_str,
                                       preprocess::ImagePreProcessParam *image_pre_process) {
  if (!data_pre_process_str.resize_width.empty()) {
    if (!ConvertIntNum(data_pre_process_str.resize_width, &image_pre_process->resize_width)) {
      MS_LOG(ERROR) << "resize_width should be a valid number.";
      return RET_INPUT_PARAM_INVALID;
    }
    if (image_pre_process->resize_width <= kMinSize || image_pre_process->resize_width > kMaxSize) {
      MS_LOG(ERROR) << "resize_width must be in [1, 65535].";
      return RET_INPUT_PARAM_INVALID;
    }
  }
  if (!data_pre_process_str.resize_height.empty()) {
    if (!ConvertIntNum(data_pre_process_str.resize_height, &image_pre_process->resize_height)) {
      MS_LOG(ERROR) << "resize_width should be a valid number.";
      return RET_INPUT_PARAM_INVALID;
    }
    if (image_pre_process->resize_height <= kMinSize || image_pre_process->resize_height > kMaxSize) {
      MS_LOG(ERROR) << "resize_height must be in [1, 65535].";
      return RET_INPUT_PARAM_INVALID;
    }
  }

  if (!data_pre_process_str.resize_method.empty()) {
    image_pre_process->resize_method = preprocess::ConvertResizeMethod(data_pre_process_str.resize_method);
    if (image_pre_process->resize_method == cv::INTER_MAX) {
      MS_LOG(ERROR) << "INPUT ILLEGAL: resize_method must be NEAREST|LINEAR|CUBIC.";
      return RET_INPUT_PARAM_INVALID;
    }
  }
  return RET_OK;
}
int PreprocessParser::ParseImageCenterCrop(const DataPreProcessString &data_pre_process_str,
                                           preprocess::ImagePreProcessParam *image_pre_process) {
  if (!data_pre_process_str.center_crop_width.empty()) {
    if (!ConvertIntNum(data_pre_process_str.center_crop_width, &image_pre_process->center_crop_width)) {
      MS_LOG(ERROR) << "center_crop_width should be a valid number.";
      return RET_INPUT_PARAM_INVALID;
    }
    if (image_pre_process->center_crop_width <= kMinSize || image_pre_process->center_crop_width > kMaxSize) {
      MS_LOG(ERROR) << "center_crop_width must be in [1, 65535].";
      return RET_INPUT_PARAM_INVALID;
    }
  }
  if (!data_pre_process_str.center_crop_height.empty()) {
    if (!ConvertIntNum(data_pre_process_str.center_crop_height, &image_pre_process->center_crop_height)) {
      MS_LOG(ERROR) << "center_crop_height should be a valid number.";
      return RET_INPUT_PARAM_INVALID;
    }
    if (image_pre_process->center_crop_height <= kMinSize || image_pre_process->center_crop_height > kMaxSize) {
      MS_LOG(ERROR) << "center_crop_height must be in [1, 65535].";
      return RET_INPUT_PARAM_INVALID;
    }
  }
  return RET_OK;
}
}  // namespace lite
}  // namespace mindspore
