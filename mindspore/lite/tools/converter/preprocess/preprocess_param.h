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
#ifndef MINDSPORE_LITE_TOOLS_CONVERTER_PREPROCESS_PREPROCESS_PARAM_H
#define MINDSPORE_LITE_TOOLS_CONVERTER_PREPROCESS_PREPROCESS_PARAM_H
#include <string>
#include <vector>
#include <map>
#include <opencv2/opencv.hpp>
namespace mindspore {
namespace lite {
namespace preprocess {
enum InputType { IMAGE, BIN, INPUT_TYPE_MAX };
enum ImageToFormat { RGB, GRAY, BGR, IMAGE_TO_FORMAT_MAX };

struct ImagePreProcessParam {
  ImageToFormat image_to_format = IMAGE_TO_FORMAT_MAX;
  cv::ColorConversionCodes image_to_format_code = cv::COLOR_COLORCVT_MAX;
  std::vector<double> normalize_mean;
  std::vector<double> normalize_std;
  int resize_width = -1;
  int resize_height = -1;
  cv::InterpolationFlags resize_method = cv::INTER_MAX;
  int center_crop_width = -1;
  int center_crop_height = -1;
};

struct DataPreProcessParam {
  std::map<std::string, std::string> calibrate_path;
  std::map<std::string, std::vector<std::string>> calibrate_path_vector;
  int calibrate_size = 0;
  InputType input_type = INPUT_TYPE_MAX;
  ImagePreProcessParam image_pre_process;
};
}  // namespace preprocess
}  // namespace lite
}  // namespace mindspore
#endif  // MINDSPORE_LITE_TOOLS_CONVERTER_PREPROCESS_PREPROCESS_PARAM_H
