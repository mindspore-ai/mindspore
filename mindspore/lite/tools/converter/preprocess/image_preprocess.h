/**
 * Copyright 2021-2023 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_LITE_TOOLS_CONVERTER_PREPROCESS_IMAGE_PREPROCESS_H_
#define MINDSPORE_LITE_TOOLS_CONVERTER_PREPROCESS_IMAGE_PREPROCESS_H_

#include <vector>
#include <string>
#ifdef MSLITE_DEPS_OPENCV
#include <opencv2/opencv.hpp>
#endif
#include "tools/converter/preprocess/preprocess_param.h"
#include "include/api/model.h"
#include "src/tensor.h"

namespace mindspore {
namespace lite {
namespace preprocess {
#ifdef MSLITE_DEPS_OPENCV
int ReadImage(const std::string &image_path, cv::Mat *image);

int DecodeBuffer(const unsigned char *buffer, int length, cv::Mat *image);

int ConvertImageFormat(cv::Mat *image, cv::ColorConversionCodes to_format);

int Normalize(cv::Mat *image, const std::vector<double> &mean, const std::vector<double> &standard_deviation);

int Resize(cv::Mat *image, int width, int height, cv::InterpolationFlags resize_method);

int CenterCrop(cv::Mat *image, int width, int height);

int ImagePreProcess(const ImagePreProcessParam &image_preprocess_param, cv::Mat *image, void **data, size_t *size);
#endif

// NOTE:`data` must be use delete[] to free buffer.
int PreProcess(const DataPreProcessParam &data_pre_process_param, const std::string &input_name, size_t image_index,
               void **data, size_t *size);

int PreProcess(const preprocess::DataPreProcessParam &data_pre_process_param, const std::string &input_name,
               size_t image_index, mindspore::MSTensor *tensor);

int PreProcessBatch(const preprocess::DataPreProcessParam &data_pre_process_param, const std::string &input_name,
                    lite::Tensor *tensor);

}  // namespace preprocess
}  // namespace lite
}  // namespace mindspore
#endif  // MINDSPORE_LITE_TOOLS_CONVERTER_PREPROCESS_IMAGE_PREPROCESS_H_
