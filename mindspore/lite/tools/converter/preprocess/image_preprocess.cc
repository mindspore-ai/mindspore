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
#include "src/common/file_utils.h"
#include "include/errorcode.h"
#include "tools/converter/preprocess/opencv_utils.h"
namespace mindspore {
namespace lite {
namespace preprocess {
int ReadImage(const std::string &image_path, cv::Mat *image) {
  if (image == nullptr) {
    MS_LOG(ERROR) << "image is nullptr.";
    return RET_ERROR;
  }
  *image = cv::imread(image_path);
  if (image->empty() || image->data == nullptr) {
    MS_LOG(ERROR) << "missing file, improper permissions, unsupported or invalid format.";
    return RET_ERROR;
  }
  image->convertTo(*image, CV_32FC3);
  return RET_OK;
}

int ConvertImageFormat(cv::Mat *image, cv::ColorConversionCodes to_format) {
  if (image == nullptr) {
    MS_LOG(ERROR) << "image is nullptr.";
    return RET_ERROR;
  }
  if (to_format == cv::COLOR_COLORCVT_MAX) {
    MS_LOG(ERROR) << "to_format is cv::COLOR_COLORCVT_MAX";
    return RET_ERROR;
  }
  cv::cvtColor(*image, *image, to_format);
  if (to_format == cv::COLOR_BGR2GRAY) {
    image->convertTo(*image, CV_32FC1);
  }
  return RET_OK;
}

int Normalize(cv::Mat *image, const std::vector<double> &mean, const std::vector<double> &standard_deviation) {
  if (image == nullptr) {
    MS_LOG(ERROR) << "image is nullptr.";
    return RET_ERROR;
  }
  if (static_cast<int>(mean.size()) != image->channels() ||
      static_cast<int>(standard_deviation.size()) != image->channels()) {
    MS_LOG(ERROR) << "mean size:" << mean.size() << " != image->channels:" << image->channels()
                  << " or scale size:" << standard_deviation.size() << " !=image->dims:" << image->channels();
    return RET_ERROR;
  }
  std::vector<cv::Mat> channels(image->channels());
  cv::split(*image, channels);
  for (size_t i = 0; i < channels.size(); i++) {
    channels[i].convertTo(channels[i], CV_32FC1, 1.0 / standard_deviation[i], (0.0 - mean[i]) / standard_deviation[i]);
  }
  cv::merge(channels, *image);
  return RET_OK;
}

int Resize(cv::Mat *image, int width, int height, cv::InterpolationFlags resize_method) {
  if (image == nullptr) {
    MS_LOG(ERROR) << "image is nullptr.";
    return RET_ERROR;
  }
  if (width <= 0 || height <= 0) {
    MS_LOG(ERROR) << "Both width and height must be > 0."
                  << " width:" << width << " height:" << height;
    return RET_ERROR;
  }
  if (resize_method == cv::INTER_MAX) {
    MS_LOG(ERROR) << "resize method is cv::INTER_MAX";
    return RET_ERROR;
  }
  cv::resize(*image, *image, cv::Size(width, height), 0, 0, resize_method);
  return RET_OK;
}

int CenterCrop(cv::Mat *image, int width, int height) {
  if (image == nullptr) {
    MS_LOG(ERROR) << "image is nullptr.";
    return RET_ERROR;
  }
  if (width <= 0 || height <= 0) {
    MS_LOG(ERROR) << "Both width and height must be > 0."
                  << " width:" << width << " height:" << height;
    return RET_ERROR;
  }
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

int PreProcess(const preprocess::DataPreProcessParam &data_pre_process_param, const std::string &input_name,
               size_t image_index, mindspore::tensor::MSTensor *tensor) {
  if (tensor == nullptr) {
    MS_LOG(ERROR) << "tensor is nullptr.";
    return RET_NULL_PTR;
  }
  size_t size;
  char *data_buffer = nullptr;
  auto ret =
    PreProcess(data_pre_process_param, input_name, image_index, reinterpret_cast<void **>(&data_buffer), &size);
  if (data_buffer == nullptr || size == 0) {
    MS_LOG(ERROR) << "data_buffer is nullptr or size == 0";
    return RET_OK;
  }
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Preprocess failed.";
    delete[] data_buffer;
    return RET_ERROR;
  }
  auto data = tensor->MutableData();
  if (data == nullptr) {
    MS_LOG(ERROR) << "Get tensor MutableData return nullptr";
    delete[] data_buffer;
    return RET_NULL_PTR;
  }
  if (size != tensor->Size()) {
    MS_LOG(ERROR) << "the input data is not consistent with model input, file_size: " << size
                  << " input tensor size: " << tensor->Size();
    delete[] data_buffer;
    return RET_ERROR;
  }
  if (memcpy_s(data, tensor->Size(), data_buffer, size) != EOK) {
    MS_LOG(ERROR) << "memcpy data failed.";
    delete[] data_buffer;
    return RET_ERROR;
  }
  delete[] data_buffer;
  return RET_OK;
}

int PreProcess(const DataPreProcessParam &data_pre_process_param, const std::string &input_name, size_t image_index,
               void **data, size_t *size) {
  if (data == nullptr || size == nullptr) {
    MS_LOG(ERROR) << "data or size is nullptr.";
    return RET_NULL_PTR;
  }

  if (data_pre_process_param.calibrate_path_vector.find(input_name) ==
      data_pre_process_param.calibrate_path_vector.end()) {
    MS_LOG(ERROR) << "Cant find input:" << input_name;
    return RET_INPUT_PARAM_INVALID;
  }
  auto data_path = data_pre_process_param.calibrate_path_vector.at(input_name).at(image_index);
  if (data_pre_process_param.input_type == IMAGE) {
    cv::Mat mat;
    auto ret = ReadImage(data_path, &mat);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "Read image failed.";
      return ret;
    }
    ret = ImagePreProcess(data_pre_process_param.image_pre_process, &mat, data, size);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "Image Preprocess failed.";
      return ret;
    }
  } else if (data_pre_process_param.input_type == BIN) {
    *data = ReadFile(data_path.c_str(), size);
    if (*data == nullptr || *size == 0) {
      MS_LOG(ERROR) << "ReadFile return nullptr";
      return RET_NULL_PTR;
    }
  } else {
    MS_LOG(ERROR) << "INPUT ILLEGAL: input_type must be IMAGE|BIN.";
    return RET_ERROR;
  }
  return RET_OK;
}

int ImagePreProcess(const ImagePreProcessParam &image_preprocess_param, cv::Mat *image, void **data, size_t *size) {
  if (image == nullptr || data == nullptr || size == nullptr) {
    MS_LOG(ERROR) << "data or size is nullptr.";
    return RET_NULL_PTR;
  }
  int ret;
  if (image_preprocess_param.image_to_format_code != cv::COLOR_COLORCVT_MAX) {
    ret = ConvertImageFormat(image, image_preprocess_param.image_to_format_code);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "convert image format failed.";
      return ret;
    }
  }
  // normalize_mean and normalize_std vector size must equal.
  if (!image_preprocess_param.normalize_mean.empty() || !image_preprocess_param.normalize_std.empty()) {
    ret = Normalize(image, image_preprocess_param.normalize_mean, image_preprocess_param.normalize_std);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "image normalize failed.";
      return ret;
    }
  }

  if (image_preprocess_param.resize_method != cv::INTER_MAX) {
    ret = Resize(image, image_preprocess_param.resize_width, image_preprocess_param.resize_height,
                 image_preprocess_param.resize_method);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "image reize failed.";
      return ret;
    }
  }

  if (image_preprocess_param.center_crop_height != -1 && image_preprocess_param.center_crop_width != -1) {
    ret = CenterCrop(image, image_preprocess_param.center_crop_width, image_preprocess_param.center_crop_height);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "image center crop failed.";
      return ret;
    }
  }

  ret = GetMatData(*image, data, size);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Get mat data failed.";
    return ret;
  }
  return RET_OK;
}
}  // namespace preprocess
}  // namespace lite
}  // namespace mindspore
