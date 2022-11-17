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

#include "src/data_preprocessor.h"
#include <unordered_set>
#include <vector>
#include <map>
#include <utility>
#include <string>
#include "common/anf_util.h"
#include "common/file_util.h"
#include "common/op_enum.h"
#include "common/data_transpose_utils.h"
#include "common/string_util.h"
#include "src/mapper_config_parser.h"
#include "opencv2/opencv.hpp"

using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
namespace mindspore {
namespace dpico {
namespace {
const std::unordered_set<std::string> kRgbInputFormats = {"BGR_PLANAR", "RGB_PLANAR", "RGB_PACKAGE", "BGR_PACKAGE"};
const std::unordered_set<std::string> kGrayInputFormats = {"RAW_RGGB", "RAW_GRBG", "RAW_GBRG", "RAW_BGGR"};
std::vector<std::string> GetImageRealPaths(const std::string &file_path) {
  std::vector<std::string> img_paths;
  std::ifstream ifs;
  if (ReadFileToIfstream(file_path, &ifs) != RET_OK) {
    MS_LOG(ERROR) << "read file to ifstream failed.";
    return {};
  }
  size_t num_of_line = 0;
  std::string raw_line;
  while (getline(ifs, raw_line)) {
    if (num_of_line > kMaxLineCount) {
      MS_LOG(WARNING) << "the line count is exceeds the maximum range 9999.";
      return img_paths;
    }
    num_of_line++;
    if (EraseBlankSpace(&raw_line) != RET_OK) {
      MS_LOG(ERROR) << "erase blank space failed. " << raw_line;
      return {};
    }
    if (raw_line.empty() || raw_line.at(0) == '#') {
      continue;
    }
    auto img_path = RealPath(raw_line.c_str());
    if (img_path.empty()) {
      MS_LOG(ERROR) << "cur image get realpath failed. " << raw_line;
      return {};
    } else {
      img_paths.push_back(img_path);
    }
  }
  return img_paths;
}
int Normalize(cv::Mat *image, const std::map<int, double> &mean, const std::map<int, double> &var) {
  if (image == nullptr) {
    MS_LOG(ERROR) << "input image is nullptr.";
    return RET_ERROR;
  }
  std::vector<double> mean_vec;
  std::vector<double> var_vec;
  if (image->channels() < 0) {
    MS_LOG(ERROR) << "image channels should not be negative.";
    return RET_ERROR;
  }
  size_t img_channel_size = image->channels();
  if (mean.empty()) {
    mean_vec = std::vector<double>(img_channel_size, 0.0);
  } else {
    if (mean.size() != img_channel_size) {
      MS_LOG(ERROR) << "input mean_chn size " << mean.size() << " is not equal to image channels " << img_channel_size;
      return RET_ERROR;
    }
    (void)std::transform(mean.begin(), mean.end(), std::back_inserter(mean_vec),
                         [](const std::pair<int, double> &pair) { return pair.second; });
  }
  if (var.empty()) {
    var_vec = std::vector<double>(img_channel_size, 0.0);
  } else {
    if (var.size() != img_channel_size) {
      MS_LOG(ERROR) << "input var_reci_chn size " << var.size() << " is not equal to image channels "
                    << img_channel_size;
      return RET_ERROR;
    }
    (void)std::transform(var.begin(), var.end(), std::back_inserter(var_vec),
                         [](const std::pair<int, double> &pair) { return pair.second; });
  }
  std::vector<cv::Mat> channels(img_channel_size);
  cv::split(*image, channels);
  for (size_t i = 0; i < channels.size(); i++) {
    channels[i].convertTo(channels[i], CV_32FC1, var_vec.at(i), (0.0 - mean_vec.at(i)) * var_vec.at(i));
  }
  cv::merge(channels, *image);
  return RET_OK;
}
}  // namespace

DataPreprocessor *DataPreprocessor::GetInstance() {
  static DataPreprocessor instance;
  return &instance;
}

int DataPreprocessor::ModifyDynamicInputShape(std::vector<int64_t> *input_shape) {
  std::vector<size_t> indexes;
  for (size_t i = 0; i < input_shape->size(); i++) {
    if (input_shape->at(i) < 0) {
      indexes.push_back(i);
    }
  }
  if (!indexes.empty()) {
    if (indexes.size() == 1 && indexes.at(0) == 0) {
      input_shape->at(0) = 1;
    } else {
      MS_LOG(ERROR) << "dynamic graph input is unsupported by dpico.";
      return RET_NO_CHANGE;
    }
  }
  return RET_OK;
}

int DataPreprocessor::GetOutputBinDir(const std::string &op_name, std::string *output_bin_dir) {
  auto folder_name = ReplaceSpecifiedChar(op_name, '/', '_');
  *output_bin_dir = preprocessed_data_dir_ + folder_name + "/";
  if (CreateDir(output_bin_dir) != RET_OK) {
    MS_LOG(ERROR) << "Create directory failed. " << *output_bin_dir;
    return RET_ERROR;
  }
  size_t count = 0;
  while (AccessFile(*output_bin_dir + "/" + std::to_string(count), F_OK) == 0) {
    MS_LOG(DEBUG) << "current file_path has existed, file_path cnt plus 1.";  // such as: /xxx/0 ==> /xxx/1
    count++;
    if (count > kMaximumNumOfFolders) {
      MS_LOG(ERROR) << "the number of file folders exceeds the upper limit " << kMaximumNumOfFolders;
      return RET_ERROR;
    }
  }
  *output_bin_dir += std::to_string(count);
  return RET_OK;
}

int DataPreprocessor::WriteCvMatToBin(const cv::Mat &image, const std::string &op_name) {
  std::string generated_bin_dir;
  if (GetOutputBinDir(op_name, &generated_bin_dir) != RET_OK) {
    MS_LOG(ERROR) << "get output bin dir failed.";
    return RET_ERROR;
  }
  if (Mkdir(generated_bin_dir) != RET_OK) {
    MS_LOG(ERROR) << "mkdir failed. " << generated_bin_dir;
    return RET_ERROR;
  }
  std::string output_bin_path = generated_bin_dir + "/input.bin";
  std::ofstream ofs;
  ofs.open(output_bin_path, std::ios::binary);
  if (!ofs.good() || !ofs.is_open()) {
    MS_LOG(ERROR) << "open output bin file failed. " << output_bin_path;
    return RET_ERROR;
  }
  for (int i = 0; i < image.rows; i++) {
    (void)ofs.write(reinterpret_cast<const char *>(image.ptr(i)), static_cast<size_t>(image.cols) * image.elemSize());
  }
  ofs.close();
  return RET_OK;
}

int DataPreprocessor::GenerateInputBinFromTxt(const std::string &raw_data_path, const std::string &op_name,
                                              const std::vector<int64_t> &op_shape, TypeId type_id) {
  if (op_shape.empty()) {
    MS_LOG(ERROR) << "op shape shouldn't be empty.";
    return RET_ERROR;
  }
  std::ifstream ifs;
  if (ReadFileToIfstream(raw_data_path, &ifs) != RET_OK) {
    MS_LOG(ERROR) << "read file to ifstream failed.";
    return RET_ERROR;
  }
  std::string raw_line;
  batch_size_ = 0;
  while (getline(ifs, raw_line)) {
    if (batch_size_ > kMaxLineCount) {
      MS_LOG(WARNING) << "the line count is exceeds the maximum range 9999.";
      return RET_ERROR;
    }
    batch_size_++;
    auto preprocessed_line = ReplaceSpecifiedChar(raw_line, ',', ' ');  // uniformly separated by spaces
    if (EraseHeadTailSpace(&preprocessed_line) != RET_OK) {
      MS_LOG(ERROR) << "erase head & tail blank space failed. " << preprocessed_line;
      return RET_ERROR;
    }
    if (preprocessed_line.empty() || preprocessed_line.at(0) == '#') {
      continue;
    }
    int status;
    switch (type_id) {
      case kNumberTypeFloat:
      case kNumberTypeFloat32:
        status = GenerateInputBin<float>(preprocessed_line, op_shape, op_name);
        break;
      case kNumberTypeInt8:
        status = GenerateInputBin<int8_t, int16_t>(preprocessed_line, op_shape, op_name);
        break;
      case kNumberTypeUInt8:
        status = GenerateInputBin<uint8_t, uint16_t>(preprocessed_line, op_shape, op_name);
        break;
      case kNumberTypeInt16:
        status = GenerateInputBin<int16_t>(preprocessed_line, op_shape, op_name);
        break;
      case kNumberTypeUInt16:
        status = GenerateInputBin<uint16_t>(preprocessed_line, op_shape, op_name);
        break;
      case kNumberTypeInt:
      case kNumberTypeInt32:
        status = GenerateInputBin<int32_t>(preprocessed_line, op_shape, op_name);
        break;
      case kNumberTypeInt64:
        status = GenerateInputBin<int64_t>(preprocessed_line, op_shape, op_name);
        break;
      case kNumberTypeUInt64:
        status = GenerateInputBin<uint64_t>(preprocessed_line, op_shape, op_name);
        break;
      default:
        MS_LOG(ERROR) << "unsupported data type " << dpico::TypeIdToString(type_id);
        status = RET_ERROR;
    }
    if (status != RET_OK) {
      MS_LOG(ERROR) << "generate input bin files failed.";
      return RET_ERROR;
    }
  }
  return RET_OK;
}
int DataPreprocessor::GenerateInputBinFromImages(const std::string &raw_data_path, const std::string &op_name,
                                                 const std::vector<int64_t> &op_shape,
                                                 const struct AippModule &aipp_module) {
  auto img_paths = GetImageRealPaths(raw_data_path);
  if (img_paths.empty()) {
    MS_LOG(ERROR) << "[image_list] corresponding file is invalid. " << raw_data_path;
    return RET_ERROR;
  }
  batch_size_ = img_paths.size();

  if (op_shape.size() != kDims4) {
    MS_LOG(ERROR) << "op shape should be 4 when input is image.";
    return RET_ERROR;
  }
  for (const auto &img_path : img_paths) {  // preprocess input image
    cv::Mat image;
    if (kRgbInputFormats.find(aipp_module.input_format) != kRgbInputFormats.end()) {
      image = cv::imread(img_path, static_cast<int>(cv::IMREAD_COLOR));
    } else if (kGrayInputFormats.find(aipp_module.input_format) != kGrayInputFormats.end()) {
      image = cv::imread(img_path, static_cast<int>(cv::IMREAD_GRAYSCALE));
    }
    if (image.empty() || image.data == nullptr) {
      MS_LOG(ERROR) << "missing file, improper permissions, unsupported or invalid format.";
      return RET_ERROR;
    }
    if (aipp_module.model_format == "RGB") {
      cv::cvtColor(image, image, static_cast<int>(cv::COLOR_BGR2RGB));
    }
    if (image.cols != op_shape[kAxis3] || image.rows != op_shape[kAxis2]) {
      MS_LOG(INFO) << "input image shape don't match op shape, and it will be resized.";
      cv::resize(image, image, cv::Size(op_shape[kAxis3], op_shape[kAxis2]));
    }

    if (Normalize(&image, aipp_module.mean_map, aipp_module.val_map) != RET_OK) {
      MS_LOG(ERROR) << "image normalize process failed.";
      return RET_ERROR;
    }

    if (WriteCvMatToBin(image, op_name) != RET_OK) {
      MS_LOG(ERROR) << "write image to bin file failed." << op_name;
      return RET_ERROR;
    }
  }
  return RET_OK;
}
int DataPreprocessor::Run(const api::AnfNodePtrList &inputs) {
  if (inputs.empty()) {
    MS_LOG(ERROR) << "graph inputs shouldn't be empty.";
    return RET_ERROR;
  }
  auto image_lists = MapperConfigParser::GetInstance()->GetImageLists();
  auto aipp_modules = MapperConfigParser::GetInstance()->GetAippModules();

  for (const auto &input : inputs) {
    auto op_name = input->fullname_with_scope();
    if (image_lists.find(op_name) == image_lists.end()) {
      MS_LOG(ERROR) << "current op don't exist in image_lists. " << op_name;
      return RET_ERROR;
    }
    auto param_node = input->cast<api::ParameterPtr>();
    if (param_node == nullptr) {
      MS_LOG(ERROR) << "graph input node should be parameter ptr. " << input->fullname_with_scope();
      return RET_ERROR;
    }
    auto abstract_base = param_node->abstract();
    if (abstract_base == nullptr) {
      MS_LOG(ERROR) << "Abstract of parameter is nullptr, " << param_node->name();
      return lite::RET_PARAM_INVALID;
    }

    ShapeVector op_shape;
    if (FetchShapeFromAbstract(abstract_base, &op_shape) != RET_OK) {
      MS_LOG(ERROR) << "get shape vector from graph input failed. " << op_name;
      return RET_ERROR;
    }
    if (op_shape.empty()) {
      MS_LOG(ERROR) << "shape is empty." << op_name;
      return RET_ERROR;
    }
    if (ModifyDynamicInputShape(&op_shape) != RET_OK) {
      MS_LOG(ERROR) << "modify dynamic input shape failed. " << op_name;
      return RET_ERROR;
    }

    TypeId op_type;
    if (FetchTypeIdFromAbstract(abstract_base, &op_type) != RET_OK) {
      MS_LOG(ERROR) << "get type id from graph input failed. " << op_name;
      return RET_ERROR;
    }

    auto raw_data_path = image_lists.at(op_name);
    preprocessed_data_dir_ = MapperConfigParser::GetInstance()->GetOutputPath() + "preprocessed_data/";
    if (aipp_modules.find(op_name) == aipp_modules.end()) {
      if (GenerateInputBinFromTxt(raw_data_path, op_name, op_shape, op_type) != RET_OK) {
        MS_LOG(ERROR) << "generate input bin from txt failed.";
        return RET_ERROR;
      }
    } else {
      if (GenerateInputBinFromImages(raw_data_path, op_name, op_shape, aipp_modules.at(op_name)) != RET_OK) {
        MS_LOG(ERROR) << "generate input bin from images failed.";
        return RET_ERROR;
      }
    }
  }
  return RET_OK;
}
}  // namespace dpico
}  // namespace mindspore
