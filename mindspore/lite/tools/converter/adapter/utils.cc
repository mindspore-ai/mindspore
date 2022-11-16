/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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
#include "adapter/utils.h"
#include <cstdio>
#ifndef _WIN32
#include <dlfcn.h>
#include <unistd.h>
#else
#include <direct.h>
#include <io.h>
#endif
#include <dirent.h>
#include <sys/stat.h>
#include <algorithm>
#include <functional>
#include <unordered_map>
#include <memory>
#include <numeric>
#include <string>
#include <vector>
#include <regex>
#include "include/api/model.h"
#include "include/api/context.h"
#include "src/common/file_utils.h"

#ifdef _WIN32
#define ACCESS _access
#define RMDIR _rmdir
#else
#define ACCESS access
#define RMDIR rmdir
#endif
namespace mindspore {
namespace lite {
namespace converter {
constexpr int kThreadNum = 2;
constexpr bool kEnableParallel = false;
constexpr int kThreadAffinity = 1;
constexpr int kInterOpParallelNum = 1;
constexpr bool kEnableFP16 = false;
const std::unordered_map<int, std::string> kTypeIdMap{
  {kNumberTypeFloat16, "Float16"}, {kNumberTypeFloat, "Float32"},    {kNumberTypeFloat32, "Float32"},
  {kNumberTypeInt8, "Int8"},       {kNumberTypeInt16, "Int16"},      {kNumberTypeInt, "Int32"},
  {kNumberTypeInt32, "Int32"},     {kNumberTypeUInt8, "UInt8"},      {kNumberTypeUInt16, "UInt16"},
  {kNumberTypeUInt, "UInt32"},     {kNumberTypeUInt32, "UInt32"},    {kObjectTypeString, "String"},
  {kNumberTypeBool, "Bool"},       {kObjectTypeTensorType, "Tensor"}};

const std::unordered_map<mindspore::Format, std::string> kTensorFormatMap{
  {mindspore::NCHW, "NCHW"}, {mindspore::NHWC, "NHWC"},     {mindspore::NHWC4, "NHWC4"}, {mindspore::HWKC, "HWKC"},
  {mindspore::HWCK, "HWCK"}, {mindspore::KCHW, "KCHW"},     {mindspore::CKHW, "CKHW"},   {mindspore::KHWC, "KHWC"},
  {mindspore::CHWK, "CHWK"}, {mindspore::HW, "HW"},         {mindspore::HW4, "HW4"},     {mindspore::NC, "NC"},
  {mindspore::NC4, "NC4"},   {mindspore::NC4HW4, "NC4HW4"}, {mindspore::NCDHW, "NCDHW"}};

bool RemoveDir(const std::string &path) {
  std::string str_path = path;
#ifdef _WIN32
  struct _finddata_t fb;
  if (str_path.at(str_path.length() - 1) != '\\' || str_path.at(str_path.length() - 1) != '/') str_path.append("\\");
  std::string find_path = str_path + "*";
  intptr_t handle = _findfirst(find_path.c_str(), &fb);
  if (handle != -1L) {
    std::string tmp_path;
    do {
      if (strcmp(fb.name, "..") != 0 && strcmp(fb.name, ".") != 0) {
        tmp_path.clear();
        tmp_path = str_path + std::string(fb.name);
        if (fb.attrib == _A_SUBDIR) {
          (void)RemoveDir(tmp_path.c_str());
        } else {
          remove(tmp_path.c_str());
        }
      }
    } while (_findnext(handle, &fb) == 0);
    _findclose(handle);
  }
  return RMDIR(str_path.c_str()) == 0 ? true : false;
#else
  if (str_path.at(str_path.length() - 1) != '\\' || str_path.at(str_path.length() - 1) != '/') {
    (void)str_path.append("/");
  }
  DIR *d = opendir(str_path.c_str());
  if (d != nullptr) {
    struct dirent *dt = readdir(d);
    while (dt) {
      if (strcmp(dt->d_name, "..") != 0 && strcmp(dt->d_name, ".") != 0) {
        struct stat st {};
        std::string file_name = str_path + std::string(dt->d_name);
        (void)stat(file_name.c_str(), &st);
        if (S_ISDIR(st.st_mode)) {
          (void)RemoveDir(file_name);
        } else {
          (void)remove(file_name.c_str());
        }
      }
      dt = readdir(d);
    }
    (void)closedir(d);
  }
  return rmdir(str_path.c_str()) == 0;
#endif
}

int ReadInputFile(const std::vector<std::string> &in_data_files, std::vector<MSTensor> *input_tensors) {
  for (size_t i = 0; i < in_data_files.size(); i++) {
    auto &cur_tensor = (*input_tensors).at(i);
    MS_ASSERT(cur_tensor != nullptr);
    size_t size;
    char *bin_buf = lite::ReadFile(in_data_files[i].c_str(), &size);
    if (bin_buf == nullptr) {
      MS_LOG(ERROR) << "ReadFile return nullptr";
      return RET_ERROR;
    }
    if (static_cast<int>(cur_tensor.DataType()) == kObjectTypeString) {
      std::string str(bin_buf, size);
      MSTensor *input = MSTensor::StringsToTensor(cur_tensor.Name(), {str});
      if (input == nullptr) {
        MS_LOG(ERROR) << "StringsToTensor failed";
        delete[] bin_buf;
        return RET_ERROR;
      }
      cur_tensor = *input;
    } else {
      auto tensor_data_size = cur_tensor.DataSize();
      if (size != tensor_data_size) {
        MS_LOG(ERROR) << "Input binary file size error, required: " << tensor_data_size << ", in fact: " << size;
        delete[] bin_buf;
        return RET_ERROR;
      }
      auto input_data = cur_tensor.MutableData();
      if (input_data == nullptr) {
        MS_LOG(ERROR) << "input_data is nullptr.";
        delete[] bin_buf;
        return RET_ERROR;
      }
      auto ret = memcpy_s(input_data, tensor_data_size, bin_buf, size);
      if (ret != EOK) {
        MS_LOG(ERROR) << "Execute memcpy_s failed.";
        delete[] bin_buf;
        return RET_ERROR;
      }
    }
    delete[] bin_buf;
  }

  return RET_OK;
}
std::string GenerateOutputFileName(const mindspore::MSTensor *tensor, const std::string &op_name,
                                   const std::string &file_type, const size_t &idx) {
  std::string file_name = op_name;
  auto pos = file_name.find_first_of('/');
  while (pos != std::string::npos) {
    file_name.replace(pos, 1, ".");
    pos = file_name.find_first_of('/');
  }
  file_name += "_" + file_type + "_" + std::to_string(idx) + "_shape_";
  for (const auto &dim : tensor->Shape()) {
    file_name += std::to_string(dim) + "_";
  }
  if (kTypeIdMap.find(static_cast<int>(tensor->DataType())) != kTypeIdMap.end()) {
    file_name += kTypeIdMap.at(static_cast<int>(tensor->DataType()));
  }
  auto tensor_format = tensor->format();
  if (kTensorFormatMap.find(tensor_format) != kTensorFormatMap.end()) {
    file_name += "_" + kTensorFormatMap.at(tensor_format) + ".bin";
  }

  file_name += +".bin";
  return file_name;
}

int InnerPredict(const std::string &model_name, const std::string &in_data_file,
                 const std::vector<std::string> &output_names, const std::string &dump_directory,
                 const std::vector<std::vector<int64_t>> &input_shapes) {
  std::string dump_data_path = dump_directory + "dump_data/";
  if (ACCESS(dump_data_path.c_str(), 0) == 0 && !RemoveDir(dump_data_path)) {
    MS_LOG(ERROR) << "The dump data path is exist and rmdir failed: "
                  << " " << dump_data_path;
    return RET_ERROR;
  }
  if (lite::CreateOutputDir(&dump_data_path) != RET_OK) {
    MS_LOG(ERROR) << "create data output directory failed.";
    return RET_ERROR;
  }

  MSKernelCallBack ms_after_call_back = [output_names, dump_data_path](
                                          const std::vector<mindspore::MSTensor> &after_inputs,
                                          const std::vector<mindspore::MSTensor> &after_outputs,
                                          const MSCallBackParam &call_param) {
    if (std::find(output_names.begin(), output_names.end(), call_param.node_name) != output_names.end()) {
      for (size_t i = 0; i < after_outputs.size(); i++) {
        auto ms_tensor = after_outputs.at(i);
        auto file_name = GenerateOutputFileName(&ms_tensor, call_param.node_name, "output", i);
        auto abs_file_path = dump_data_path + "/" + file_name;
        if (lite::WriteToBin(abs_file_path, ms_tensor.MutableData(), ms_tensor.DataSize()) != RET_OK) {
          MS_LOG(ERROR) << "write tensor data to file failed.";
          return false;
        }
      }
    }
    return true;
  };

  mindspore::Model ms_model;
  mindspore::ModelType model_type = mindspore::ModelType::kMindIR_Lite;
  auto context = std::make_shared<mindspore::Context>();
  if (context == nullptr) {
    MS_LOG(ERROR) << "New context failed while running " << model_name.c_str();
    return RET_ERROR;
  }
  context->SetThreadNum(kThreadNum);
  context->SetEnableParallel(kEnableParallel);
  context->SetThreadAffinity(kThreadAffinity);
  context->SetInterOpParallelNum(kInterOpParallelNum);
  auto &device_list = context->MutableDeviceInfo();

  std::shared_ptr<CPUDeviceInfo> device_info = std::make_shared<CPUDeviceInfo>();
  device_info->SetEnableFP16(kEnableFP16);
  device_list.push_back(device_info);
  auto ret = ms_model.Build(model_name, model_type, context);
  if (ret != kSuccess) {
    MS_LOG(ERROR) << "Execute model " << model_name.c_str() << " build failed.";
    return RET_ERROR;
  }

  if (!input_shapes.empty()) {
    if (input_shapes.size() != ms_model.GetInputs().size()) {
      MS_LOG(ERROR) << "Size of inputs_shape_list should be equal to size of inputs";
      return RET_ERROR;
    }

    ret = ms_model.Resize(ms_model.GetInputs(), input_shapes);
    if (ret != kSuccess) {
      MS_LOG(ERROR) << "Input tensor resize failed.";
      return RET_ERROR;
    }
  }

  auto inputs = ms_model.GetInputs();
  if (inputs.empty()) {
    MS_LOG(ERROR) << "Inputs is empty.";
    return RET_ERROR;
  }
  std::regex re{"[\\s,]+"};
  std::vector<std::string> in_data_list = std::vector<std::string>{
    std::sregex_token_iterator(in_data_file.begin(), in_data_file.end(), re, -1), std::sregex_token_iterator()};
  if (in_data_list.size() != inputs.size()) {
    MS_LOG(ERROR) << "Size of inputs should be equal to size of input inDataPath";
    return RET_ERROR;
  }
  auto status = ReadInputFile(in_data_list, &inputs);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "ReadInputFile error, " << status;
    return status;
  }

  std::vector<MSTensor> outputs;
  ret = ms_model.Predict(inputs, &outputs, nullptr, ms_after_call_back);
  if (ret != kSuccess) {
    MS_LOG(ERROR) << "Predict error ";
    return RET_ERROR;
  }

  return RET_OK;
}
}  // namespace converter
}  // namespace lite
}  // namespace mindspore
