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
#include "coder/utils/coder_utils.h"
#include <set>
#include <queue>
#include <string>
#include <memory>
#include <fstream>
#include "coder/log.h"
#include "coder/utils/type_cast.h"
#include "coder/allocator/allocator.h"

namespace mindspore::lite::micro {
template <typename T>
void TensorDataToFile(const lite::Tensor *tensor, std::ofstream &ofs) {
  const int NUM = 45;
  T *data = reinterpret_cast<T *>(tensor->data_c());
  if (data == nullptr) {
    MS_LOG(ERROR) << "data is nullptr";
    return;
  }
  ofs << "{\n";
  if (typeid(T) == typeid(float)) {
    ofs.precision(kWeightPrecision);
  }
  int len = tensor->ElementsNum();
  for (int i = 0; i < len; ++i) {
    ofs << std::to_string(data[i]) << ", ";
    if (i % NUM == NUM - 1) {
      ofs << "\n";
    }
  }
  ofs << "\n};\n\n";
}

void PrintTensorData(const lite::Tensor *tensor, std::ofstream &ofs) {
  TypeId type = tensor->data_type();
  switch (tensor->data_type()) {
    case kNumberTypeFloat:
    case kNumberTypeFloat32:
      TensorDataToFile<float>(tensor, ofs);
      break;
    case kNumberTypeInt8:
      TensorDataToFile<int8_t>(tensor, ofs);
      break;
    case kNumberTypeInt:
    case kNumberTypeInt32:
      TensorDataToFile<int32_t>(tensor, ofs);
      break;
    case kNumberTypeInt64:
      TensorDataToFile<int64_t>(tensor, ofs);
      break;
    case kNumberTypeUInt8:
      TensorDataToFile<uint8_t>(tensor, ofs);
      break;
    case kNumberTypeUInt32:
      TensorDataToFile<uint32_t>(tensor, ofs);
      break;
    default:
      MS_LOG(ERROR) << "unsupported data type: " << EnumNameDataType(type);
      break;
  }
}

std::string TensorsToString(const std::vector<Tensor *> &tensors, const std::string &is_input) {
  MemoryAllocator *allocator = MemoryAllocator::GetInstance();
  std::string info;
  for (const auto &tensor : tensors) {
    if (tensor->category() == Tensor::Category::CONST_TENSOR) {
      continue;
    }
    info += "      {\n";
    info += "      int dim[] = " + ArrayToString(tensor->shape()) + ";\n";
    info += "      MicroTensor tensor = {";
    info += EnumMicroTensorDataType(tensor->data_type()) + ", ";
    info += EnumMicroTensorFormat(tensor->format()) + ", ";
    info += std::to_string(tensor->shape().size()) + ", dim, ";
    info += allocator->GetRuntimeAddr(tensor) + "};\n";
    info += "      fprintf(output_file, \"" + is_input + " Tensor: " + allocator->GetRuntimeAddr(tensor) + "\\n\");\n";
    info += "      PrintTensor(&tensor, output_file, \"" + is_input + "\");\n";
    info += "      }\n";
  }
  return info;
}

std::vector<std::string> AddDumpDataInfo(const std::vector<std::string> &blocks,
                                         const std::vector<std::unique_ptr<OperatorCoder>> &opcoders) {
  std::vector<std::string> results;
  if (blocks.size() != opcoders.size()) {
    MS_LOG(ERROR) << "error, coder blocks size is not equal to opcoders size";
    return results;
  }
  size_t num = opcoders.size();
  for (size_t i = 0; i < num; ++i) {
    auto &opcoder = opcoders.at(i);
    std::string code = blocks.at(i);
    std::string name = opcoder->name();
    code += "    {\n";
    code += "      FILE *output_file = fopen(\"./" + name + ".ir\", \"w\");\n";
    code += "      fprintf(output_file, \"Node:" + name + "\\n\");\n";
    code += TensorsToString(opcoder->input_tensors(), "input");
    code += TensorsToString(opcoder->output_tensors(), "output");
    code += "      fclose(output_file);\n";
    code += "    }\n";
    results.emplace_back(code);
  }
  return results;
}

std::vector<std::string> SplitString(std::string str, const std::string &pattern) {
  std::vector<std::string> results;
  if (str.empty()) {
    MS_LOG(ERROR) << "source string is empty";
    return results;
  }
  str += pattern;
  while (!str.empty()) {
    size_t size = str.size();
    size_t pos = str.find(pattern);
    std::string sub_string = str.substr(0, pos);
    results.push_back(sub_string);
    str = str.substr(pos + 1, size);
  }
  return results;
}
}  // namespace mindspore::lite::micro
