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

#include "coder/generator/utils/generator_utils.h"
#include <map>
#include <fstream>
#include <vector>
#include <utility>
#include <string>
#include "include/errorcode.h"
#include "coder/log.h"
#include "coder/utils/print_utils.h"
#include "src/common/file_utils.h"

namespace mindspore::lite::micro {

int WriteContentToFile(const std::string &file, const std::string &content) {
  std::ofstream of(file);
  if (of.bad()) {
    MS_LOG(ERROR) << "open file error " << file.c_str();
    return RET_ERROR;
  }
  MS_LOG(INFO) << "write " << file.c_str();
  of << content;
  of.close();
  return RET_OK;
}

void CodeReadModelParams(const std::map<std::string, Tensor *> &saved_weights,
                         const std::map<Tensor *, std::string> &tensors_map, std::ofstream &ofs) {
  ofs << "\n\tstruct ModelParameter {\n"
      << "\t\tvoid *addr;\n"
      << "\t\tsize_t size;\n"
      << "\t\tsize_t offset;\n"
      << "\t};\n";

  size_t params_num = 0;
  size_t offset = 0;
  ofs << "\n\tstruct ModelParameter model_params[] = {\n";
  for (const auto &item : saved_weights) {
    std::string name = item.first;
    Tensor *tensor = item.second;
    if (tensor->category() == Tensor::Category::CONST_TENSOR) {
      auto iter = std::find_if(tensors_map.begin(), tensors_map.end(),
                               [&tensor](const std::pair<Tensor *, std::string> &t) { return t.first == tensor; });
      if (iter != tensors_map.end()) {
        ofs << "\t\t{" << name << ", " << tensor->Size() << ", " << offset << "},\n";
        params_num++;
      }
      offset += tensor->Size();
    }
  }
  ofs << "\t};\n";

  offset = 0;
  for (const auto &item : saved_weights) {
    std::string name = item.first;
    Tensor *tensor = item.second;
    if (tensor->category() == Tensor::Category::CONST_TENSOR) {
      auto iter = std::find_if(tensors_map.begin(), tensors_map.end(),
                               [&tensor](const std::pair<Tensor *, std::string> &t) { return t.first == tensor; });
      if (iter == tensors_map.end()) {
        TypeId data_type = tensor->data_type();
        ofs << "\t" << GetTensorDataType(data_type) << "*" << name << " = (weight_buffer + " << offset << ");\n";
      }
      offset += tensor->Size();
    }
  }
  ofs << "\n";

  ofs << "\tfor(int i = 0; i < " << params_num << "; ++i) {\n"
      << "\t\tif (model_params[i].offset + model_params[i].size > weight_size) {\n"
         "\t\t\tMICRO_ERROR(\"buffer is invalid, size: %d, offset: %lu\", weight_size, model_params[i].offset);\n"
         "\t\t\treturn RET_ERROR;\n"
         "\t\t}\n"
      << "\t\tmemcpy(model_params[i].addr, (weight_buffer + model_params[i].offset), model_params[i].size);\n"
      << "\t}\n";
}

int SaveDataToNet(const std::map<std::string, Tensor *> &tensors_map, const std::string &net_file) {
  std::ofstream out(net_file, std::ios::out | std::ios::trunc | std::ios::binary);
  MS_CHECK_TRUE(out.is_open(), "net file open failed!");
  for (auto &item : tensors_map) {
    std::string name = item.first;
    Tensor *tensor = item.second;
    if (tensor->category() == Tensor::Category::CONST_TENSOR) {
      if (tensor->data_c() == nullptr) {
        continue;
      }
      out.write(reinterpret_cast<const char *>(tensor->data_c()), tensor->Size());
    }
  }
  out.close();
  return RET_OK;
}

void CodeModelParamsDefine(const std::map<std::string, Tensor *> &address_map, std::ofstream &hfile,
                           std::ofstream &cfile) {
  for (auto &item : address_map) {
    std::string name = item.first;
    Tensor *tensor = item.second;
    if (tensor->data_c() == nullptr) {
      continue;
    }
    if (tensor->category() == Tensor::Category::CONST_TENSOR) {
      PrintTensorForNet(tensor, cfile, hfile, name);
    } else if (tensor->category() == Tensor::Category::VAR) {
      hfile << "extern " << GetTensorDataType(tensor->data_type()) << " *" << name << ";\n";
      cfile << GetTensorDataType(tensor->data_type()) << "*" << name << " = NULL;\n";
    }
  }
  cfile << "\n";
}

void CodeModelParamsDefineAndData(const std::map<std::string, Tensor *> &address_map, std::ofstream &hfile,
                                  std::ofstream &cfile) {
  for (auto &item : address_map) {
    std::string name = item.first;
    Tensor *tensor = item.second;
    if (tensor->category() == Tensor::Category::CONST_TENSOR) {
      if (tensor->data_c() == nullptr) {
        continue;
      }
      PrintTensor(tensor, cfile, hfile, name);
    }
  }
}

int PrintMicroTensors(std::ofstream &ofs, std::vector<Tensor *> tensors, const std::string &name,
                      const std::map<Tensor *, std::string> &tensors_map) {
  for (size_t index = 0; index < tensors.size(); ++index) {
    Tensor *tensor = tensors[index];
    auto item = tensors_map.find(tensor);
    if (item == tensors_map.end()) {
      MS_LOG(ERROR) << "nonexistent tensor";
      return RET_ERROR;
    }
    ofs << "  static int dim[] = {";
    for (size_t i = 0; i < tensor->shape().size(); ++i) {
      ofs << tensor->shape()[i] << ", ";
    }
    ofs << "};\n";
    ofs << "  " << name << "[" << index << "].ndim = " << tensor->shape().size() << ";\n";
    ofs << "  " << name << "[" << index << "].dim = dim;\n";
    ofs << "  " << name << "[" << index << "].type = " << GetMicroTensorDataType(tensor->data_type()) << ";\n";
    ofs << "  " << name << "[" << index << "].format = " << std::to_string(tensor->format()) << ";\n";
    ofs << "  " << name << "[" << index << "].data =" << item->second << ";\n";
  }
  return RET_OK;
}

void IncludeCmsisDirectories(std::ofstream &ofs) {
  ofs << "include_directories(${OP_HEADER_PATH}/cmsis)\n";
  ofs << "include_directories(${OP_HEADER_PATH}/cmsis/CMSIS/NN/Include)\n";
  ofs << "include_directories(${OP_HEADER_PATH}/cmsis/CMSIS/DSP/Include)\n";
  ofs << "include_directories(${OP_HEADER_PATH}/cmsis/CMSIS/Core/Include)\n";
}

}  // namespace mindspore::lite::micro
