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

#ifndef MINDSPORE_MICRO_PRINT_UTILS_H_
#define MINDSPORE_MICRO_PRINT_UTILS_H_

#include <fstream>
#include <sstream>
#include <string>
#include <map>
#include <typeindex>
#include <typeinfo>
#include "src/tensor.h"
#include "nnacl/int8/quantize.h"

namespace mindspore::lite::micro {

constexpr int kWeightPrecision = 9;

std::string GetPrintFormat(const lite::Tensor *tensor);

void PrintTensor(const lite::Tensor *tensor, std::ofstream &weightOf, std::ofstream &hOf,
                 const std::string &tensorName);

void PrintTensorForNet(const lite::Tensor *tensor, std::ofstream &weightOf, std::ofstream &hOf,
                       const std::string &tensorName);

std::string GetTensorDataType(const TypeId typeId);

std::string GetMicroTensorDataType(TypeId type);

/**
 * @tparam T
 * @param t, basic data type variable, or tensor
 * @return, data type name
 */
template <typename T>
std::string GetVariableTypeName() {
  std::map<std::type_index, std::string> types_name = {{std::type_index(typeid(int)), "int"},
                                                       {std::type_index(typeid(int32_t)), "int32_t"},
                                                       {std::type_index(typeid(int16_t)), "int16_t"},
                                                       {std::type_index(typeid(int8_t)), "int8_t"},
                                                       {std::type_index(typeid(float)), "float"},
                                                       {std::type_index(typeid(double)), "double"},
                                                       {std::type_index(typeid(::QuantArg)), "QuantArg"},
                                                       {std::type_index(typeid(int *)), "int *"},
                                                       {std::type_index(typeid(int32_t *)), "int32_t *"},
                                                       {std::type_index(typeid(int16_t *)), "int16_t *"},
                                                       {std::type_index(typeid(int8_t *)), "int8_t *"},
                                                       {std::type_index(typeid(float *)), "float *"}};

  auto item = types_name.find(std::type_index(typeid(T)));
  if (item != types_name.end()) {
    return item->second;
  }
  MS_LOG(ERROR) << "unsupported variable type";
  return "";
}
}  // namespace mindspore::lite::micro

#endif  // MINDSPORE_MICRO_PRINT_UTILS_H_
