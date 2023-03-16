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
#ifndef MINDSPORE_LITE_TOOLS_CONVERTER_MICRO_CODER_UTILS_TYPE_CAST_H_
#define MINDSPORE_LITE_TOOLS_CONVERTER_MICRO_CODER_UTILS_TYPE_CAST_H_

#include <map>
#include <limits>
#include <vector>
#include <string>
#include <typeinfo>
#include <typeindex>
#include "ir/dtype/type_id.h"
#include "include/api/format.h"
#include "src/common/log_adapter.h"
#include "nnacl/op_base.h"
#include "tools/converter/micro/coder/config.h"

namespace mindspore::lite::micro {
std::string EnumNameDataType(TypeId type);

std::string EnumNameMSDataType(TypeId type);

std::string GetTensorDataType(TypeId type);

std::string EnumMicroTensorFormat(mindspore::Format format);

std::string EnumMicroTensorDataType(TypeId type);

std::string EnumNameTarget(Target target);

/**
 * @tparam T
 * @param t, basic data type variable, or tensor
 * @return, data type name
 */
template <typename T>
std::string GetVariableTypeName() {
  std::map<std::type_index, std::string> types_name = {
    {std::type_index(typeid(int)), "int32_t"},           {std::type_index(typeid(int32_t)), "int32_t"},
    {std::type_index(typeid(int16_t)), "int16_t"},       {std::type_index(typeid(uint16_t)), "uint16_t"},
    {std::type_index(typeid(int8_t)), "int8_t"},         {std::type_index(typeid(uint8_t)), "uint8_t"},
    {std::type_index(typeid(float)), "float"},           {std::type_index(typeid(double)), "double"},
    {std::type_index(typeid(::QuantArg)), "QuantArg"},   {std::type_index(typeid(void *)), "void *"},
    {std::type_index(typeid(std::string)), "float *"},   {std::type_index(typeid(int *)), "int32_t *"},
    {std::type_index(typeid(int32_t *)), "int32_t *"},   {std::type_index(typeid(int16_t *)), "int16_t *"},
    {std::type_index(typeid(uint16_t *)), "uint16_t *"}, {std::type_index(typeid(int8_t *)), "int8_t *"},
    {std::type_index(typeid(uint8_t *)), "uint8_t *"},   {std::type_index(typeid(float *)), "float *"}};
  auto item = types_name.find(std::type_index(typeid(T)));
  if (item != types_name.end()) {
    return item->second;
  }
  MS_LOG(ERROR) << "unsupported variable type";
  return "";
}
}  // namespace mindspore::lite::micro
#endif  // MINDSPORE_LITE_TOOLS_CONVERTER_MICRO_CODER_UTILS_TYPE_CAST_H_
