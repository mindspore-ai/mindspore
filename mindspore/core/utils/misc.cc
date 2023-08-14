/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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

#include "utils/misc.h"
#include <complex>
#include <map>
#ifndef _MSC_VER
#include <cxxabi.h>
#endif
namespace mindspore {
const int RET_SUCCESS = 0;
const int RET_FAILED = 1;
const int RET_CONTINUE = 2;
const int RET_BREAK = 3;

std::string demangle(const char *name) {
#ifdef _MSC_VER
  return name;
#else
  int status = -1;
  std::unique_ptr<char, void (*)(void *)> res{abi::__cxa_demangle(name, nullptr, nullptr, &status), std::free};
  return (status == 0) ? res.get() : name;
#endif
}

static std::map<TypeId, size_t> datatype_size_map = {{TypeId::kNumberTypeFloat16, sizeof(float) / 2},  // 1/2 of float
                                                     {TypeId::kNumberTypeFloat32, sizeof(float)},
                                                     {TypeId::kNumberTypeFloat64, sizeof(double)},
                                                     {TypeId::kNumberTypeBFloat16, sizeof(float) / 2},
                                                     {TypeId::kNumberTypeInt8, sizeof(int8_t)},
                                                     {TypeId::kNumberTypeInt16, sizeof(int16_t)},
                                                     {TypeId::kNumberTypeInt32, sizeof(int32_t)},
                                                     {TypeId::kNumberTypeInt64, sizeof(int64_t)},
                                                     {TypeId::kNumberTypeUInt8, sizeof(uint8_t)},
                                                     {TypeId::kNumberTypeUInt16, sizeof(uint16_t)},
                                                     {TypeId::kNumberTypeUInt32, sizeof(uint32_t)},
                                                     {TypeId::kNumberTypeUInt64, sizeof(uint64_t)},
                                                     {TypeId::kNumberTypeBool, sizeof(bool)},
                                                     {TypeId::kNumberTypeFloat, sizeof(float)},
                                                     {TypeId::kNumberTypeComplex64, sizeof(std::complex<float>)},
                                                     {TypeId::kNumberTypeComplex128, sizeof(std::complex<double>)}};

size_t GetDataTypeSize(const TypeId &type) {
  if (datatype_size_map.find(type) != datatype_size_map.end()) {
    return datatype_size_map[type];
  } else {
    MS_LOG(ERROR) << "Illegal tensor data type!";
    return kTypeUnknown;
  }
}
}  // namespace mindspore
