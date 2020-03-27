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

#ifndef MINDSPORE_CCSRC_UTILS_LOG_DE_ERROR_CODE_H_
#define MINDSPORE_CCSRC_UTILS_LOG_DE_ERROR_CODE_H_

#include <map>
#include <string>

namespace DataEngineBase {
// system ID
const int SYSID_MD = 20;

// Runtime location
enum LogRuntime {
  RT_HOST = 0b01,
  RT_DEVICE = 0b10,
};

// sub model
enum SubModuleId {
  COMMON_MODULE = 0,
  DATAGET_MODULE,
  MINDRECORD_MODULE,
};

// error code type
enum ErrorCodeType {
  ERROR_CODE = 0b01,
  EXCEPTION_CODE = 0b10,
};

// error level
enum ErrorLevel {
  COMMON_LEVEL = 0b000,
  SUGGESTION_LEVEL = 0b001,
  MINOR_LEVEL = 0b010,
  MAJOR_LEVEL = 0b011,
  CRITICAL_LEVEL = 0b100,
};

// code compose(4 byte), runtime: 2 bit,  type: 2 bit,   level: 3 bit,  sysid: 8 bit, modid: 5 bit, value: 12 bit
#define DE_ERRORNO(runtime, type, level, sysid, submodid, name, value, desc)                                       \
  constexpr DataEngineBase::Status name = ((0xFF & ((uint8_t)runtime)) << 30) | ((0xFF & ((uint8_t)type)) << 28) | \
                                          ((0xFF & ((uint8_t)level)) << 25) | ((0xFF & ((uint8_t)sysid)) << 17) |  \
                                          ((0xFF & ((uint8_t)submodid)) << 12) | (0x0FFF & ((uint16_t)value));     \
  const DataEngineBase::ErrorNoRegisterar g_##name##_errorno(name, desc);

// each module defines error codes using the following macros
#define DE_ERRORNO_COMMON(name, value, desc)                                                    \
  DE_ERRORNO(DataEngineBase::RT_HOST, DataEngineBase::ERROR_CODE, DataEngineBase::COMMON_LEVEL, \
             DataEngineBase::SYSID_MD, DataEngineBase::COMMON_MODULE, name, value, desc)

#define DE_ERRORNO_DATASET(name, value, desc)                                                   \
  DE_ERRORNO(DataEngineBase::RT_HOST, DataEngineBase::ERROR_CODE, DataEngineBase::COMMON_LEVEL, \
             DataEngineBase::SYSID_MD, DataEngineBase::DATAGET_MODULE, name, value, desc)

#define DE_ERRORNO_MINDRECORD(name, value, desc)                                                \
  DE_ERRORNO(DataEngineBase::RT_HOST, DataEngineBase::ERROR_CODE, DataEngineBase::COMMON_LEVEL, \
             DataEngineBase::SYSID_MD, DataEngineBase::MINDRECORD_MODULE, name, value, desc)

// get error code description
#define DE_GET_ERRORNO_STR(value) DataEngineBase::StatusFactory::Instance()->GetErrDesc(value)

class StatusFactory {
 public:
  static StatusFactory *Instance() {
    static StatusFactory instance;
    return &instance;
  }

  void RegisterErrorNo(uint32_t err, const std::string &desc) {
    if (err_desc_.find(err) != err_desc_.end()) return;
    err_desc_[err] = desc;
  }

  std::string GetErrDesc(uint32_t err) {
    auto iter_find = err_desc_.find(err);
    if (iter_find == err_desc_.end()) return "";
    return iter_find->second;
  }

 protected:
  StatusFactory() = default;

  ~StatusFactory() = default;

 private:
  std::map<uint32_t, std::string> err_desc_;
};

class ErrorNoRegisterar {
 public:
  ErrorNoRegisterar(uint32_t err, const std::string &desc) { StatusFactory::Instance()->RegisterErrorNo(err, desc); }

  ~ErrorNoRegisterar() = default;
};

using Status = uint32_t;
}  // namespace DataEngineBase
#endif  // MINDSPORE_CCSRC_UTILS_LOG_DE_ERROR_CODE_H_
