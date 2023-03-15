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
#include "tools/converter/micro/coder/utils/type_cast.h"
#include <string>

namespace mindspore::lite::micro {
std::string EnumNameDataType(TypeId type) {
  switch (type) {
    case kNumberTypeInt:
      return "kNumberTypeInt";
    case kNumberTypeInt8:
      return "kNumberTypeInt8";
    case kNumberTypeInt16:
      return "kNumberTypeInt16";
    case kNumberTypeInt32:
      return "kNumberTypeInt32";
    case kNumberTypeInt64:
      return "kNumberTypeInt64";
    case kNumberTypeUInt:
      return "kNumberTypeUInt";
    case kNumberTypeUInt8:
      return "kNumberTypeUInt8";
    case kNumberTypeUInt16:
      return "kNumberTypeUInt16";
    case kNumberTypeUInt32:
      return "kNumberTypeUInt32";
    case kNumberTypeFloat:
    case kNumberTypeFloat32:
      return "kNumberTypeFloat32";
    case kNumberTypeFloat16:
      return "kNumberTypeFloat16";
    case kNumberTypeFloat64:
      return "kNumberTypeFloat64";
    case kTypeUnknown:
      return "kTypeUnknown";
    default:
      return "unsupported";
  }
}

std::string EnumNameMSDataType(TypeId type) {
  switch (type) {
    case kNumberTypeInt:
      return "kMSDataTypeNumberTypeInt32";
    case kNumberTypeInt8:
      return "kMSDataTypeNumberTypeInt8";
    case kNumberTypeInt16:
      return "kMSDataTypeNumberTypeInt16";
    case kNumberTypeInt32:
      return "kMSDataTypeNumberTypeInt32";
    case kNumberTypeInt64:
      return "kMSDataTypeNumberTypeUInt64";
    case kNumberTypeUInt:
      return "kMSDataTypeNumberTypeUInt32";
    case kNumberTypeUInt8:
      return "kMSDataTypeNumberTypeUInt8";
    case kNumberTypeUInt16:
      return "kMSDataTypeNumberTypeUInt16";
    case kNumberTypeUInt32:
      return "kMSDataTypeNumberTypeUInt32";
    case kNumberTypeFloat:
    case kNumberTypeFloat32:
      return "kMSDataTypeNumberTypeFloat32";
    case kNumberTypeFloat16:
      return "kMSDataTypeNumberTypeFloat16";
    case kNumberTypeFloat64:
      return "kMSDataTypeNumberTypeFloat64";
    case kTypeUnknown:
      return "kMSDataTypeUnknown";
    default:
      return "unsupported";
  }
}

std::string GetTensorDataType(TypeId type) {
  switch (type) {
    case kNumberTypeFloat:
    case kNumberTypeFloat32:
      return "float ";
    case kNumberTypeFloat16:
      return "uint16_t ";
    case kNumberTypeInt8:
      return "int8_t ";
    case kNumberTypeInt16:
      return "int16_t ";
    case kNumberTypeInt:
    case kNumberTypeInt32:
      return "int32_t ";
    case kNumberTypeUInt8:
      return "uint8_t ";
    case kNumberTypeUInt32:
      return "uint32_t ";
    case kNumberTypeInt64:
      return "int64_t ";
    default:
      MS_LOG(ERROR) << "unsupported data type: " << EnumNameDataType(type);
      return "";
  }
}

std::string EnumMicroTensorFormat(mindspore::Format format) {
  switch (format) {
    case mindspore::NHWC:
      return "Format_NHWC";
    case mindspore::NCHW:
      return "Format_NCHW";
    case mindspore::HWKC:
      return "Format_HWKC";
    case mindspore::HWCK:
      return "Format_HWCK";
    case mindspore::KCHW:
      return "Format_KCHW";
    case mindspore::CKHW:
      return "Format_CKHW";
    case mindspore::NC4HW4:
      return "Format_NC4HW4";
    default:
      MS_LOG(ERROR) << "unsupported format: " << format;
      return "Format_NUM_OF_FORMAT";
  }
}

std::string EnumMicroTensorDataType(TypeId type) {
  switch (type) {
    case kNumberTypeFloat:
    case kNumberTypeFloat32:
      return "DataType_DT_FLOAT";
    case kNumberTypeInt8:
      return "DataType_DT_INT8";
    case kNumberTypeInt:
    case kNumberTypeInt32:
      return "DataType_DT_INT32";
    case kNumberTypeUInt8:
      return "DataType_DT_UINT8";
    case kNumberTypeInt16:
      return "DataType_DT_INT16";
    case kNumberTypeUInt32:
      return "DataType_DT_UINT32";
    case kNumberTypeInt64:
      return "DataType_DT_INT64";
    case kNumberTypeUInt16:
      return "DataType_DT_UINT16";
    case kNumberTypeFloat16:
      MS_LOG(WARNING) << "unsupported data type: kNumberTypeFloat16";
      return "DataType_DT_FLOAT16";
    default:
      MS_LOG(WARNING) << "unsupported data type: " << type << ", reference: " << kNumberTypeInt;
      return "DataType_DT_UNDEFINED";
  }
}

std::string EnumNameTarget(Target target) {
  switch (target) {
    case kX86:
      return "kX86";
    case kCortex_M:
      return "kCortex_M";
    case kARM32:
      return "kARM32";
    case kARM64:
      return "kARM64";
    case kAllTargets:
      return "kAllTargets";
    default:
      return "kTargetUnknown";
  }
}
}  // namespace mindspore::lite::micro
