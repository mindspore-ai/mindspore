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
    case kNumberTypeFloat32:
      return "kNumberTypeFloat32";
    case kNumberTypeFloat16:
      return "kNumberTypeFloat16";
    case kNumberTypeFloat64:
      return "kNumberTypeFloat64";
    case kTypeUnknown:
      return "kTypeUnknown";
    default:
      return "unsupported type, " + std::to_string(type);
  }
}

}  // namespace mindspore::lite::micro
