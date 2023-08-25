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

#include "include/c_api/ms/value.h"
#include <memory>
#include "c_api/src/helper.h"
#include "c_api/src/common.h"
#include "ir/tensor.h"
#include "c_api/src/utils.h"

ValueHandle MSNewValueInt64(ResMgrHandle res_mgr, const int64_t v) {
  if (res_mgr == nullptr) {
    MS_LOG(ERROR) << "Input Handle [res_mgr] is nullptr.";
    return nullptr;
  }
  auto value = std::make_shared<Int64ImmImpl>(v);
  return GetRawPtr(res_mgr, value);
}

ValueHandle MSNewValueFloat32(ResMgrHandle res_mgr, const float v) {
  if (res_mgr == nullptr) {
    MS_LOG(ERROR) << "Input Handle [res_mgr] is nullptr.";
    return nullptr;
  }
  auto value = std::make_shared<Float32ImmImpl>(v);
  return GetRawPtr(res_mgr, value);
}

ValueHandle MSNewValueBool(ResMgrHandle res_mgr, const bool v) {
  if (res_mgr == nullptr) {
    MS_LOG(ERROR) << "Input Handle [res_mgr] is nullptr.";
    return nullptr;
  }
  auto value = std::make_shared<BoolImmImpl>(v);
  return GetRawPtr(res_mgr, value);
}

ValueHandle MSNewValueType(ResMgrHandle res_mgr, DataTypeC type) {
  if (res_mgr == nullptr) {
    MS_LOG(ERROR) << "Input Handle [res_mgr] is nullptr.";
    return nullptr;
  }
  auto type_value = mindspore::TypeIdToType(mindspore::TypeId(type));
  return GetRawPtr(res_mgr, type_value);
}

ValueHandle MSNewValueStrings(ResMgrHandle res_mgr, const char *strs[], size_t vec_len) {
  if (res_mgr == nullptr) {
    MS_LOG(ERROR) << "Input Handle [res_mgr] is nullptr.";
    return nullptr;
  }
  std::vector<std::string> converted_strs;
  for (size_t i = 0; i < vec_len; i++) {
    std::string converted_ele(strs[i]);
    converted_strs.push_back(converted_ele);
  }
  auto value = mindspore::MakeValue(converted_strs);
  return GetRawPtr(res_mgr, value);
}

ValueHandle MSNewValueArray(ResMgrHandle res_mgr, void *value, size_t vec_size, DataTypeC data_type) {
  if (res_mgr == nullptr) {
    MS_LOG(ERROR) << "Input Handle [res_mgr] is nullptr.";
    return nullptr;
  }

  // Allow empty attribute value
  if (value == nullptr && vec_size != 0) {
    MS_LOG(ERROR) << "Input Handle [value] is nullptr.";
    return nullptr;
  }

  mindspore::ValuePtr value_ptr;
  switch (data_type) {
    case MS_BOOL: {
      std::vector<bool> vec_value(static_cast<bool *>(value), static_cast<bool *>(value) + vec_size);
      value_ptr = mindspore::MakeValue(vec_value);
      break;
    }
    case MS_INT32: {
      std::vector<int32_t> vec_value(static_cast<int32_t *>(value), static_cast<int32_t *>(value) + vec_size);
      value_ptr = mindspore::MakeValue(vec_value);
      break;
    }
    case MS_INT64: {
      std::vector<int64_t> vec_value(static_cast<int64_t *>(value), static_cast<int64_t *>(value) + vec_size);
      value_ptr = mindspore::MakeValue(vec_value);
      break;
    }
    case MS_FLOAT32: {
      std::vector<float> vec_value(static_cast<float *>(value), static_cast<float *>(value) + vec_size);
      value_ptr = mindspore::MakeValue(vec_value);
      break;
    }
    default:
      MS_LOG(ERROR) << "Unrecognized datatype w/ DataTypeC ID: " << data_type << std::endl;
      return nullptr;
  }
  return GetRawPtr(res_mgr, value_ptr);
}
