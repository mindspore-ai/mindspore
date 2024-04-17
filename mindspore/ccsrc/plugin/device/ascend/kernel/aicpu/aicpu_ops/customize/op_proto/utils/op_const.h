/**
 * Copyright (c) 2022-2022 Huawei Technologies Co., Ltd.  All rights reserved.
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

/*!
 * \file op_const.h
 * \brief
 */

#ifndef CANN_CUSTOMIZE_OPS_CONST_H_
#define CANN_CUSTOMIZE_OPS_CONST_H_

#include <vector>
#include "graph/operator.h"
#include "op_util.h"

namespace ops {
using namespace ge;

template <typename T1, typename T2>
inline void GetDataToVector(const uint8_t *const_data, size_t data_size, std::vector<T1> &result) {
  size_t size = data_size / sizeof(T2);
  result.resize(size);
  const T2 *data = reinterpret_cast<const T2 *>(const_data);
  for (size_t i = 0; i < size; i++) {
    result[i] = static_cast<T1>(*(data + i));
  }
}

/*
 * @brief: read constvalue from paras store into values
 * @param [in] paras: ge::Operator
 * @param [in] const_input_idx: constvalue axes index
 * @param [out] values: vector to store return values.
 * @return bool: flag of success or not
 */
template <typename T>
bool GetConstIntData(const ge::Operator &paras, const std::string &const_input_name, std::vector<T> &values) {
  Tensor const_tensor;
  auto status = paras.GetInputConstData(const_input_name, const_tensor);
  if (status == GRAPH_FAILED) {
    OP_LOGW("GetConstIntData", "constvalue [%s] is not exists.", const_input_name.c_str());
    return false;
  }

  auto data = const_tensor.GetData();
  if (data == nullptr) {
    OP_LOGW("GetConstIntData", "constvalue [%s] is nullptr.", const_input_name.c_str());
    return false;
  }
  auto size = const_tensor.GetSize();
  DataType dtype = paras.GetInputDescByName(const_input_name.c_str()).GetDataType();
  switch (dtype) {
    case DT_UINT64:
      GetDataToVector<T, uint64_t>(data, size, values);
      break;
    case DT_INT64:
      GetDataToVector<T, int64_t>(data, size, values);
      break;
    case DT_UINT32:
      GetDataToVector<T, uint32_t>(data, size, values);
      break;
    case DT_INT32:
      GetDataToVector<T, int32_t>(data, size, values);
      break;
    case DT_FLOAT:
      GetDataToVector<T, float>(data, size, values);
      break;
    default: {
      OP_LOGW("GetConstIntData", "GetConstValue of dtype[%d] has not implement.", dtype);
      return false;
    } break;
  }
  return true;
}

/*
 * @brief: read constvalue from paras store into value
 * @param [in] paras: ge::Operator
 * @param [in] const_input_idx: constvalue axes index
 * @param [out] value: integer to store return value.
 * @return bool: flag of success or not
 */
template <typename T>
bool GetConstInt(const ge::Operator &paras, const std::string &const_input_name, T &value) {
  Tensor const_tensor;
  if (paras.GetInputConstData(const_input_name, const_tensor) == GRAPH_FAILED) {
    OP_LOGW("GetConstIntData", "constvalue [%s] is not exists.", const_input_name.c_str());
    return false;
  }

  auto data = const_tensor.GetData();
  if (data == nullptr) {
    OP_LOGW("GetConstIntData", "constvalue [%s] is nullptr.", const_input_name.c_str());
    return false;
  }
  DataType dtype = paras.GetInputDescByName(const_input_name.c_str()).GetDataType();
  switch (dtype) {
    case DT_UINT64:
      value = static_cast<T>(*reinterpret_cast<const uint64_t *>(data));
      break;
    case DT_INT64:
      value = static_cast<T>(*reinterpret_cast<const int64_t *>(data));
      break;
    case DT_UINT32:
      value = static_cast<T>(*reinterpret_cast<const uint32_t *>(data));
      break;
    case DT_INT32:
      value = static_cast<T>(*reinterpret_cast<const int32_t *>(data));
      break;
    default: {
      OP_LOGW("GetConstInt", "GetConstInt of dtype[%d] has not implement.", dtype);
      return false;
    } break;
  }
  return true;
}
}  // namespace ops
#endif  // CANN_CUSTOMIZE_OPS_CONST_H_
