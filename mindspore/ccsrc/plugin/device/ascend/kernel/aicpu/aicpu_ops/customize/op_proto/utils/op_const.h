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
#include "external/graph/operator.h"
#include "graph/utils/op_desc_utils.h"
#include "runtime/tiling_context.h"
#include "runtime/infer_shape_context.h"
#include "op_util.h"
#include "context_util.h"

namespace ops {
using namespace ge;

template <typename T1, typename T2>
inline void GetDataToVector(const uint8_t *const_data, size_t data_size, std::vector<T1> &result) {
  size_t size = data_size / sizeof(T2);
  result.resize(size);
  const T2 *data = reinterpret_cast<const T2 *>(const_data);
  for (size_t i = 0; i < size; i++) {
    result[i] = *(data + i);
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
bool GetConstIntData(const ge::Operator &paras, const int64_t const_input_idx, std::vector<T> &values) {
  auto op_desc = OpDescUtils::GetOpDescFromOperator(paras);
  ConstGeTensorBarePtr const_tensor = OpDescUtils::GetInputConstData(paras, const_input_idx);
  if (const_tensor == nullptr) {
    auto input_name = op_desc->GetInputNameByIndex(const_input_idx);
    OP_LOGW("GetConstIntData", "constvalue [%s] is not exists.", input_name.c_str());
    return false;
  }

  const auto &tensor_data = const_tensor->GetData();
  auto data = tensor_data.GetData();
  if (data == nullptr) {
    auto input_name = op_desc->GetInputNameByIndex(const_input_idx);
    OP_LOGW("GetConstIntData", "constvalue [%s] is nullptr.", input_name.c_str());
    return false;
  }
  auto size = tensor_data.GetSize();
  DataType dtype = op_desc->MutableInputDesc(const_input_idx)->GetDataType();
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
bool GetConstInt(const ge::Operator &paras, const int64_t const_input_idx, T &value) {
  auto op_desc = OpDescUtils::GetOpDescFromOperator(paras);
  ConstGeTensorBarePtr const_tensor = OpDescUtils::GetInputConstData(paras, const_input_idx);
  if (const_tensor == nullptr) {
    auto input_name = op_desc->GetInputNameByIndex(const_input_idx);
    OP_LOGW("GetConstIntData", "constvalue [%s] is not exists.", input_name.c_str());
    return false;
  }

  const auto &tensor_data = const_tensor->GetData();
  auto data = tensor_data.GetData();
  if (data == nullptr) {
    auto input_name = op_desc->GetInputNameByIndex(const_input_idx);
    OP_LOGW("GetConstIntData", "constvalue [%s] is nullptr.", input_name.c_str());
    return false;
  }
  DataType dtype = op_desc->MutableInputDesc(const_input_idx)->GetDataType();
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

/*
 * @brief: read constvalue from paras store into value
 * @param [in] context: gert::InferShapeContext
 * @param [in] const_input_idx: constvalue axes index
 * @param [out] value: integer to store return value.
 * @return bool: flag of success or not
 */
template <typename T>
bool GetConstInt(gert::InferShapeContext *context, const int64_t const_input_idx, T &value) {
  auto axes_tensor = context->GetInputTensor(const_input_idx);
  if (axes_tensor == nullptr) {
    OP_LOGW("GetConstIntData", "constvalue is nullptr");
    return false;
  }
  auto dtype = axes_tensor->GetDataType();
  switch (dtype) {
    case ge::DT_UINT64:
      value = static_cast<T>(axes_tensor->GetData<uint64_t>()[0]);
      break;
    case ge::DT_INT64:
      value = static_cast<T>(axes_tensor->GetData<int64_t>()[0]);
      break;
    case ge::DT_UINT32:
      value = static_cast<T>(axes_tensor->GetData<uint32_t>()[0]);
      break;
    case ge::DT_INT32:
      value = static_cast<T>(axes_tensor->GetData<int32_t>()[0]);
      break;
    default: {
      OP_LOGW("GetConstInt", "GetConstInt of dtype[%d] has not implement yet.", dtype);
      return false;
    } break;
  }
  OP_LOGI("GetConstInt", "GetConstInt of value is %d", value);
  return true;
}

/*
 * @brief: read constvalue from paras store into value
 * @param [in] context: gert::TilingContext
 * @param [in] const_input_idx: constvalue axes index
 * @param [out] value: integer to store return value.
 * @return bool: flag of success or not
 */
template <typename T>
bool GetConstInt(gert::TilingContext *context, const int64_t const_input_idx, T &value) {
  const gert::Tensor *const_tensor = context->GetInputTensor(const_input_idx);
  OPS_CHECK_NULL_WITH_CONTEXT_RET(context, const_tensor, false);
  if (!IsConstTensor(const_tensor)) {
    OP_LOGW(context->GetNodeName(), "the input[%ld] is not const tensor, will return failed.", const_input_idx);
    return false;
  }

  ge::DataType dtype = const_tensor->GetDataType();
  switch (dtype) {
    case ge::DT_UINT64:
      value = static_cast<T>(const_tensor->GetData<uint64_t>()[0]);
      break;
    case ge::DT_INT64:
      value = static_cast<T>(const_tensor->GetData<int64_t>()[0]);
      break;
    case ge::DT_UINT32:
      value = static_cast<T>(const_tensor->GetData<uint32_t>()[0]);
      break;
    case ge::DT_INT32:
      value = static_cast<T>(const_tensor->GetData<int32_t>()[0]);
      break;
    default: {
      OP_LOGW(context->GetNodeName(), "GetConstInt only support [int32, int64, uint64, uint32]. but is %s",
              ops::ToString(dtype).c_str());
      return false;
    } break;
  }
  OP_LOGD("GetConstInt", "GetConstInt of value is %d", value);
  return true;
}

template <typename T>
void GetConstValueToShape(const gert::Tensor *tensor, size_t size, gert::Shape *shape) {
  const T *value = tensor->GetData<T>();
  shape->SetDimNum(size);
  for (size_t i = 0; i < size; i++) {
    shape->SetDim(i, value[i]);
  }
}

template <typename T>
void GetValueToShape(const gert::Tensor *const_tensor, gert::Shape *const_shape) {
  const T *const_value = const_tensor->GetData<T>();
  const size_t const_num = const_tensor->GetShapeSize();
  const_shape->SetDimNum(0);
  for (size_t i = 0; i < const_num; ++i) {
    const_shape->AppendDim(const_value[i]);
  }
}

template <typename T>
void GetValueToShape(const gert::Tensor *const_tensor, gert::Shape &const_shape) {
  const T *const_value = const_tensor->GetData<T>();
  const size_t const_num = const_tensor->GetShapeSize();
  const_shape.SetDimNum(0);
  for (size_t i = 0; i < const_num; ++i) {
    const_shape.AppendDim(const_value[i]);
  }
}

template <typename T>
bool GetConstIntToShape(T *context, const int64_t const_idx, gert::Shape &const_shape) {
  const gert::Tensor *const_tensor = context->GetInputTensor(const_idx);
  OPS_CHECK_NULL_WITH_CONTEXT_RET(context, const_tensor, false);
  if (!IsConstTensor(const_tensor)) {
    OP_LOGW(context->GetNodeName(), "the input[%ld] is not const tensor, will return failed.", const_idx);
    return false;
  }

  ge::DataType const_dtype = const_tensor->GetDataType();

  switch (const_dtype) {
    case ge::DT_INT32: {
      GetValueToShape<int32_t>(const_tensor, const_shape);
      break;
    }
    case ge::DT_INT64: {
      GetValueToShape<int64_t>(const_tensor, const_shape);
      break;
    }
    case ge::DT_UINT64: {
      GetValueToShape<uint64_t>(const_tensor, const_shape);
      break;
    }
    case ge::DT_UINT32: {
      GetValueToShape<uint32_t>(const_tensor, const_shape);
      break;
    }
    default:
      OP_LOGW(context->GetNodeName(), "GetConstIntToShape only support [int32, int64, uint64, uint32]. but is %s",
              ops::ToString(const_dtype).c_str());
      return false;
  }

  OP_LOGI(context->GetNodeName(), "GetConstIntToShape: output shape is %s", ToString(const_shape).c_str());
  return true;
}
}  // namespace ops
#endif  // CANN_CUSTOMIZE_OPS_CONST_H_
