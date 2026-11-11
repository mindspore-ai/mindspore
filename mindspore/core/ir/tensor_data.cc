/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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
#include "ir/tensor_data.h"
#include "base/float16.h"
#include "utils/temp_file_manager.h"

namespace mindspore::tensor {
std::string GetTensorDataString(TypeId data_type, const ShapeVector &shape, void *data, size_t size, size_t ndim,
                                bool use_comma) {
  switch (data_type) {
    case kNumberTypeBool:
      return TensorStringifier<bool>(static_cast<bool *>(data), size, ndim).ToString(data_type, shape, use_comma);
    case kNumberTypeUInt8:
      return TensorStringifier<uint8_t>(static_cast<uint8_t *>(data), size, ndim).ToString(data_type, shape, use_comma);
    case kNumberTypeInt4:
      return TensorStringifier<int8_t>(static_cast<int8_t *>(data), size, ndim).ToString(data_type, shape, use_comma);
    case kNumberTypeInt8:
      return TensorStringifier<int8_t>(static_cast<int8_t *>(data), size, ndim).ToString(data_type, shape, use_comma);
    case kNumberTypeInt16:
      return TensorStringifier<int16_t>(static_cast<int16_t *>(data), size, ndim).ToString(data_type, shape, use_comma);
    case kNumberTypeInt:
    case kNumberTypeInt32:
      return TensorStringifier<int32_t>(static_cast<int32_t *>(data), size, ndim).ToString(data_type, shape, use_comma);
    case kNumberTypeInt64:
      return TensorStringifier<int64_t>(static_cast<int64_t *>(data), size, ndim).ToString(data_type, shape, use_comma);
    case kNumberTypeUInt16:
      return TensorStringifier<uint16_t>(static_cast<uint16_t *>(data), size, ndim)
        .ToString(data_type, shape, use_comma);
    case kNumberTypeUInt32:
      return TensorStringifier<uint32_t>(static_cast<uint32_t *>(data), size, ndim)
        .ToString(data_type, shape, use_comma);
    case kNumberTypeUInt64:
      return TensorStringifier<uint64_t>(static_cast<uint64_t *>(data), size, ndim)
        .ToString(data_type, shape, use_comma);
    case kNumberTypeFloat16:
      return TensorStringifier<float16>(static_cast<float16 *>(data), size, ndim).ToString(data_type, shape, use_comma);
    case kNumberTypeFloat:
      return TensorStringifier<float>(static_cast<float *>(data), size, ndim).ToString(data_type, shape, use_comma);
    case kNumberTypeFloat32:
      return TensorStringifier<float>(static_cast<float *>(data), size, ndim).ToString(data_type, shape, use_comma);
    case kNumberTypeFloat64:
      return TensorStringifier<double>(static_cast<double *>(data), size, ndim).ToString(data_type, shape, use_comma);
#ifndef KERNEL_EXECUTOR_ANDROID
    case kNumberTypeBFloat16:
      return TensorStringifier<bfloat16>(static_cast<bfloat16 *>(data), size, ndim)
        .ToString(data_type, shape, use_comma);
#endif
    case kNumberTypeComplex64:
      return TensorStringifier<ComplexStorage<float>>(static_cast<ComplexStorage<float> *>(data), size, ndim)
        .ToString(data_type, shape, use_comma);
    case kNumberTypeComplex128:
      return TensorStringifier<ComplexStorage<double>>(static_cast<ComplexStorage<double> *>(data), size, ndim)
        .ToString(data_type, shape, use_comma);
    case kObjectTypeString:
      return TensorStringifier<uint8_t>(static_cast<uint8_t *>(data), size, ndim).ToString(data_type, shape, use_comma);
    case kObjectTypeTensorType:
      return TensorStringifier<int>(static_cast<int *>(data), size, ndim).ToString(data_type, shape, use_comma);
    default:
      break;
  }
  MS_LOG(EXCEPTION) << "Cannot construct Tensor because of unsupported data type: " << TypeIdToString(data_type) << ".";
}
}  // namespace mindspore::tensor
