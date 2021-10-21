/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#include "tools/converter/parser/onnx/onnx_node_parser.h"
#include <algorithm>
#include <vector>
#include <memory>
#include <unordered_map>
#include "tools/converter/parser/onnx/onnx_model_parser.h"
#include "nnacl/op_base.h"

namespace mindspore {
namespace lite {
namespace {
static std::unordered_map<int, mindspore::TypeId> kOnnxTypeTransferMap = {
  {onnx::TensorProto_DataType_INT8, mindspore::kNumberTypeInt8},
  {onnx::TensorProto_DataType_UINT8, mindspore::kNumberTypeUInt8},
  {onnx::TensorProto_DataType_INT16, mindspore::kNumberTypeInt16},
  {onnx::TensorProto_DataType_INT32, mindspore::kNumberTypeInt32},
  {onnx::TensorProto_DataType_UINT32, mindspore::kNumberTypeUInt32},
  {onnx::TensorProto_DataType_INT64, mindspore::kNumberTypeInt64},
  {onnx::TensorProto_DataType_FLOAT16, mindspore::kNumberTypeFloat16},
  {onnx::TensorProto_DataType_FLOAT, mindspore::kNumberTypeFloat32},
  {onnx::TensorProto_DataType_BOOL, mindspore::kNumberTypeBool}};
}  // namespace

int64_t OnnxNodeParser::opset_version_ = 0;

mindspore::PadMode OnnxNodeParser::GetOnnxPadMode(const onnx::AttributeProto &onnx_node_attr) {
  if (onnx_node_attr.s() == "NOTSET") {
    return mindspore::PadMode::PAD;
  } else if (onnx_node_attr.s() == "SAME_UPPER" || onnx_node_attr.s() == "SAME_LOWER") {
    return mindspore::PadMode::SAME;
  } else if (onnx_node_attr.s() == "VALID") {
    return mindspore::PadMode::VALID;
  } else {
    MS_LOG(ERROR) << "unsupported padMode";
    return mindspore::PadMode::PAD;
  }
}

STATUS OnnxNodeParser::CopyOnnxTensorData(const onnx::TensorProto &onnx_const_tensor,
                                          const tensor::TensorPtr &tensor_info) {
  if (tensor_info == nullptr) {
    MS_LOG(ERROR) << "tensor_info is nullptr.";
    return RET_NULL_PTR;
  }
  bool overflow = false;
  auto data_count = GetOnnxElementNum(onnx_const_tensor, &overflow);
  if (overflow) {
    MS_LOG(ERROR) << "data count overflow";
    return RET_ERROR;
  }
  size_t data_size = 0;
  auto data_type = GetDataTypeFromOnnx(static_cast<onnx::TensorProto_DataType>(onnx_const_tensor.data_type()));
  const void *onnx_data = GetOnnxRawData(onnx_const_tensor, data_type, data_count, &data_size);
  if (data_size == 0) {
    return RET_OK;
  }
  if (onnx_data == nullptr) {
    MS_LOG(ERROR) << "origin data in onnx model is nullptr";
    return RET_MEMORY_FAILED;
  }
  auto tensor_data = reinterpret_cast<uint8_t *>(tensor_info->data_c());
  if (memcpy_s(tensor_data, tensor_info->data().nbytes(), onnx_data, data_size) != EOK) {
    MS_LOG(ERROR) << "memcpy_s failed";
    return RET_ERROR;
  }
  return RET_OK;
}

TypeId OnnxNodeParser::GetDataTypeFromOnnx(onnx::TensorProto_DataType onnx_type) {
  auto iter = kOnnxTypeTransferMap.find(onnx_type);
  if (iter == kOnnxTypeTransferMap.end()) {
    MS_LOG(ERROR) << "unsupported onnx data type: " << onnx_type;
    return kTypeUnknown;
  }
  return iter->second;
}

STATUS OnnxNodeParser::GetTensorDataFromOnnx(const onnx::TensorProto &onnx_tensor, std::vector<float> *value,
                                             int *type) {
  if (value == nullptr || type == nullptr) {
    MS_LOG(ERROR) << "Input value or type is nullptr";
    return RET_INPUT_PARAM_INVALID;
  }
  bool overflow = false;
  auto data_count = GetOnnxElementNum(onnx_tensor, &overflow);
  if (overflow) {
    MS_LOG(ERROR) << "data count overflow";
    return RET_ERROR;
  }
  switch (onnx_tensor.data_type()) {
    case onnx::TensorProto_DataType_FLOAT:
      *type = GetDataTypeFromOnnx(onnx::TensorProto_DataType_FLOAT);
      if (onnx_tensor.float_data_size() > 0) {
        for (int i = 0; i < onnx_tensor.float_data_size(); i++) {
          value->push_back(onnx_tensor.float_data(i));
        }
      } else {
        for (size_t i = 0; i < data_count; i++) {
          value->push_back(reinterpret_cast<const float *>(onnx_tensor.raw_data().data())[i]);
        }
      }
      break;
    case onnx::TensorProto_DataType_INT32:
      *type = GetDataTypeFromOnnx(onnx::TensorProto_DataType_INT32);
      if (onnx_tensor.int32_data_size() > 0) {
        for (int i = 0; i < onnx_tensor.int32_data_size(); i++) {
          value->push_back(onnx_tensor.int32_data(i));
        }
      } else {
        for (size_t i = 0; i < data_count; i++) {
          value->push_back(static_cast<float>(reinterpret_cast<const int32_t *>(onnx_tensor.raw_data().data())[i]));
        }
      }
      break;
    case onnx::TensorProto_DataType_INT64:
      *type = GetDataTypeFromOnnx(onnx::TensorProto_DataType_INT32);
      if (onnx_tensor.int64_data_size() > 0) {
        for (int i = 0; i < onnx_tensor.int64_data_size(); i++) {
          value->push_back(onnx_tensor.int64_data(i));
        }
      } else {
        for (size_t i = 0; i < data_count; i++) {
          value->push_back(static_cast<float>(reinterpret_cast<const int64_t *>(onnx_tensor.raw_data().data())[i]));
        }
      }
      break;
    default:
      MS_LOG(ERROR) << "The data type is not supported.";
      return RET_ERROR;
  }
  return RET_OK;
}

size_t OnnxNodeParser::GetOnnxElementNum(const onnx::TensorProto &onnx_tensor, bool *overflowed) {
  size_t data_count = 1;
  bool is_overflow = false;
  if (!onnx_tensor.dims().empty()) {
    std::for_each(onnx_tensor.dims().begin(), onnx_tensor.dims().end(), [&data_count, &is_overflow](int dim) {
      if (is_overflow || dim < 0) {
        is_overflow = true;
        data_count = 0;
        return;
      }
      auto udim = static_cast<size_t>(dim);
      if (INT_MUL_OVERFLOW_THRESHOLD(data_count, udim, SIZE_MAX)) {
        is_overflow = true;
        data_count = 0;
        return;
      }
      data_count *= udim;
    });
  }
  if (overflowed != nullptr) {
    *overflowed = is_overflow;
  }
  return data_count;
}

const void *OnnxNodeParser::GetOnnxRawData(const onnx::TensorProto &onnx_const_tensor, TypeId data_type,
                                           size_t data_count, size_t *data_size) {
  MS_ASSERT(data_size != nullptr);
  const void *onnx_data = nullptr;
  switch (data_type) {
    case kNumberTypeFloat32:
      if (INT_MUL_OVERFLOW_THRESHOLD(data_count, sizeof(float), SIZE_MAX)) {
        MS_LOG(ERROR) << "data_size overflow";
        return nullptr;
      }
      *data_size = data_count * sizeof(float);
      if (onnx_const_tensor.float_data_size() == 0) {
        onnx_data = onnx_const_tensor.raw_data().data();
      } else {
        onnx_data = onnx_const_tensor.float_data().data();
      }
      break;
    case kNumberTypeFloat64:
      if (INT_MUL_OVERFLOW_THRESHOLD(data_count, sizeof(double), SIZE_MAX)) {
        MS_LOG(ERROR) << "data_size overflow";
        return nullptr;
      }
      *data_size = data_count * sizeof(double);
      if (onnx_const_tensor.double_data_size() == 0) {
        onnx_data = onnx_const_tensor.raw_data().data();
      } else {
        onnx_data = onnx_const_tensor.double_data().data();
      }
      break;
    case kNumberTypeInt32:
      if (INT_MUL_OVERFLOW_THRESHOLD(data_count, sizeof(int), SIZE_MAX)) {
        MS_LOG(ERROR) << "data_size overflow";
        return nullptr;
      }
      *data_size = data_count * sizeof(int);
      if (onnx_const_tensor.int32_data_size() == 0) {
        onnx_data = onnx_const_tensor.raw_data().data();
      } else {
        onnx_data = onnx_const_tensor.int32_data().data();
      }
      break;
    case kNumberTypeInt64:
      if (INT_MUL_OVERFLOW_THRESHOLD(data_count, sizeof(int64_t), SIZE_MAX)) {
        MS_LOG(ERROR) << "data_size overflow";
        return nullptr;
      }
      *data_size = data_count * sizeof(int64_t);
      if (onnx_const_tensor.int64_data_size() == 0) {
        onnx_data = onnx_const_tensor.raw_data().data();
      } else {
        onnx_data = onnx_const_tensor.int64_data().data();
      }
      break;
    case kNumberTypeUInt8:
    case kNumberTypeInt8:
    case kNumberTypeBool:
      if (INT_MUL_OVERFLOW_THRESHOLD(data_count, sizeof(uint8_t), SIZE_MAX)) {
        MS_LOG(ERROR) << "data_size overflow";
        return nullptr;
      }
      *data_size = data_count * sizeof(uint8_t);
      onnx_data = onnx_const_tensor.raw_data().data();
      break;
    default:
      MS_LOG(ERROR) << "unsupported data type " << data_type;
      return nullptr;
  }
  return onnx_data;
}
}  // namespace lite
}  // namespace mindspore
