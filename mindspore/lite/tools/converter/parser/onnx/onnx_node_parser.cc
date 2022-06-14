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
#include "src/common/file_utils.h"

namespace mindspore {
namespace lite {
namespace {
constexpr int kMaxValidCharacters = 10;
static std::unordered_map<int, mindspore::TypeId> kOnnxTypeTransferMap = {
  {onnx::TensorProto_DataType_INT8, mindspore::kNumberTypeInt8},
  {onnx::TensorProto_DataType_UINT8, mindspore::kNumberTypeUInt8},
  {onnx::TensorProto_DataType_INT16, mindspore::kNumberTypeInt16},
  {onnx::TensorProto_DataType_INT32, mindspore::kNumberTypeInt32},
  {onnx::TensorProto_DataType_UINT32, mindspore::kNumberTypeUInt32},
  {onnx::TensorProto_DataType_INT64, mindspore::kNumberTypeInt64},
  {onnx::TensorProto_DataType_FLOAT16, mindspore::kNumberTypeFloat16},
  {onnx::TensorProto_DataType_FLOAT, mindspore::kNumberTypeFloat32},
  {onnx::TensorProto_DataType_DOUBLE, mindspore::kNumberTypeFloat64},
  {onnx::TensorProto_DataType_BOOL, mindspore::kNumberTypeBool}};
}  // namespace

int64_t OnnxNodeParser::opset_version_ = 0;

STATUS ExternalDataInfo::Create(const google::protobuf::RepeatedPtrField<onnx::StringStringEntryProto> &external_data,
                                ExternalDataInfo *external_data_info) {
  const int data_size = external_data.size();
  for (int i = 0; i != data_size; ++i) {
    onnx::StringStringEntryProto string_map = external_data[i];
    if (!string_map.has_key()) {
      MS_LOG(ERROR) << "No key is in external data.";
      return RET_ERROR;
    }
    if (!string_map.has_value()) {
      MS_LOG(ERROR) << "No value is in external data.";
      return RET_ERROR;
    }

    if (StringMapKeyIs("location", string_map)) {
      external_data_info->relative_path_ = string_map.value();
    } else if (StringMapKeyIs("offset", string_map)) {
      external_data_info->offset_ = strtol(string_map.value().c_str(), nullptr, kMaxValidCharacters);
      if (std::to_string(external_data_info->offset_).length() != string_map.value().length()) {
        MS_LOG(ERROR) << "Failed to parse offset with size " << std::to_string(external_data_info->offset_).length()
                      << ", expected size is " << string_map.value().length();
        return RET_ERROR;
      }
    } else if (StringMapKeyIs("length", string_map)) {
      external_data_info->length_ =
        static_cast<size_t>(strtol(string_map.value().c_str(), nullptr, kMaxValidCharacters));
      if (std::to_string(external_data_info->length_).length() != string_map.value().length()) {
        MS_LOG(ERROR) << "Failed to parse length with size " << std::to_string(external_data_info->offset_).length()
                      << ", expected size is " << string_map.value().length();
        return RET_ERROR;
      }
    } else if (StringMapKeyIs("checksum", string_map)) {
      external_data_info->checksum_ = string_map.value();
    } else {
      MS_LOG(ERROR) << "Invalid model format";
      return RET_ERROR;
    }
  }
  return RET_OK;
}

bool ExternalDataInfo::StringMapKeyIs(const std::string &key, const onnx::StringStringEntryProto &string_map) {
  return string_map.key() == key && !string_map.value().empty();
}

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

STATUS OnnxNodeParser::LoadOnnxExternalTensorData(const onnx::TensorProto &onnx_const_tensor,
                                                  const tensor::TensorPtr &tensor_info, const std::string &model_file) {
  if (tensor_info == nullptr) {
    MS_LOG(ERROR) << "tensor_info is nullptr.";
    return RET_NULL_PTR;
  }
  size_t data_size = 0;
  const void *onnx_data = LoadOnnxRawData(onnx_const_tensor, &data_size, model_file);
  if (onnx_data == nullptr) {
    MS_LOG(ERROR) << "origin data from external data is nullptr.";
    return RET_MEMORY_FAILED;
  }
  auto tensor_data = reinterpret_cast<uint8_t *>(tensor_info->data_c());
  if (memcpy_s(tensor_data, tensor_info->data().nbytes(), onnx_data, data_size) != EOK) {
    MS_LOG(ERROR) << "memcpy_s from onnx tensor data to mindspore tensor data failed.";
    return RET_ERROR;
  }
  return RET_OK;
}

STATUS OnnxNodeParser::SetExternalTensorFile(const std::string &model_file, std::string *external_tensor_dir) {
  auto i_end_index = model_file.find_last_of('/');
  if (i_end_index == std::string::npos) {
    i_end_index = model_file.find_last_of('\\');
  }
  if (i_end_index == std::string::npos) {
    *external_tensor_dir = ".";
  } else {
    *external_tensor_dir = model_file.substr(0, i_end_index);
  }
  return RET_OK;
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

const void *OnnxNodeParser::LoadOnnxRawData(const onnx::TensorProto &onnx_const_tensor, size_t *data_size,
                                            const std::string &model_file) {
  MS_ASSERT(data_size != nullptr);
  ExternalDataInfo external_data_info;
  if (ExternalDataInfo::Create(onnx_const_tensor.external_data(), &external_data_info) != RET_OK) {
    MS_LOG(ERROR) << "Create ExternalDataInfo failed.";
    return nullptr;
  }
  std::string external_tensor_dir;
  if (SetExternalTensorFile(model_file, &external_tensor_dir) != RET_OK) {
    MS_LOG(ERROR) << "Failed to set external tensor file.";
    return nullptr;
  }
#ifdef _WIN32
  std::string external_data_file = external_tensor_dir + "\\" + external_data_info.GetRelativePath();
#else
  std::string external_data_file = external_tensor_dir + "/" + external_data_info.GetRelativePath();
#endif
  return ReadFile(external_data_file.c_str(), data_size);
}
}  // namespace lite
}  // namespace mindspore
