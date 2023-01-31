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
  {onnx::TensorProto_DataType_UINT64, mindspore::kNumberTypeUInt64},
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

tensor::TensorPtr OnnxNodeParser::CopyOnnxTensorData(const onnx::TensorProto &onnx_const_tensor) {
  auto onnx_data_type = static_cast<onnx::TensorProto_DataType>(onnx_const_tensor.data_type());
  auto data_type = OnnxNodeParser::GetDataTypeFromOnnx(onnx_data_type);
  if (data_type == kTypeUnknown) {
    MS_LOG(ERROR) << "not support onnx data type " << onnx_data_type;
    return nullptr;
  }
  std::vector<int64_t> shape_vector(onnx_const_tensor.dims().begin(), onnx_const_tensor.dims().end());
  auto tensor_info = std::make_shared<tensor::Tensor>(data_type, shape_vector);
  if (tensor_info == nullptr) {
    MS_LOG(ERROR) << "new a tensor::Tensor failed, data type: " << data_type << ", shape: " << shape_vector;
    return nullptr;
  }
  bool overflow = false;
  auto data_count = GetOnnxElementNum(onnx_const_tensor, &overflow);
  if (overflow) {
    MS_LOG(ERROR) << "data count overflow, tensor shape: " << shape_vector;
    return nullptr;
  }
  if (data_count == 0) {
    return tensor_info;
  }
  auto type_size = lite::DataTypeSize(data_type);
  if (type_size == 0) {
    MS_LOG(ERROR) << "Unsupported data type: " << data_type;
    return nullptr;
  }
  if (INT_MUL_OVERFLOW_THRESHOLD(data_count, type_size, SIZE_MAX)) {
    MS_LOG(ERROR) << "data_size overflow";
    return nullptr;
  }
  auto data_size = data_count * type_size;
  auto tensor_data = tensor_info->data_c();
  if (tensor_data == nullptr) {
    MS_LOG(ERROR) << "Dst tensor cannot be nullptr";
    return nullptr;
  }
  auto dst_bytes_size = tensor_info->data().nbytes();
  if (dst_bytes_size != SizeToLong(data_size)) {
    MS_LOG(ERROR) << "Calculated data size " << data_size << " != tensor bytes size " << dst_bytes_size;
    return nullptr;
  }
  if (onnx_const_tensor.raw_data().size() != 0) {
    auto ret = GetOnnxRawData(onnx_const_tensor, data_count, tensor_info);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "Failed to get tensor data, data count " << data_count << ", data type " << data_type;
      return nullptr;
    }
  } else {
    auto ret = GetOnnxListData(onnx_const_tensor, data_count, tensor_info);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "Failed to get tensor data, data count " << data_count << ", data type " << data_type;
      return nullptr;
    }
  }
  return tensor_info;
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
                                                  const tensor::TensorPtr &tensor_info, const std::string &model_file,
                                                  std::map<std::string, std::pair<size_t, uint8_t *>> *external_datas) {
  if (tensor_info == nullptr) {
    MS_LOG(ERROR) << "tensor_info is nullptr.";
    return RET_NULL_PTR;
  }
  size_t data_size = 0;
  const void *onnx_data = LoadOnnxRawData(onnx_const_tensor, &data_size, model_file, external_datas);
  if (onnx_data == nullptr) {
    MS_LOG(ERROR) << "origin data from external data is nullptr.";
    return RET_MEMORY_FAILED;
  }
  auto tensor_data = reinterpret_cast<uint8_t *>(tensor_info->data_c());
  if (memcpy_s(tensor_data, tensor_info->data().nbytes(), onnx_data, data_size) != EOK) {
    MS_LOG(ERROR) << "memcpy_s from onnx tensor data to mindspore tensor data failed, dst size "
                  << tensor_info->data().nbytes() << ", src size " << data_size;
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

template <class DstT, class SrcT>
static int CopyOnnxData(void *dst_v, const void *src_v, size_t data_count) {
  if (dst_v == nullptr || src_v == nullptr) {
    MS_LOG(ERROR) << "Dst or src data cannot be nullptr";
    return RET_ERROR;
  }
  if (sizeof(DstT) == sizeof(SrcT)) {
    if (memcpy_s(dst_v, data_count * sizeof(DstT), src_v, data_count * sizeof(SrcT)) != EOK) {
      MS_LOG(ERROR) << "memcpy_s failed, data size " << data_count * sizeof(DstT);
      return RET_ERROR;
    }
    return RET_OK;
  }
  auto src = reinterpret_cast<const SrcT *>(src_v);
  auto dst = reinterpret_cast<DstT *>(dst_v);
  for (size_t i = 0; i < data_count; i++) {
    dst[i] = static_cast<DstT>(src[i]);
  }
  return RET_OK;
}

int OnnxNodeParser::GetOnnxRawData(const onnx::TensorProto &onnx_const_tensor, size_t data_count,
                                   const tensor::TensorPtr &tensor_info) {
  auto data_size = LongToSize(tensor_info->data().nbytes());
  auto tensor_data = tensor_info->data_c();
  auto onnx_data = onnx_const_tensor.raw_data().data();
  if (onnx_const_tensor.raw_data().size() != data_size) {
    MS_LOG(ERROR) << "Tensor raw data size " << onnx_const_tensor.raw_data().size() << " != expected size "
                  << data_size;
    return RET_ERROR;
  }
  return CopyOnnxData<uint8_t, uint8_t>(tensor_data, onnx_data, data_size);
}

int OnnxNodeParser::GetOnnxListData(const onnx::TensorProto &onnx_const_tensor, size_t data_count,
                                    const tensor::TensorPtr &tensor_info) {
  const void *onnx_data = nullptr;
  auto tensor_data = tensor_info->data_c();
  TypeId data_type = tensor_info->Dtype()->type_id();
  auto type_size = lite::DataTypeSize(data_type);
  switch (data_type) {
    case kNumberTypeFloat32:
      MS_CHECK_EQ(onnx_const_tensor.float_data_size(), SizeToLong(data_count), RET_ERROR);
      onnx_data = onnx_const_tensor.float_data().data();
      return CopyOnnxData<float, float>(tensor_data, onnx_data, data_count);
    case kNumberTypeFloat64:
      MS_CHECK_EQ(onnx_const_tensor.double_data_size(), SizeToLong(data_count), RET_ERROR);
      onnx_data = onnx_const_tensor.double_data().data();
      return CopyOnnxData<double, double>(tensor_data, onnx_data, data_count);
    case kNumberTypeInt64:
      MS_CHECK_EQ(onnx_const_tensor.int64_data_size(), SizeToLong(data_count), RET_ERROR);
      onnx_data = onnx_const_tensor.int64_data().data();
      return CopyOnnxData<int64_t, int64_t>(tensor_data, onnx_data, data_count);
    case kNumberTypeUInt64:
    case kNumberTypeUInt32:
      MS_CHECK_EQ(onnx_const_tensor.uint64_data_size(), SizeToLong(data_count), RET_ERROR);
      onnx_data = onnx_const_tensor.uint64_data().data();
      if (data_type == kNumberTypeUInt32) {
        return CopyOnnxData<uint32_t, uint64_t>(tensor_data, onnx_data, data_count);
      } else {
        return CopyOnnxData<uint64_t, uint64_t>(tensor_data, onnx_data, data_count);
      }
    case kNumberTypeInt32:
    case kNumberTypeInt16:
    case kNumberTypeInt8:
    case kNumberTypeUInt16:
    case kNumberTypeUInt8:
    case kNumberTypeBool:
    case kNumberTypeFloat16:
      MS_CHECK_EQ(onnx_const_tensor.int32_data_size(), SizeToLong(data_count), RET_ERROR);
      onnx_data = onnx_const_tensor.int32_data().data();
      if (type_size == sizeof(int32_t)) {
        return CopyOnnxData<int32_t, int32_t>(tensor_data, onnx_data, data_count);
      } else if (type_size == sizeof(uint16_t)) {
        return CopyOnnxData<uint16_t, int32_t>(tensor_data, onnx_data, data_count);
      } else if (type_size == sizeof(uint8_t)) {
        return CopyOnnxData<uint8_t, int32_t>(tensor_data, onnx_data, data_count);
      }
      break;
    default:
      break;
  }
  MS_LOG(ERROR) << "unsupported data type " << data_type;
  return RET_ERROR;
}

const void *OnnxNodeParser::LoadOnnxRawData(const onnx::TensorProto &onnx_const_tensor, size_t *data_size,
                                            const std::string &model_file,
                                            std::map<std::string, std::pair<size_t, uint8_t *>> *external_datas) {
  MS_ERROR_IF_NULL_W_RET_VAL(data_size, nullptr);
  MS_ERROR_IF_NULL_W_RET_VAL(external_datas, nullptr);
  ExternalDataInfo external_data_info;
  if (ExternalDataInfo::Create(onnx_const_tensor.external_data(), &external_data_info) != RET_OK) {
    MS_LOG(ERROR) << "Create ExternalDataInfo failed.";
    return nullptr;
  }
  auto data_path = external_data_info.GetRelativePath();
  auto it = external_datas->find(data_path);
  size_t external_data_size = 0;
  uint8_t *external_data = nullptr;
  if (it == external_datas->end()) {
    std::string external_tensor_dir;
    if (SetExternalTensorFile(model_file, &external_tensor_dir) != RET_OK) {
      MS_LOG(ERROR) << "Failed to set external tensor file.";
      return nullptr;
    }
#ifdef _WIN32
    std::string external_data_file = external_tensor_dir + "\\" + data_path;
#else
    std::string external_data_file = external_tensor_dir + "/" + data_path;
#endif
    external_data = reinterpret_cast<uint8_t *>(ReadFile(external_data_file.c_str(), &external_data_size));
    if (external_data == nullptr || external_data_size == 0) {
      MS_LOG(ERROR) << "Failed to read external tensor file " << external_data_file;
      return nullptr;
    }
    external_datas->emplace(data_path, std::make_pair(external_data_size, external_data));
  } else {
    external_data_size = it->second.first;
    external_data = it->second.second;
  }
  auto offset = external_data_info.GetOffset();
  auto length = external_data_info.GetLength();
  if (length == 0 && offset == 0) {  // not set length and offset
    *data_size = external_data_size;
    return external_data;
  }
  if (length == 0 || external_data_size < offset || external_data_size - offset < length) {
    MS_LOG(ERROR) << "Invalid external data info, data path " << data_path << ", offset " << offset << ", length "
                  << length << ", file length " << external_data_size;
    return nullptr;
  }
  *data_size = length;
  return external_data + offset;
}

const onnx::TensorProto *OnnxNodeParser::GetConstantTensorData(const onnx::GraphProto &onnx_graph,
                                                               const std::string &input_name) {
  auto &initializer = onnx_graph.initializer();
  auto init_iter = std::find_if(initializer.begin(), initializer.end(),
                                [input_name](const onnx::TensorProto &proto) { return proto.name() == input_name; });
  if (init_iter != initializer.end()) {
    return &(*init_iter);
  }
  auto &nodes = onnx_graph.node();
  auto node_iter = std::find_if(nodes.begin(), nodes.end(), [input_name](const onnx::NodeProto &proto) {
    if (proto.op_type() != "Constant" || proto.output_size() != 1) {
      return false;
    }
    return proto.output(0) == input_name;
  });
  if (node_iter == nodes.end()) {
    MS_LOG(ERROR) << "Cannot find const input " << input_name;
    return nullptr;
  }
  auto &onnx_node = *node_iter;
  for (const auto &onnx_node_attr : onnx_node.attribute()) {
    const auto &attribute_name = onnx_node_attr.name();
    if (attribute_name == "value") {
      if (onnx_node_attr.has_t()) {
        return &onnx_node_attr.t();
      }
      break;
    }
  }
  MS_LOG(ERROR) << "Failed to find const value from input " << input_name;
  return nullptr;
}
}  // namespace lite
}  // namespace mindspore
