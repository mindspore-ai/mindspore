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
#include "utils/tensorprint_utils.h"
#include <atomic>
#include <fstream>
#include <memory>
#include <string>
#include <vector>
#include "ir/tensor.h"
#include "pybind11/pybind11.h"
#include "utils/ms_utils.h"
#ifndef NO_DLIB
#include "tdt/tsd_client.h"
#include "tdt/tdt_host_interface.h"
#include "tdt/data_common.h"
#endif

namespace py = pybind11;
namespace mindspore {
const char kShapeSeperator[] = ",";
const char kShapeScalar[] = "[0]";
const char kShapeNone[] = "[]";
static std::map<std::string, TypeId> print_type_map = {
  {"int8_t", TypeId::kNumberTypeInt8},     {"uint8_t", TypeId::kNumberTypeUInt8},
  {"int16_t", TypeId::kNumberTypeInt16},   {"uint16_t", TypeId::kNumberTypeUInt16},
  {"int32_t", TypeId::kNumberTypeInt32},   {"uint32_t", TypeId::kNumberTypeUInt32},
  {"int64_t", TypeId::kNumberTypeInt64},   {"uint64_t", TypeId::kNumberTypeUInt64},
  {"float16", TypeId::kNumberTypeFloat16}, {"float", TypeId::kNumberTypeFloat32},
  {"double", TypeId::kNumberTypeFloat64},  {"bool", TypeId::kNumberTypeBool}};

static std::map<std::string, size_t> type_size_map = {
  {"int8_t", sizeof(int8_t)},     {"uint8_t", sizeof(uint8_t)},   {"int16_t", sizeof(int16_t)},
  {"uint16_t", sizeof(uint16_t)}, {"int32_t", sizeof(int32_t)},   {"uint32_t", sizeof(uint32_t)},
  {"int64_t", sizeof(int64_t)},   {"uint64_t", sizeof(uint64_t)}, {"float16", sizeof(float) / 2},
  {"float", sizeof(float)},       {"double", sizeof(double)},     {"bool", sizeof(bool)}};

std::string GetParseType(const std::string &tensorType_) {
  static const std::map<std::string, std::string> print_parse_map = {
    {"int8_t", "Int8"},     {"uint8_t", "Uint8"},   {"int16_t", "Int16"},  {"uint16_t", "Uint16"},
    {"int32_t", "Int32"},   {"uint32_t", "Uint32"}, {"int64_t", "Int64"},  {"uint64_t", "Uint64"},
    {"float16", "Float16"}, {"float", "Float32"},   {"double", "Float64"}, {"bool", "Bool"}};
  auto type_iter = print_parse_map.find(tensorType_);
  if (type_iter == print_parse_map.end()) {
    MS_LOG(EXCEPTION) << "type of tensor need to print is not support " << tensorType_;
  }
  return type_iter->second;
}

bool ParseTensorShape(const std::string &input_shape_str, std::vector<int> *const tensor_shape, size_t *dims) {
  if (tensor_shape == nullptr) {
    return false;
  }
  MS_EXCEPTION_IF_NULL(dims);
  std::string shape_str = input_shape_str;
  if (shape_str.size() <= 2) {
    return false;
  }
  (void)shape_str.erase(shape_str.begin());
  shape_str.pop_back();
  shape_str += kShapeSeperator;
  string::size_type pos_begin = 0;
  string::size_type pos_end = shape_str.find(kShapeSeperator);
  while (pos_end != std::string::npos) {
    string dim_str = shape_str.substr(pos_begin, pos_end - pos_begin);
    tensor_shape->emplace_back(std::stoi(dim_str));
    (*dims) = (*dims) * std::stoul(dim_str);
    pos_begin = pos_end + sizeof(kShapeSeperator) - 1;
    pos_end = shape_str.find(kShapeSeperator, pos_begin);
  }
  return true;
}

bool PrintTensorToString(const char *str_data_ptr, mindspore::tensor::Tensor *const print_tensor,
                         const size_t &memory_size) {
  MS_EXCEPTION_IF_NULL(str_data_ptr);
  MS_EXCEPTION_IF_NULL(print_tensor);
  auto *tensor_data_ptr = static_cast<uint8_t *>(print_tensor->data_c());
  MS_EXCEPTION_IF_NULL(tensor_data_ptr);
  auto cp_ret =
    memcpy_s(tensor_data_ptr, static_cast<size_t>(print_tensor->data().nbytes()), str_data_ptr, memory_size);
  if (cp_ret != EOK) {
    MS_LOG(ERROR) << "Print op Failed to copy the memory to py::tensor " << cp_ret;
    return false;
  }
  return true;
}

template <typename T>
void PrintScalarToString(const char *str_data_ptr, const string &tensor_type, std::ostringstream *const buf) {
  MS_EXCEPTION_IF_NULL(str_data_ptr);
  MS_EXCEPTION_IF_NULL(buf);
  *buf << "Tensor shape:[1] " << tensor_type;
  *buf << "\nval:";
  const T *data_ptr = reinterpret_cast<const T *>(str_data_ptr);
  if constexpr (std::is_same<T, int8_t>::value || std::is_same<T, uint8_t>::value) {
    const int int_data = static_cast<int>(*data_ptr);
    *buf << int_data << "\n";
  } else {
    *buf << *data_ptr << "\n";
  }
}

void PrintScalarToBoolString(const char *str_data_ptr, const string &tensor_type, std::ostringstream *const buf) {
  MS_EXCEPTION_IF_NULL(str_data_ptr);
  MS_EXCEPTION_IF_NULL(buf);
  const bool *data_ptr = reinterpret_cast<const bool *>(str_data_ptr);
  *buf << "Tensor shape:[1] " << tensor_type;
  *buf << "\nval:";
  if (*data_ptr) {
    *buf << "True\n";
  } else {
    *buf << "False\n";
  }
}

void convertDataItem2Scalar(const char *str_data_ptr, const string &tensor_type, std::ostringstream *const buf) {
  MS_EXCEPTION_IF_NULL(str_data_ptr);
  MS_EXCEPTION_IF_NULL(buf);
  auto type_iter = print_type_map.find(tensor_type);
  auto type_id = type_iter->second;
  if (type_id == TypeId::kNumberTypeBool) {
    PrintScalarToBoolString(str_data_ptr, tensor_type, buf);
  } else if (type_id == TypeId::kNumberTypeInt8) {
    PrintScalarToString<int8_t>(str_data_ptr, tensor_type, buf);
  } else if (type_id == TypeId::kNumberTypeUInt8) {
    PrintScalarToString<uint8_t>(str_data_ptr, tensor_type, buf);
  } else if (type_id == TypeId::kNumberTypeInt16) {
    PrintScalarToString<int16_t>(str_data_ptr, tensor_type, buf);
  } else if (type_id == TypeId::kNumberTypeUInt16) {
    PrintScalarToString<uint16_t>(str_data_ptr, tensor_type, buf);
  } else if (type_id == TypeId::kNumberTypeInt32) {
    PrintScalarToString<int32_t>(str_data_ptr, tensor_type, buf);
  } else if (type_id == TypeId::kNumberTypeUInt32) {
    PrintScalarToString<uint32_t>(str_data_ptr, tensor_type, buf);
  } else if (type_id == TypeId::kNumberTypeInt64) {
    PrintScalarToString<int64_t>(str_data_ptr, tensor_type, buf);
  } else if (type_id == TypeId::kNumberTypeUInt64) {
    PrintScalarToString<uint64_t>(str_data_ptr, tensor_type, buf);
  } else if (type_id == TypeId::kNumberTypeFloat16) {
    PrintScalarToString<float16>(str_data_ptr, tensor_type, buf);
  } else if (type_id == TypeId::kNumberTypeFloat32) {
    PrintScalarToString<float>(str_data_ptr, tensor_type, buf);
  } else if (type_id == TypeId::kNumberTypeFloat64) {
    PrintScalarToString<double>(str_data_ptr, tensor_type, buf);
  } else {
    MS_LOG(EXCEPTION) << "Cannot print scalar because of unsupported data type: " << tensor_type << ".";
  }
}

bool judgeLengthValid(const size_t str_len, const string &tensor_type) {
  auto type_iter = type_size_map.find(tensor_type);
  if (type_iter == type_size_map.end()) {
    MS_LOG(EXCEPTION) << "type of scalar to print is not support.";
  }
  return str_len == type_iter->second;
}

#ifndef NO_DLIB
bool ConvertDataItem2Tensor(const std::vector<tdt::DataItem> &items) {
  //  Acquire Python GIL
  py::gil_scoped_acquire gil_acquire;
  std::ostringstream buf;
  bool ret_end_sequence = false;
  for (auto &item : items) {
    if (item.dataType_ == tdt::TDT_END_OF_SEQUENCE) {
      ret_end_sequence = true;
      break;
    }
    std::shared_ptr<std::string> str_data_ptr = std::static_pointer_cast<std::string>(item.dataPtr_);
    MS_EXCEPTION_IF_NULL(str_data_ptr);
    if (item.tensorShape_ == kShapeScalar || item.tensorShape_ == kShapeNone) {
      if (!judgeLengthValid(str_data_ptr->size(), item.tensorType_)) {
        MS_LOG(EXCEPTION) << "Print op receive data length is invalid.";
      }
      convertDataItem2Scalar(str_data_ptr->data(), item.tensorType_, &buf);
      continue;
    }

    std::vector<int> tensor_shape;
    size_t totaldims = 1;
    if (!ParseTensorShape(item.tensorShape_, &tensor_shape, &totaldims)) {
      MS_LOG(ERROR) << "Tensor print can not parse tensor shape, receive info" << item.tensorShape_;
      continue;
    }

    if (item.tensorType_ == "string") {
      std::string data(reinterpret_cast<const char *>(str_data_ptr->c_str()), item.dataLen_);
      buf << data << std::endl;
    } else {
      auto type_iter = print_type_map.find(item.tensorType_);
      if (type_iter == print_type_map.end()) {
        MS_LOG(ERROR) << "type of tensor need to print is not support " << item.tensorType_;
        continue;
      }
      auto type_id = type_iter->second;
      mindspore::tensor::Tensor print_tensor(type_id, tensor_shape);
      auto memory_size = totaldims * type_size_map[item.tensorType_];
      if (PrintTensorToString(str_data_ptr->data(), &print_tensor, memory_size)) {
        buf << print_tensor.ToStringRepr() << std::endl;
      }
    }
  }
  std::cout << buf.str() << std::endl;
  return ret_end_sequence;
}

bool SaveDataItem2File(const std::vector<tdt::DataItem> &items, const std::string &print_file_path, prntpb::Print print,
                       std::fstream *output) {
  bool ret_end_thread = false;
  for (auto &item : items) {
    if (item.dataType_ == tdt::TDT_END_OF_SEQUENCE) {
      ret_end_thread = true;
      break;
    }
    prntpb::Print_Value *value = print.add_value();
    std::shared_ptr<std::string> str_data_ptr = std::static_pointer_cast<std::string>(item.dataPtr_);
    MS_EXCEPTION_IF_NULL(str_data_ptr);
    if (item.tensorShape_ == kShapeScalar || item.tensorShape_ == kShapeNone) {
      if (!judgeLengthValid(str_data_ptr->size(), item.tensorType_)) {
        MS_LOG(ERROR) << "Print op receive data length is invalid.";
        ret_end_thread = true;
      }
    }

    std::vector<int> tensor_shape;
    size_t totaldims = 1;
    if (!ParseTensorShape(item.tensorShape_, &tensor_shape, &totaldims)) {
      MS_LOG(ERROR) << "Tensor print can not parse tensor shape, receive info" << item.tensorShape_;
      ret_end_thread = true;
    }

    if (item.tensorType_ == "string") {
      std::string data(reinterpret_cast<const char *>(str_data_ptr->c_str()), item.dataLen_);
      value->set_desc(data);
    } else {
      auto parse_type = GetParseType(item.tensorType_);
      prntpb::TensorProto *tensor = value->mutable_tensor();
      if (!(item.tensorShape_ == kShapeScalar) && !(item.tensorShape_ == kShapeNone)) {
        for (const auto &dim : tensor_shape) {
          tensor->add_dims(static_cast<::google::protobuf::int64>(dim));
        }
      }
      tensor->set_tensor_type(parse_type);
      std::string data(reinterpret_cast<const char *>(str_data_ptr->c_str()), item.dataLen_);
      tensor->set_tensor_content(data);
    }

    if (!print.SerializeToOstream(output)) {
      MS_LOG(ERROR) << "Save print file:" << print_file_path << " fail.";
      ret_end_thread = true;
      break;
    }
    print.Clear();
  }
  return ret_end_thread;
}

void TensorPrint::operator()() {
  prntpb::Print print;
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  std::string print_file_path = ms_context->print_file_path();
  if (print_file_path == "") {
    while (true) {
      std::vector<tdt::DataItem> bundle;
      if (tdt::TdtHostPopData("_npu_log", bundle) != 0) {
        break;
      }
      if (ConvertDataItem2Tensor(bundle)) {
        break;
      }
    }
  } else {
    std::fstream output(print_file_path, std::ios::out | std::ios::trunc | std::ios::binary);
    while (true) {
      std::vector<tdt::DataItem> bundle;
      if (tdt::TdtHostPopData("_npu_log", bundle) != 0) {
        break;
      }
      if (SaveDataItem2File(bundle, print_file_path, print, &output)) {
        break;
      }
    }
    output.close();
    std::string path_string = print_file_path;
    if (chmod(common::SafeCStr(path_string), S_IRUSR) == -1) {
      MS_LOG(ERROR) << "Modify file:" << print_file_path << " to r fail.";
      return;
    }
  }
}
#endif
}  // namespace mindspore
