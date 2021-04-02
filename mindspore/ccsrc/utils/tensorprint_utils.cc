/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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
#include "utils/shape_utils.h"

namespace py = pybind11;
namespace mindspore {

#ifndef NO_DLIB
static std::map<aclDataType, TypeId> print_acl_data_type_map = {
  {ACL_INT8, TypeId::kNumberTypeInt8},       {ACL_UINT8, TypeId::kNumberTypeUInt8},
  {ACL_INT16, TypeId::kNumberTypeInt16},     {ACL_UINT16, TypeId::kNumberTypeUInt16},
  {ACL_INT32, TypeId::kNumberTypeInt32},     {ACL_UINT32, TypeId::kNumberTypeUInt32},
  {ACL_INT64, TypeId::kNumberTypeInt64},     {ACL_UINT64, TypeId::kNumberTypeUInt64},
  {ACL_FLOAT16, TypeId::kNumberTypeFloat16}, {ACL_FLOAT, TypeId::kNumberTypeFloat32},
  {ACL_DOUBLE, TypeId::kNumberTypeFloat64},  {ACL_BOOL, TypeId::kNumberTypeBool}};

static std::map<aclDataType, size_t> acl_data_type_size_map = {
  {ACL_INT8, sizeof(int8_t)},     {ACL_UINT8, sizeof(uint8_t)},   {ACL_INT16, sizeof(int16_t)},
  {ACL_UINT16, sizeof(uint16_t)}, {ACL_INT32, sizeof(int32_t)},   {ACL_UINT32, sizeof(uint32_t)},
  {ACL_INT64, sizeof(int64_t)},   {ACL_UINT64, sizeof(uint64_t)}, {ACL_FLOAT16, sizeof(float) / 2},
  {ACL_FLOAT, sizeof(float)},     {ACL_DOUBLE, sizeof(double)},   {ACL_BOOL, sizeof(bool)}};

std::string GetParseType(const aclDataType &acl_data_type) {
  static const std::map<aclDataType, std::string> print_tensor_parse_map = {
    {ACL_INT8, "Int8"},       {ACL_UINT8, "Uint8"},   {ACL_INT16, "Int16"},    {ACL_UINT16, "Uint16"},
    {ACL_INT32, "Int32"},     {ACL_UINT32, "Uint32"}, {ACL_INT64, "Int64"},    {ACL_UINT64, "Uint64"},
    {ACL_FLOAT16, "Float16"}, {ACL_FLOAT, "Float32"}, {ACL_DOUBLE, "Float64"}, {ACL_BOOL, "Bool"}};
  auto type_iter = print_tensor_parse_map.find(acl_data_type);
  if (type_iter == print_tensor_parse_map.end()) {
    MS_LOG(EXCEPTION) << "type of tensor need to print is not support " << acl_data_type;
  }
  return type_iter->second;
}

bool PrintTensorToString(const char *str_data_ptr, mindspore::tensor::Tensor *const print_tensor,
                         const size_t &memory_size) {
  MS_EXCEPTION_IF_NULL(str_data_ptr);
  MS_EXCEPTION_IF_NULL(print_tensor);
  auto *tensor_data_ptr = static_cast<uint8_t *>(print_tensor->data_c());
  MS_EXCEPTION_IF_NULL(tensor_data_ptr);

  size_t dest_size = static_cast<size_t>(print_tensor->data().nbytes());
  size_t target_size = memory_size;

  auto cp_ret = memcpy_s(tensor_data_ptr, dest_size, str_data_ptr, target_size);
  if (cp_ret != EOK) {
    MS_LOG(ERROR) << "Print op Failed to copy the memory to py::tensor " << cp_ret;
    return false;
  }
  return true;
}

template <typename T>
void PrintScalarToString(const char *str_data_ptr, const aclDataType &acl_data_type, std::ostringstream *const buf) {
  MS_EXCEPTION_IF_NULL(str_data_ptr);
  MS_EXCEPTION_IF_NULL(buf);
  *buf << "Tensor(shape=[], dtype=" << GetParseType(acl_data_type) << ", value=";
  const T *data_ptr = reinterpret_cast<const T *>(str_data_ptr);
  if constexpr (std::is_same<T, int8_t>::value || std::is_same<T, uint8_t>::value) {
    const int int_data = static_cast<int>(*data_ptr);
    *buf << int_data << ")\n";
  } else {
    *buf << *data_ptr << ")\n";
  }
}

void PrintScalarToBoolString(const char *str_data_ptr, const aclDataType &acl_data_type,
                             std::ostringstream *const buf) {
  MS_EXCEPTION_IF_NULL(str_data_ptr);
  MS_EXCEPTION_IF_NULL(buf);
  const bool *data_ptr = reinterpret_cast<const bool *>(str_data_ptr);
  *buf << "Tensor(shape=[], dtype=" << GetParseType(acl_data_type) << ", value=";
  if (*data_ptr) {
    *buf << "True)\n";
  } else {
    *buf << "False)\n";
  }
}

void convertDataItem2Scalar(const char *str_data_ptr, const aclDataType &acl_data_type, std::ostringstream *const buf) {
  MS_EXCEPTION_IF_NULL(str_data_ptr);
  MS_EXCEPTION_IF_NULL(buf);
  auto type_iter = print_acl_data_type_map.find(acl_data_type);
  auto type_id = type_iter->second;
  if (type_id == TypeId::kNumberTypeBool) {
    PrintScalarToBoolString(str_data_ptr, acl_data_type, buf);
  } else if (type_id == TypeId::kNumberTypeInt8) {
    PrintScalarToString<int8_t>(str_data_ptr, acl_data_type, buf);
  } else if (type_id == TypeId::kNumberTypeUInt8) {
    PrintScalarToString<uint8_t>(str_data_ptr, acl_data_type, buf);
  } else if (type_id == TypeId::kNumberTypeInt16) {
    PrintScalarToString<int16_t>(str_data_ptr, acl_data_type, buf);
  } else if (type_id == TypeId::kNumberTypeUInt16) {
    PrintScalarToString<uint16_t>(str_data_ptr, acl_data_type, buf);
  } else if (type_id == TypeId::kNumberTypeInt32) {
    PrintScalarToString<int32_t>(str_data_ptr, acl_data_type, buf);
  } else if (type_id == TypeId::kNumberTypeUInt32) {
    PrintScalarToString<uint32_t>(str_data_ptr, acl_data_type, buf);
  } else if (type_id == TypeId::kNumberTypeInt64) {
    PrintScalarToString<int64_t>(str_data_ptr, acl_data_type, buf);
  } else if (type_id == TypeId::kNumberTypeUInt64) {
    PrintScalarToString<uint64_t>(str_data_ptr, acl_data_type, buf);
  } else if (type_id == TypeId::kNumberTypeFloat16) {
    PrintScalarToString<float16>(str_data_ptr, acl_data_type, buf);
  } else if (type_id == TypeId::kNumberTypeFloat32) {
    PrintScalarToString<float>(str_data_ptr, acl_data_type, buf);
  } else if (type_id == TypeId::kNumberTypeFloat64) {
    PrintScalarToString<double>(str_data_ptr, acl_data_type, buf);
  } else {
    MS_LOG(EXCEPTION) << "Cannot print scalar because of unsupported data type: " << GetParseType(acl_data_type) << ".";
  }
}

bool judgeLengthValid(const size_t str_len, const aclDataType &acl_data_type) {
  auto type_iter = acl_data_type_size_map.find(acl_data_type);
  if (type_iter == acl_data_type_size_map.end()) {
    MS_LOG(EXCEPTION) << "type of scalar to print is not support.";
  }
  return str_len == type_iter->second;
}

bool ConvertDataset2Tensor(acltdtDataset *acl_dataset) {
  //  Acquire Python GIL
  py::gil_scoped_acquire gil_acquire;
  std::ostringstream buf;
  bool ret_end_sequence = false;

  size_t acl_dataset_size = acltdtGetDatasetSize(acl_dataset);

  for (size_t i = 0; i < acl_dataset_size; i++) {
    acltdtDataItem *item = acltdtGetDataItem(acl_dataset, i);
    if (acltdtGetTensorTypeFromItem(item) == ACL_TENSOR_DATA_END_OF_SEQUENCE) {
      ret_end_sequence = true;
      MS_LOG(INFO) << "end of sequence" << std::endl;
      break;
    }

    size_t dim_num = acltdtGetDimNumFromItem(item);
    void *acl_addr = acltdtGetDataAddrFromItem(item);
    size_t acl_data_size = acltdtGetDataSizeFromItem(item);
    aclDataType acl_data_type = acltdtGetDataTypeFromItem(item);
    char *acl_data = reinterpret_cast<char *>(acl_addr);
    acl_data = const_cast<char *>(reinterpret_cast<std::string *>(acl_data)->c_str());
    MS_EXCEPTION_IF_NULL(acl_data);

    ShapeVector tensorShape;
    tensorShape.resize(dim_num);

    if (acltdtGetDimsFromItem(item, tensorShape.data(), dim_num) != ACL_SUCCESS) {
      MS_LOG(ERROR) << "ACL failed to get dim-size from acl channel data";
    }

    if ((tensorShape.size() == 1 && tensorShape[0] == 0) || tensorShape.size() == 0) {
      if (!judgeLengthValid(acl_data_size, acl_data_type)) {
        MS_LOG(EXCEPTION) << "Print op receive data length is invalid.";
      }
      convertDataItem2Scalar(acl_data, acl_data_type, &buf);
      continue;
    }

    if (acl_data_type == ACL_STRING) {
      std::string data(reinterpret_cast<const char *>(acl_data), acl_data_size);
      buf << data << std::endl;
    } else {
      auto type_iter = print_acl_data_type_map.find(acl_data_type);
      if (type_iter == print_acl_data_type_map.end()) {
        MS_LOG(ERROR) << "type of tensor need to print is not support " << GetParseType(acl_data_type);
        continue;
      }
      auto type_id = type_iter->second;
      mindspore::tensor::Tensor print_tensor(type_id, tensorShape);
      if (PrintTensorToString(acl_data, &print_tensor, acl_data_size)) {
        buf << print_tensor.ToStringNoLimit() << std::endl;
      }
    }
  }
  std::cout << buf.str() << std::endl;
  return ret_end_sequence;
}

bool SaveDataset2File(acltdtDataset *acl_dataset, const std::string &print_file_path, prntpb::Print print,
                      std::fstream *output) {
  bool ret_end_thread = false;

  for (size_t i = 0; i < acltdtGetDatasetSize(acl_dataset); i++) {
    acltdtDataItem *item = acltdtGetDataItem(acl_dataset, i);
    MS_EXCEPTION_IF_NULL(item);
    acltdtTensorType acl_tensor_type = acltdtGetTensorTypeFromItem(item);

    if (acl_tensor_type == ACL_TENSOR_DATA_END_OF_SEQUENCE) {
      MS_LOG(INFO) << "Acl channel received end-of-sequence for print op.";
      ret_end_thread = true;
      break;
    } else if (acl_tensor_type == ACL_TENSOR_DATA_ABNORMAL) {
      MS_LOG(INFO) << "Acl channel received abnormal for print op.";
      return true;
    } else if (acl_tensor_type == ACL_TENSOR_DATA_UNDEFINED) {
      MS_LOG(INFO) << "Acl channel received undefined message type for print op.";
      return false;
    }

    prntpb::Print_Value *value = print.add_value();
    size_t dim_num = acltdtGetDimNumFromItem(item);
    void *acl_addr = acltdtGetDataAddrFromItem(item);
    size_t acl_data_size = acltdtGetDataSizeFromItem(item);
    aclDataType acl_data_type = acltdtGetDataTypeFromItem(item);
    char *acl_data = reinterpret_cast<char *>(acl_addr);
    acl_data = const_cast<char *>(reinterpret_cast<std::string *>(acl_data)->c_str());
    MS_EXCEPTION_IF_NULL(acl_data);

    ShapeVector tensorShape;
    tensorShape.resize(dim_num);

    if (acltdtGetDimsFromItem(item, tensorShape.data(), dim_num) != ACL_SUCCESS) {
      MS_LOG(ERROR) << "ACL failed to get dim-size from acl channel data";
    }

    if ((tensorShape.size() == 1 && tensorShape[0] == 0) || tensorShape.size() == 0) {
      if (!judgeLengthValid(acl_data_size, acl_data_type)) {
        MS_LOG(ERROR) << "Print op receive data length is invalid.";
        ret_end_thread = true;
      }
    }

    if (acl_data_type == ACL_STRING) {
      std::string data(reinterpret_cast<const char *>(acl_data), acl_data_size);
      value->set_desc(data);
    } else {
      auto parse_type = GetParseType(acl_data_type);
      prntpb::TensorProto *tensor = value->mutable_tensor();
      if (tensorShape.size() > 1 || (tensorShape.size() == 1 && tensorShape[0] != 1)) {
        for (const auto &dim : tensorShape) {
          tensor->add_dims(static_cast<::google::protobuf::int64>(dim));
        }
      }

      tensor->set_tensor_type(parse_type);
      std::string data(reinterpret_cast<const char *>(acl_data), acl_data_size);
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
  std::string print_file_path = ms_context->get_param<std::string>(MS_CTX_PRINT_FILE_PATH);
  if (print_file_path == "") {
    while (true) {
      acltdtDataset *acl_dataset = acltdtCreateDataset();
      if (acl_dataset == nullptr) {
        MS_LOG(ERROR) << "Failed to create acl dateaset.";
      }
      if (acltdtReceiveTensor(acl_handle_, acl_dataset, -1 /* no timeout */) != ACL_SUCCESS) {
        MS_LOG(ERROR) << "AclHandle failed to receive tensor.";
        break;
      }
      if (ConvertDataset2Tensor(acl_dataset)) {
        break;
      }
    }
  } else {
    std::fstream output(print_file_path, std::ios::out | std::ios::trunc | std::ios::binary);
    while (true) {
      acltdtDataset *acl_dataset = acltdtCreateDataset();
      if (acl_dataset == nullptr) {
        MS_LOG(ERROR) << "Failed to create acl dateaset.";
      }
      if (acltdtReceiveTensor(acl_handle_, acl_dataset, -1 /* no timeout */) != ACL_SUCCESS) {
        MS_LOG(ERROR) << "Acltdt failed to receive tensor.";
        break;
      }
      if (SaveDataset2File(acl_dataset, print_file_path, print, &output)) {
        break;
      }
    }
    output.close();
    std::string path_string = print_file_path;
    if (chmod(common::SafeCStr(path_string), S_IRUSR) == -1) {
      MS_LOG(ERROR) << "Modify file:" << print_file_path << " fail.";
      return;
    }
  }
}
#endif
}  // namespace mindspore
