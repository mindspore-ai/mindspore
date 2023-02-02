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
#include "plugin/device/ascend/hal/device/tensorprint_utils.h"
#include <ctime>
#include <fstream>
#include <memory>
#include <string>
#include <thread>
#include "ir/tensor.h"
#include "pybind11/pybind11.h"
#include "include/common/utils/utils.h"
#include "utils/shape_utils.h"
#include "utils/file_utils.h"
#include "utils/ms_context.h"
#include "ir/dtype/type.h"
#include "proto/print.pb.h"
#include "plugin/device/ascend/hal/device/ascend_data_queue.h"

namespace py = pybind11;
namespace mindspore::device::ascend {
namespace {
std::thread g_acl_tdt_print = {};

const std::map<aclDataType, TypeId> kPrintAclDataTypeMap = {
  {ACL_INT8, TypeId::kNumberTypeInt8},       {ACL_UINT8, TypeId::kNumberTypeUInt8},
  {ACL_INT16, TypeId::kNumberTypeInt16},     {ACL_UINT16, TypeId::kNumberTypeUInt16},
  {ACL_INT32, TypeId::kNumberTypeInt32},     {ACL_UINT32, TypeId::kNumberTypeUInt32},
  {ACL_INT64, TypeId::kNumberTypeInt64},     {ACL_UINT64, TypeId::kNumberTypeUInt64},
  {ACL_FLOAT16, TypeId::kNumberTypeFloat16}, {ACL_FLOAT, TypeId::kNumberTypeFloat32},
  {ACL_DOUBLE, TypeId::kNumberTypeFloat64},  {ACL_BOOL, TypeId::kNumberTypeBool}};

const std::map<aclDataType, size_t> kAclDataTypeSizeMap = {
  {ACL_INT8, sizeof(int8_t)},     {ACL_UINT8, sizeof(uint8_t)},   {ACL_INT16, sizeof(int16_t)},
  {ACL_UINT16, sizeof(uint16_t)}, {ACL_INT32, sizeof(int32_t)},   {ACL_UINT32, sizeof(uint32_t)},
  {ACL_INT64, sizeof(int64_t)},   {ACL_UINT64, sizeof(uint64_t)}, {ACL_FLOAT16, sizeof(float) / 2},
  {ACL_FLOAT, sizeof(float)},     {ACL_DOUBLE, sizeof(double)},   {ACL_BOOL, sizeof(bool)}};

const std::map<aclDataType, std::string> kPrintTensorParseMap = {
  {ACL_INT8, "Int8"},       {ACL_UINT8, "UInt8"},   {ACL_INT16, "Int16"},    {ACL_UINT16, "UInt16"},
  {ACL_INT32, "Int32"},     {ACL_UINT32, "UInt32"}, {ACL_INT64, "Int64"},    {ACL_UINT64, "UInt64"},
  {ACL_FLOAT16, "Float16"}, {ACL_FLOAT, "Float32"}, {ACL_DOUBLE, "Float64"}, {ACL_BOOL, "Bool"}};

std::string GetParseType(const aclDataType &acl_data_type) {
  auto type_iter = kPrintTensorParseMap.find(acl_data_type);
  if (type_iter == kPrintTensorParseMap.end()) {
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
void PrintScalarToString(const void *str_data_ptr, const aclDataType &acl_data_type, std::ostringstream *const buf) {
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

void ConvertDataItem2Scalar(const void *str_data_ptr, const aclDataType &acl_data_type, std::ostringstream *const buf) {
  MS_EXCEPTION_IF_NULL(str_data_ptr);
  MS_EXCEPTION_IF_NULL(buf);
  auto type_iter = kPrintAclDataTypeMap.find(acl_data_type);
  auto type_id = type_iter->second;
  if (type_id == TypeId::kNumberTypeBool) {
    PrintScalarToBoolString(reinterpret_cast<const char *>(str_data_ptr), acl_data_type, buf);
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
  auto type_iter = kAclDataTypeSizeMap.find(acl_data_type);
  if (type_iter == kAclDataTypeSizeMap.end()) {
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
    if (AclHandle::GetInstance().GetChannelType() != ChannelType::kMbuf) {
      acl_data = const_cast<char *>(reinterpret_cast<std::string *>(acl_data)->c_str());
    }
    MS_EXCEPTION_IF_NULL(acl_data);

    ShapeVector tensor_shape;
    tensor_shape.resize(dim_num);

    if (acltdtGetDimsFromItem(item, tensor_shape.data(), dim_num) != ACL_SUCCESS) {
      MS_LOG(ERROR) << "ACL failed to get dim-size from acl channel data";
    }

    if ((tensor_shape.size() == 1 && tensor_shape[0] == 0) || tensor_shape.size() == 0) {
      if (!judgeLengthValid(acl_data_size, acl_data_type)) {
        MS_LOG(EXCEPTION) << "Print op receive data length is invalid.";
      }
      ConvertDataItem2Scalar(reinterpret_cast<void *>(acl_data), acl_data_type, &buf);
      continue;
    }

    if (acl_data_type == ACL_STRING) {
      std::string data(reinterpret_cast<const char *>(acl_data), acl_data_size);
      buf << data << std::endl;
    } else {
      auto type_iter = kPrintAclDataTypeMap.find(acl_data_type);
      if (type_iter == kPrintAclDataTypeMap.end()) {
        MS_LOG(ERROR) << "type of tensor need to print is not support " << GetParseType(acl_data_type);
        continue;
      }
      auto type_id = type_iter->second;
      mindspore::tensor::Tensor print_tensor(type_id, tensor_shape);
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
    if (AclHandle::GetInstance().GetChannelType() != ChannelType::kMbuf) {
      acl_data = const_cast<char *>(reinterpret_cast<std::string *>(acl_data)->c_str());
    }
    MS_EXCEPTION_IF_NULL(acl_data);

    ShapeVector tensor_shape;
    tensor_shape.resize(dim_num);

    if (acltdtGetDimsFromItem(item, tensor_shape.data(), dim_num) != ACL_SUCCESS) {
      MS_LOG(ERROR) << "ACL failed to get dim-size from acl channel data";
    }

    if ((tensor_shape.size() == 1 && tensor_shape[0] == 0) || tensor_shape.size() == 0) {
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
      if (tensor_shape.size() > 1 || (tensor_shape.size() == 1 && tensor_shape[0] != 1)) {
        for (const auto &dim : tensor_shape) {
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

void TensorPrintStdOut(const acltdtChannelHandle *acl_handle) {
  int ret = ACL_SUCCESS;
  acltdtDataset *acl_dataset;

  while (true) {
    do {
      acl_dataset = acltdtCreateDataset();
      if (acl_dataset == nullptr) {
        ret = -1;
        MS_LOG(ERROR) << "Failed to create acl dateaset.";
        break;
      }
      // no timeout
      ret = acltdtReceiveTensor(acl_handle, acl_dataset, -1);
      if (AclHandle::GetInstance().GetChannelType() == ChannelType::kMbuf && ret == ACL_ERROR_RT_QUEUE_EMPTY) {
        MS_LOG(DEBUG) << "queue is empty.";
        break;
      }

      if (ret != ACL_SUCCESS) {
        MS_LOG(ERROR) << "AclHandle failed to receive tensor.ret = " << ret;
        break;
      }
      if (ConvertDataset2Tensor(acl_dataset)) {
        ret = -1;
        break;
      }
    } while (0);

    if (acl_dataset != nullptr && acltdtDestroyDataset(acl_dataset) != ACL_SUCCESS) {
      MS_LOG(ERROR) << "Std out: AcltdtDestroyDataset failed.";
      break;
    }

    if (ret != ACL_SUCCESS) {
      break;
    }
  }
}

void TensorPrintOut2File(const acltdtChannelHandle *acl_handle, const std::string &print_file_path) {
  prntpb::Print print;
  ChangeFileMode(print_file_path, S_IWUSR);
  std::fstream output(print_file_path, std::ios::out | std::ios::trunc | std::ios::binary);

  int ret = ACL_SUCCESS;
  acltdtDataset *acl_dataset;
  while (true) {
    do {
      acl_dataset = acltdtCreateDataset();
      if (acl_dataset == nullptr) {
        MS_LOG(ERROR) << "Failed to create acl dateaset.";
        ret = -1;
        break;
      }
      // no timeout
      ret = acltdtReceiveTensor(acl_handle, acl_dataset, -1);
      if (AclHandle::GetInstance().GetChannelType() == ChannelType::kMbuf && ret == ACL_ERROR_RT_QUEUE_EMPTY) {
        MS_LOG(INFO) << "queue is empty.";
        break;
      }
      if (ret != ACL_SUCCESS) {
        MS_LOG(ERROR) << "Acltdt failed to receive tensor.";
        break;
      }

      if (SaveDataset2File(acl_dataset, print_file_path, print, &output)) {
        ret = -1;
        break;
      }
    } while (0);

    if (acl_dataset != nullptr && acltdtDestroyDataset(acl_dataset) != ACL_SUCCESS) {
      MS_LOG(ERROR) << "Out to file: AcltdtDestroyDataset failed.";
      break;
    }

    if (ret != ACL_SUCCESS) {
      break;
    }
  }

  output.close();
  ChangeFileMode(print_file_path, S_IRUSR);
}

void JoinAclPrintThread(std::thread *thread) {
  try {
    if (thread->joinable()) {
      MS_LOG(INFO) << "join acl tdt host receive process";
      thread->join();
    }
  } catch (const std::exception &e) {
    MS_LOG(ERROR) << "tdt thread join failed: " << e.what();
  }
}
}  // namespace

void TensorPrint::operator()() {
  if (print_file_path_ == "") {
    TensorPrintStdOut(acl_handle_);
  } else {
    TensorPrintOut2File(acl_handle_, print_file_path_);
  }
}

bool AclHandle::CreateChannel(uint32_t deviceId, std::string name, size_t capacity) {
  acl_handle_ = acltdtCreateChannelWithCapacity(deviceId, name.c_str(), capacity);
  if (acl_handle_ == nullptr) {
    MS_LOG(INFO) << "try create tdt channel ...";
    const std::string receive_prefix = "TF_RECEIVE_";
    acl_handle_ = acltdtCreateChannel(deviceId, (receive_prefix + name).c_str());
    channel_type_ = ChannelType::kTDT;
  } else {
    MS_LOG(INFO) << "created mbuf channel.";
  }
  return acl_handle_ != nullptr;
}

void CreateTensorPrintThread(const PrintThreadCrt &ctr) {
  MS_EXCEPTION_IF_NULL(MsContext::GetInstance());
  if (MsContext::GetInstance()->get_param<bool>(MS_CTX_ENABLE_GE_HETEROGENOUS)) {
    return;
  }
  uint32_t device_id = MsContext::GetInstance()->get_param<uint32_t>(MS_CTX_DEVICE_ID);
  std::string channel_name = "_npu_log";

  const size_t capacity_size = 128;
  if (!AclHandle::GetInstance().CreateChannel(device_id, channel_name, capacity_size)) {
    MS_LOG(EXCEPTION) << "create acl channel failed";
  }
  MS_LOG(INFO) << "Success to create acl channel handle, tsd reference = "
               << MsContext::GetInstance()->get_param<uint32_t>(MS_CTX_TSD_REF) << ".";
  std::string print_file_path = MsContext::GetInstance()->get_param<std::string>(MS_CTX_PRINT_FILE_PATH);
  g_acl_tdt_print = ctr(print_file_path, AclHandle::GetInstance().Get());
}

void DestroyTensorPrintThread() {
  MS_EXCEPTION_IF_NULL(MsContext::GetInstance());
  if (MsContext::GetInstance()->get_param<bool>(MS_CTX_ENABLE_GE_HETEROGENOUS)) {
    return;
  }
  auto acl_handle = AclHandle::GetInstance().Get();
  auto channel_type = AclHandle::GetInstance().GetChannelType();
  if (channel_type == ChannelType::kMbuf) {
    // avoid incorrect execution order in acl function
    const int32_t sleep_time = 500;
    usleep(sleep_time);
  }
  // if TdtHandle::DestroyHandle called at taskmanager, all acl_handle will be set to nullptr;
  // but not joined the print thread, so add a protection to join the thread.
  if (acl_handle == nullptr) {
    MS_LOG(INFO) << "The acl handle has been destroyed and the point is nullptr";
    JoinAclPrintThread(&g_acl_tdt_print);
    return;
  }
  aclError stop_status = acltdtStopChannel(acl_handle);
  if (stop_status != ACL_SUCCESS) {
    MS_LOG(ERROR) << "Failed stop acl data channel and the stop_status is " << stop_status << std::endl;
    return;
  }
  MS_LOG(INFO) << "Succeed stop acl data channel for host queue ";

  if (channel_type != ChannelType::kMbuf) {
    JoinAclPrintThread(&g_acl_tdt_print);
  }
  aclError destroyed_status = acltdtDestroyChannel(acl_handle);
  if (destroyed_status != ACL_SUCCESS) {
    MS_LOG(ERROR) << "Failed destroy acl channel and the destroyed_status is " << destroyed_status << std::endl;
    return;
  }
  if (channel_type == ChannelType::kMbuf) {
    JoinAclPrintThread(&g_acl_tdt_print);
  }
  tdt_handle::DelHandle(&acl_handle);
  MS_LOG(INFO) << "Succeed destroy acl channel";
}
}  // namespace mindspore::device::ascend
