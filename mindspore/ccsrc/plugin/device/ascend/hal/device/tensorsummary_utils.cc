/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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
#include "plugin/device/ascend/hal/device/tensorsummary_utils.h"
#include <string>
#include <thread>
#include <vector>
#include <utility>
#include "pybind11/pybind11.h"
#include "ir/tensor.h"
#include "ir/dtype/type.h"
#include "plugin/device/ascend/hal/device/ascend_data_queue.h"
#include "include/common/utils/python_adapter.h"
#include "include/transform/graph_ir/types.h"

namespace py = pybind11;
using mindspore::transform::Status;
namespace mindspore::device::ascend {
const size_t kMbufCapacitySize = 128;
const int32_t kMbufDestroyDelayTime = 500;
const char PYTHON_MOD_CALLBACK_MODULE[] = "mindspore.train.callback._callback";
const char PYTHON_FUN_PROCESS_SUMMARY[] = "summary_cb_for_save_op";

const std::map<aclDataType, TypeId> kPrintAclDataTypeMap = {
  {ACL_INT8, TypeId::kNumberTypeInt8},       {ACL_UINT8, TypeId::kNumberTypeUInt8},
  {ACL_INT16, TypeId::kNumberTypeInt16},     {ACL_UINT16, TypeId::kNumberTypeUInt16},
  {ACL_INT32, TypeId::kNumberTypeInt32},     {ACL_UINT32, TypeId::kNumberTypeUInt32},
  {ACL_INT64, TypeId::kNumberTypeInt64},     {ACL_UINT64, TypeId::kNumberTypeUInt64},
  {ACL_FLOAT16, TypeId::kNumberTypeFloat16}, {ACL_FLOAT, TypeId::kNumberTypeFloat32},
  {ACL_DOUBLE, TypeId::kNumberTypeFloat64},  {ACL_BOOL, TypeId::kNumberTypeBool}};

TensorSummaryUtils &TensorSummaryUtils::GetInstance() {
  static TensorSummaryUtils instance;
  return instance;
}

TDTTensorUtils &TDTTensorUtils::GetInstance() {
  static TDTTensorUtils instance;
  return instance;
}

void TensorSummaryUtils::CreateTDTSummaryThread() {
  MS_EXCEPTION_IF_NULL(MsContext::GetInstance());
  if (MsContext::GetInstance()->get_param<bool>(MS_CTX_ENABLE_GE_HETEROGENOUS)) {
    return;
  }
  std::vector<std::string> channel_names;
  channel_names.push_back("ms_tensor_summary");
  channel_names.push_back("ms_image_summary");
  channel_names.push_back("ms_scalar_summary");
  channel_names.push_back("ms_histogram_summary");

  for (std::string channel_name : channel_names) {
    ChannelType channel_type{ChannelType::kMbuf};
    acltdtChannelHandle *acl_handle = TDTTensorUtils::GetInstance().CreateChannel(channel_name, &channel_type);
    if (acl_handle == nullptr) {
      MS_LOG(ERROR) << "CreateChannel failed: " << channel_name;
      continue;
    }
    std::thread t(GetSummaryData, channel_name, acl_handle);
    t.detach();
    TDTInfo tdt_info;
    tdt_info.channel_name = channel_name;
    tdt_info.acl_handle = acl_handle;
    tdt_info.channel_type = channel_type;
    tdt_info.dtd_thread = &t;
    TDTTensorUtils::GetInstance().tdt_infos.insert(std::pair<std::string, TDTInfo>(channel_name, tdt_info));
  }
}

void TensorSummaryUtils::GetSummaryData(string channel_name, acltdtChannelHandle *acl_handle) {
  TDTTensorUtils::GetInstance().ReceiveData(channel_name, acl_handle);
}

acltdtChannelHandle *TDTTensorUtils::CreateChannel(std::string name, ChannelType *channel_type) {
  uint32_t device_id = MsContext::GetInstance()->get_param<uint32_t>(MS_CTX_DEVICE_ID);
  acltdtChannelHandle *acl_handle_ = acltdtCreateChannelWithCapacity(device_id, name.c_str(), kMbufCapacitySize);
  if (acl_handle_ == nullptr) {
    MS_LOG(INFO) << "For Print ops, select TDT channel.";
    const std::string receive_prefix = "TF_RECEIVE_";
    acl_handle_ = acltdtCreateChannel(device_id, (receive_prefix + name).c_str());
    if (acl_handle_ == nullptr) {
      MS_LOG(EXCEPTION) << "create tdt channel failed";
    }
    *channel_type = ChannelType::kTDT;
  } else {
    MS_LOG(INFO) << "For Print ops, select MBUF channel.";
  }
  return acl_handle_;
}

void TDTTensorUtils::ReceiveData(string channel_name, const acltdtChannelHandle *acl_handle) {
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

      ChannelType channel_type{ChannelType::kMbuf};
      std::map<std::string, TDTInfo>::iterator iter = tdt_infos.find(channel_name);
      if (iter != tdt_infos.end()) {
        channel_type = iter->second.channel_type;
      }

      if (channel_type == ChannelType::kMbuf && ret == ACL_ERROR_RT_QUEUE_EMPTY) {
        MS_LOG(DEBUG) << "queue is empty.";
        break;
      }

      if (ret != ACL_SUCCESS) {
        MS_LOG(ERROR) << "AclHandle failed to receive tensor.ret = " << ret;
        break;
      }
      const char *tensor_name = acltdtGetDatasetName(acl_dataset);
      std::string summary_name = TensorNameToSummaryName(channel_name, tensor_name);
      MS_LOG(DEBUG) << "acltdtReceiveTensor name: " << summary_name;
      bool ret_tensor = ConvertDataset2Tensor(acl_dataset, channel_type, summary_name.c_str());
      if (ret_tensor == false) {
        MS_LOG(WARNING) << "ConvertDataset2Tensor null";
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
  MS_LOG(INFO) << "ReceiveData end channel_name: " << channel_name;
}

std::string TDTTensorUtils::TensorNameToSummaryName(std::string channel_name, std::string tensor_name) {
  std::string suffix = "[:]";
  if (channel_name == "ms_tensor_summary") {
    suffix = "[:Tensor]";
  } else if (channel_name == "ms_image_summary") {
    suffix = "[:Image]";
  } else if (channel_name == "ms_scalar_summary") {
    suffix = "[:Scalar]";
  } else if (channel_name == "ms_histogram_summary") {
    suffix = "[:Histogram]";
  }
  return tensor_name + suffix;
}

bool TDTTensorUtils::ConvertDataset2Tensor(acltdtDataset *acl_dataset, ChannelType channel_type,
                                           const char *summary_name) {
  //  Acquire Python GIL
  bool ret_tensor = false;
  py::gil_scoped_acquire gil_acquire;
  size_t acl_dataset_size = acltdtGetDatasetSize(acl_dataset);

  for (size_t i = 0; i < acl_dataset_size; i++) {
    acltdtDataItem *item = acltdtGetDataItem(acl_dataset, i);
    MS_EXCEPTION_IF_NULL(item);
    if (acltdtGetTensorTypeFromItem(item) == ACL_TENSOR_DATA_END_OF_SEQUENCE) {
      MS_LOG(INFO) << "end of sequence" << std::endl;
      break;
    }

    size_t dim_num = acltdtGetDimNumFromItem(item);
    void *acl_addr = acltdtGetDataAddrFromItem(item);
    size_t acl_data_size = acltdtGetDataSizeFromItem(item);
    aclDataType acl_data_type = acltdtGetDataTypeFromItem(item);

    char *acl_data = reinterpret_cast<char *>(acl_addr);
    if (channel_type != ChannelType::kMbuf) {
      acl_data = reinterpret_cast<std::string *>(acl_data)->data();
    }
    MS_EXCEPTION_IF_NULL(acl_data);

    ShapeVector tensor_shape;
    tensor_shape.resize(dim_num);

    if (acltdtGetDimsFromItem(item, tensor_shape.data(), dim_num) != ACL_SUCCESS) {
      MS_LOG(ERROR) << "ACL failed to get dim-size from acl channel data";
    }

    auto type_iter = kPrintAclDataTypeMap.find(acl_data_type);
    if (type_iter == kPrintAclDataTypeMap.end()) {
      MS_LOG(ERROR) << "type of tensor need to print is not support: " << acl_data_type;
      continue;
    }
    auto type_id = type_iter->second;
    mindspore::tensor::Tensor print_tensor(type_id, tensor_shape);

    if (PrintTensorToString(acl_data, &print_tensor, acl_data_size)) {
      py::list summary_list = py::list();
      py::dict summary_value_dict;
      summary_value_dict["name"] = summary_name;
      summary_value_dict["data"] = print_tensor;
      summary_list.append(summary_value_dict);
      py::bool_ ret = python_adapter::CallPyFn(PYTHON_MOD_CALLBACK_MODULE, PYTHON_FUN_PROCESS_SUMMARY, summary_list);

      auto bool_ret = py::cast<bool>(ret);
      if (!bool_ret) {
        MS_LOG(ERROR) << "Python checkpoint return false during callback";
        ret_tensor = false;
        return ret_tensor;
      }
      ret_tensor = true;
    }
  }
  return ret_tensor;
}

bool TDTTensorUtils::PrintTensorToString(const char *str_data_ptr, mindspore::tensor::Tensor *print_tensor,
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

void TensorSummaryUtils::DestroyTDTSummaryThread() {
  MS_EXCEPTION_IF_NULL(MsContext::GetInstance());
  if (MsContext::GetInstance()->get_param<bool>(MS_CTX_ENABLE_GE_HETEROGENOUS)) {
    return;
  }
  std::map<std::string, TDTInfo> tdt_infos = TDTTensorUtils::GetInstance().tdt_infos;
  for (auto tdt_info : tdt_infos) {
    auto channel_type = tdt_info.second.channel_type;
    MS_LOG(DEBUG) << "begin  destroy channel:" << tdt_info.second.channel_name;

    if (channel_type == ChannelType::kMbuf) {
      // avoid incorrect execution order in acl function
      usleep(kMbufDestroyDelayTime);
    }
    auto acl_handle = tdt_info.second.acl_handle;
    if (acl_handle == nullptr) {
      MS_LOG(INFO) << "The acl handle has been destroyed and the point is nullptr";
      JoinAclPrintThread(tdt_info.second.dtd_thread);
      continue;
    }

    aclError stop_status = acltdtStopChannel(acl_handle);
    if (stop_status != ACL_SUCCESS) {
      MS_LOG(ERROR) << "Failed stop acl data channel and the stop_status is " << stop_status << std::endl;
      return;
    }
    MS_LOG(DEBUG) << "Succeed stop acl data channel for host queue ";

    if (channel_type != ChannelType::kMbuf) {
      JoinAclPrintThread(tdt_info.second.dtd_thread);
    }
    aclError destroyed_status = acltdtDestroyChannel(acl_handle);
    if (destroyed_status != ACL_SUCCESS) {
      MS_LOG(ERROR) << "Failed destroy acl channel and the destroyed_status is " << destroyed_status << std::endl;
      return;
    }
    MS_LOG(INFO) << "Succeed destroy acl data channel for host queue ";
  }
  tdt_infos.clear();
  MS_LOG(INFO) << "Succeed destroy all summary acl channel";
}
}  // namespace mindspore::device::ascend
