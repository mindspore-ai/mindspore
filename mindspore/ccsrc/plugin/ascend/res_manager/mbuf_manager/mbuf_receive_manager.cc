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
#include "plugin/ascend/res_manager/mbuf_manager/mbuf_receive_manager.h"
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <ctime>
#include <fstream>
#include <ios>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include "include/utils/env_vars.h"
#include "plugin/ascend/res_manager/device_context_conf/op_debug_conf.h"
#include "ir/tensor.h"
#include "utils/log_adapter.h"
#include "utils/ms_context.h"
#include "plugin/ascend/res_manager/symbol_interface/acl_tdt_symbol.h"
#include "plugin/ascend/res_manager/symbol_interface/symbol_utils.h"
#include "ir/tensor_new.h"

namespace mindspore::device::ascend {

namespace {
// Data may not be received when the process exits; reserve a timeout period.
constexpr std::chrono::milliseconds stop_time_out{100};

bool CopyDataToTensor(const uint8_t *src_addr, mindspore::tensor::TensorPtr tensor_ptr, const size_t size) {
  MS_EXCEPTION_IF_NULL(src_addr);
  MS_EXCEPTION_IF_NULL(tensor_ptr);
  auto *dst_addr = reinterpret_cast<uint8_t *>(tensor_ptr->data_c());
  MS_EXCEPTION_IF_NULL(dst_addr);
  size_t dst_size = static_cast<size_t>(tensor_ptr->DataNBytes());
  MS_EXCEPTION_IF_CHECK_FAIL(dst_size >= size, "The destination size is smaller than the source size.");
  size_t remain_size = size;
  while (remain_size > SECUREC_MEM_MAX_LEN) {
    auto cp_ret = memcpy_s(dst_addr, SECUREC_MEM_MAX_LEN, src_addr, SECUREC_MEM_MAX_LEN);
    if (cp_ret != EOK) {
      MS_LOG(ERROR) << "Failed to copy the memory to py::tensor " << cp_ret;
      return false;
    }
    remain_size -= SECUREC_MEM_MAX_LEN;
    dst_addr += SECUREC_MEM_MAX_LEN;
    src_addr += SECUREC_MEM_MAX_LEN;
  }
  if (remain_size != 0U) {
    auto cp_ret = memcpy_s(dst_addr, remain_size, src_addr, remain_size);
    if (cp_ret != EOK) {
      MS_LOG(ERROR) << "Failed to copy the memory to py::tensor " << cp_ret;
      return false;
    }
  }

  return true;
}

mindspore::tensor::TensorPtr ConvertDataItemToTensorPtr(acltdtDataItem *item) {
  size_t dim_num = CALL_ASCEND_API(acltdtGetDimNumFromItem, item);
  void *acl_addr = CALL_ASCEND_API(acltdtGetDataAddrFromItem, item);
  size_t acl_data_size = CALL_ASCEND_API(acltdtGetDataSizeFromItem, item);
  aclDataType acl_data_type = CALL_ASCEND_API(acltdtGetDataTypeFromItem, item);

  auto acl_data = reinterpret_cast<uint8_t *>(acl_addr);
  if (acl_data_size > 0) {
    MS_EXCEPTION_IF_NULL(acl_data);
  }

  ShapeVector tensor_shape;
  tensor_shape.resize(dim_num);

  if (CALL_ASCEND_API(acltdtGetDimsFromItem, item, tensor_shape.data(), dim_num) != ACL_SUCCESS) {
    MS_LOG(ERROR) << "ACL failed to get dim-size from acl channel data";
    return nullptr;
  }

  auto type_iter = kAclDataTypeMap.find(acl_data_type);
  if (type_iter == kAclDataTypeMap.end()) {
    MS_LOG(ERROR) << "The type of aclData not support: " << acl_data_type;
    return nullptr;
  }
  auto type_id = type_iter->second;
  auto tensor_ptr = tensor::from_spec(type_id, tensor_shape, device::DeviceType::kCPU);
  if (acl_data_size == 0) {
    return tensor_ptr;
  }
  if (CopyDataToTensor(acl_data, tensor_ptr, acl_data_size)) {
    return tensor_ptr;
  }
  return nullptr;
}

std::string &GetRecvStateString(const RecvDataState &state) {
  static std::string state_strings[] = {"ReadyToReceive", "WaitForEndOfSlice", "WaitForSynchronizedSlice"};
  if (state >= RecvDataState::kReadyToRecv && state <= RecvDataState::kWaitForSync) {
    return state_strings[static_cast<uint8_t>(state)];
  }
  MS_LOG(EXCEPTION) << "Invalid receive data state " << state;
}
}  // namespace

bool ScopeAclTdtDataset::ProcessFullTensor(acltdtDataItem *item) {
  aclDataType acl_data_type = CALL_ASCEND_API(acltdtGetDataTypeFromItem, item);
  if (acl_data_type == ACL_STRING) {
    void *acl_addr = CALL_ASCEND_API(acltdtGetDataAddrFromItem, item);
    size_t acl_data_size = CALL_ASCEND_API(acltdtGetDataSizeFromItem, item);
    data_items_.emplace_back(std::string(static_cast<char *>(acl_addr), acl_data_size));
    return true;
  }

  auto tensor_ptr = ConvertDataItemToTensorPtr(item);
  if (tensor_ptr == nullptr) {
    return false;
  }
  data_items_.emplace_back(tensor_ptr);
  return true;
}

bool ScopeAclTdtDataset::ProcessSliceTensor(acltdtDataItem *item) {
  size_t slice_num, slice_id;
  auto ret = CALL_ASCEND_API(acltdtGetSliceInfoFromItem, item, &slice_num, &slice_id);
  if (ret != ACL_SUCCESS) {
    MS_LOG(WARNING) << "Get slice info failed with error code " << ret;
    return false;
  }
  MS_LOG(DEBUG) << "Process slice tensor, slice_num=" << slice_num << ", slice_id=" << slice_id;

  // slice_num is 0, that means the tensor is not sliced
  if (slice_num == 0) {
    if (sliced_tensor_ != nullptr) {
      MS_LOG(WARNING) << "Expect slice id " << sliced_tensor_->slice_id_ << ", but got slice id " << slice_id;
      return false;
    }
    return ProcessFullTensor(item);
  }

  // current data item is just a slice of tensor
  if (sliced_tensor_ == nullptr) {
    size_t dim_num = CALL_ASCEND_API(acltdtGetDimNumFromItem, item);
    aclDataType acl_data_type = CALL_ASCEND_API(acltdtGetDataTypeFromItem, item);

    ShapeVector tensor_shape;
    tensor_shape.resize(dim_num);

    if (CALL_ASCEND_API(acltdtGetDimsFromItem, item, tensor_shape.data(), dim_num) != ACL_SUCCESS) {
      MS_LOG(WARNING) << "ACL failed to get dim-size from acl channel data";
      return false;
    }
    sliced_tensor_ = std::make_shared<SlicedTensor>(slice_num, acl_data_type, tensor_shape);
  }
  if (slice_id != sliced_tensor_->slice_id_) {
    MS_LOG(WARNING) << "Expect slice id " << sliced_tensor_->slice_id_ << ", but got slice id " << slice_id;
    return false;
  }

  void *acl_addr = CALL_ASCEND_API(acltdtGetDataAddrFromItem, item);
  size_t acl_data_size = CALL_ASCEND_API(acltdtGetDataSizeFromItem, item);
  if (acl_data_size > 0) {
    sliced_tensor_->buffer_ << std::string(static_cast<char *>(acl_addr), acl_data_size);
  }
  sliced_tensor_->slice_id_ += 1;
  // when received last piece of tensor
  if (sliced_tensor_->slice_id_ == sliced_tensor_->slice_num_) {
    return FinishSliceTensor();
  }
  return true;
}

bool ScopeAclTdtDataset::FinishSliceTensor() {
  aclDataType acl_data_type = sliced_tensor_->data_type_;
  std::string tensor_data = sliced_tensor_->buffer_.str();
  if (acl_data_type == ACL_STRING) {
    data_items_.emplace_back(tensor_data);
  } else {
    auto type_iter = kAclDataTypeMap.find(acl_data_type);
    if (type_iter == kAclDataTypeMap.end()) {
      MS_LOG(WARNING) << "The type of aclData not support: " << acl_data_type;
      return false;
    }
    auto type_id = type_iter->second;
    auto tensor_ptr = tensor::from_spec(type_id, sliced_tensor_->tensor_shape_, device::DeviceType::kCPU);
    if (!CopyDataToTensor(reinterpret_cast<const uint8_t *>(tensor_data.c_str()), tensor_ptr, tensor_data.size())) {
      return false;
    }
    data_items_.emplace_back(tensor_ptr);
  }
  sliced_tensor_ = nullptr;
  return true;
}

bool ScopeAclTdtDataset::ProcessDataset(acltdtDataset *acl_dataset) {
  bool is_end_output = (tensor_type_ != ACL_TENSOR_DATA_SLICE_TENSOR);
  bool error_flag = false;

  // NOTE: ONLY the FIRST dataset containing the dataset name
  // May be the acltdtDataset is empty but has name
  if (tensor_type_ == ACL_TENSOR_DATA_UNDEFINED && dataset_name_.empty()) {
    dataset_name_ = CALL_ASCEND_API(acltdtGetDatasetName, acl_dataset);
  }

  size_t acl_dataset_size = CALL_ASCEND_API(acltdtGetDatasetSize, acl_dataset);
  MS_LOG(DEBUG) << "Receive one dataset with size " << acl_dataset_size << ", tensor_type_=" << tensor_type_
                << ", sliced_tensor_=" << sliced_tensor_.get();

  for (size_t i = 0; i < acl_dataset_size; i++) {
    acltdtDataItem *item = CALL_ASCEND_API(acltdtGetDataItem, acl_dataset, i);
    MS_EXCEPTION_IF_NULL(item);

    auto type = CALL_ASCEND_API(acltdtGetTensorTypeFromItem, item);
    MS_LOG(DEBUG) << "Process data item " << i << "/" << acl_dataset_size << ", type is " << type << ", data type is "
                  << CALL_ASCEND_API(acltdtGetDataTypeFromItem, item) << ", data length is "
                  << CALL_ASCEND_API(acltdtGetDataSizeFromItem, item);

    if (type == ACL_TENSOR_DATA_END_OF_SEQUENCE) {
      MS_LOG(INFO) << "Encounter end of sequence";
      break;
    }
    if (type != ACL_TENSOR_DATA_TENSOR && type != ACL_TENSOR_DATA_SLICE_TENSOR && type != ACL_TENSOR_DATA_END_TENSOR) {
      MS_LOG(WARNING) << "Encounter invalid data item of type " << type << ", ignore it.";
      continue;
    }
    if (!CheckAndSetTensorType(type)) {
      error_flag = true;
      break;
    }

    if (type == ACL_TENSOR_DATA_TENSOR) {
      if (!ProcessFullTensor(item)) {
        error_flag = true;
        break;
      }
      continue;
    }

    // dataitem is a slice tensor, i.e. type of which is ACL_TENSOR_DATA_SLICE_TENSOR or ACL_TENSOR_DATA_END_TENSOR
    is_end_output = false;
    if (!ProcessSliceTensor(item)) {
      error_flag = true;
      break;
    }

    if (type == ACL_TENSOR_DATA_END_TENSOR) {
      // reach the end of current output
      is_end_output = true;
      break;
    }
  }

  if (error_flag) {
    // when encounter error, drop out all processed data
    is_end_output = false;
    Reset();
  }

  MS_LOG(DEBUG) << "Return with is_end_output=" << std::boolalpha << is_end_output;

  return is_end_output;
}

bool ScopeAclTdtDataset::CheckAndSetTensorType(acltdtTensorType tensor_type) {
  switch (tensor_type) {
    case ACL_TENSOR_DATA_TENSOR: {
      if (tensor_type_ == ACL_TENSOR_DATA_UNDEFINED) {
        tensor_type_ = ACL_TENSOR_DATA_TENSOR;
      } else if (tensor_type_ != ACL_TENSOR_DATA_TENSOR) {
        MS_LOG(WARNING) << "Encounter mismatched tensor type, expect " << tensor_type_ << " but got " << tensor_type;
        return false;
      }
      break;
    }
    case ACL_TENSOR_DATA_SLICE_TENSOR:
    case ACL_TENSOR_DATA_END_TENSOR: {
      if (tensor_type_ == ACL_TENSOR_DATA_UNDEFINED) {
        tensor_type_ = ACL_TENSOR_DATA_SLICE_TENSOR;
      } else if (tensor_type_ != ACL_TENSOR_DATA_SLICE_TENSOR) {
        MS_LOG(WARNING) << "Encounter mismatched tensor type, expect " << tensor_type_ << " but got " << tensor_type;
        return false;
      }
      break;
    }
    default:
      MS_LOG(WARNING) << "Encounter invalid tensor type " << tensor_type << ", ignore it.";
      return false;
  }
  return true;
}

MbufDataHandler::MbufDataHandler(MbufFuncType func, uint32_t device_id, string channel_name, string prim_name,
                                 size_t capacity, int32_t timeout)
    : func_(func), device_id_(device_id), channel_name_(channel_name), prim_name_(prim_name), timeout_(timeout) {
  constexpr size_t kMinChannelSize = 2;
  // NOTE: when call acltdtCreateChannelWithCapacity with input parameter capacity's value equal to 2, the real size of
  // tdt-channel is 1, only in this case, the second stage wait of OutfeedEnqueueOpV2 can be triggered
  capacity_ = common::GetDumpSliceSize() > 0 ? kMinChannelSize : capacity;
  MS_LOG(INFO) << "Channel " << channel_name_ << " begins the construction process witch capacity " << capacity_;
  MS_VLOG(VL_PRINT_DUMP_V0) << "Channel " << channel_name_ << " begins the construction process witch capacity "
                            << capacity_;
  acl_handle_ = CALL_ASCEND_API(acltdtCreateChannelWithCapacity, device_id_, channel_name_.c_str(), capacity_);
  if (acl_handle_ == nullptr) {
    string warning_info = "The creation of " + channel_name_ + " channel failed";
    if (!prim_name_.empty()) {
      warning_info += " and the corresponding " + prim_name_ + " Primitive cannot be used in GRAPH_MODE";
    }
    MS_LOG(WARNING) << warning_info;
    return;
  }
  thread_receive_ = std::thread(&MbufDataHandler::ReceiveData, this);
  thread_process_ = std::thread(&MbufDataHandler::ProcessData, this);
}

MbufDataHandler::~MbufDataHandler() {
  MS_LOG(INFO) << "Channel " << channel_name_ << " begins the destruction process.";
  // Stop the child thread from receiving data
  stop_receive_.store(true, std::memory_order_release);
  if (thread_receive_.joinable()) {
    thread_receive_.join();
  }
  if (thread_process_.joinable()) {
    thread_process_.join();
  }
  if (acl_handle_) {
    aclError status = CALL_ASCEND_API(acltdtDestroyChannel, acl_handle_);
    acl_handle_ = nullptr;
    if (status != ACL_SUCCESS) {
      MS_LOG(ERROR) << "Channel " << channel_name_ << " failed destroy acl channel. Error code: " << status;
      return;
    }
  } else {
    MS_LOG(INFO) << "Channel " << channel_name_ << ": acl handle has been destroyed.";
  }
}

bool MbufDataHandler::QueryChannelSize(size_t *size) {
  aclError status = CALL_ASCEND_API(acltdtQueryChannelSize, acl_handle_, size);
  if (status != ACL_SUCCESS) {
    MS_LOG(ERROR) << "Channel " << channel_name_ << " failed to QueryChannelSize. Error code is " << status;
    return false;
  }
  return true;
}

void MbufDataHandler::CleanChannel() {
  if (acl_handle_ == nullptr) {
    return;
  }

  aclError status = CALL_ASCEND_API(acltdtCleanChannel, acl_handle_);
  if (status != ACL_SUCCESS) {
    MS_LOG(ERROR) << "Clean channel " << channel_name_ << " failed with error code: " << status;
  }
  MS_LOG(INFO) << "Clean channel " << channel_name_ << " successfully.";
}

AclDatasetInfoPtr MbufDataHandler::CreateAclDatasetInfo(bool first_slice) {
  auto acl_dataset = CALL_ASCEND_API(acltdtCreateDataset);
  if (acl_dataset == nullptr) {
    MS_LOG(ERROR) << "Channel " << channel_name_ << " failed to create acltdtDateaset.";
    return nullptr;
  }
  return std::make_shared<AclDatasetInfo>(acl_dataset, first_slice);
}

acltdtDataItem *GetLastDataItem(const AclDatasetInfoPtr &ds_info) {
  if (ds_info == nullptr || ds_info->Get() == nullptr) {
    return nullptr;
  }

  size_t acl_dataset_size = CALL_ASCEND_API(acltdtGetDatasetSize, ds_info->Get());
  if (acl_dataset_size < 1) {
    MS_VLOG(VL_PRINT_DUMP_V1) << "Size of acltdtDataset is " << acl_dataset_size << ".";
    return nullptr;
  }

  // get type of the last data item in dataset
  return CALL_ASCEND_API(acltdtGetDataItem, ds_info->Get(), acl_dataset_size - 1);
}

RecvDataState MbufDataHandler::UpdateReceiveState(const AclDatasetInfoPtr &ds_info) {
  auto last_state = receive_state_;

  acltdtDataItem *item = GetLastDataItem(ds_info);
  if (item == nullptr) {
    return last_state;
  }

  auto type = CALL_ASCEND_API(acltdtGetTensorTypeFromItem, item);
  switch (type) {
    case ACL_TENSOR_DATA_TENSOR:
    case ACL_TENSOR_DATA_END_TENSOR: {
      receive_state_ = common::GetDumpSliceSize() > 0 ? RecvDataState::kWaitForSync : RecvDataState::kReadyToRecv;
      break;
    }
    case ACL_TENSOR_DATA_SLICE_TENSOR: {
      receive_state_ = RecvDataState::kWaitForEnd;
      break;
    }
    case ACL_TENSOR_DATA_END_OF_SEQUENCE: {
      receive_state_ = RecvDataState::kReadyToRecv;
      break;
    }
    default:
      break;
  }
  if (receive_state_ == last_state) {
    MS_VLOG(VL_PRINT_DUMP_V0) << "Channel " << channel_name_ << " dataset ends with data item of type " << type
                              << ", receive state keep unchanged " << GetRecvStateString(last_state) << ".";
  } else {
    MS_VLOG(VL_PRINT_DUMP_V0) << "Channel " << channel_name_ << " dataset ends with data item of type " << type
                              << ", receive state changed from " << GetRecvStateString(last_state) << " to "
                              << GetRecvStateString(receive_state_) << ".";
  }

  return last_state;
}

bool MbufDataHandler::ReceiveOneDataset(AclDatasetInfoPtr *ds_info_ptr) {
  // NOTE: ReceiveOneDataset is a internal method, the caller make sure ds_info_ptr is not nullptr
  AclDatasetInfoPtr &ds_info = *ds_info_ptr;
  if (ds_info == nullptr) {
    bool is_first_slice = (receive_state_ == RecvDataState::kReadyToRecv);
    if ((ds_info = CreateAclDatasetInfo(is_first_slice)) == nullptr) {
      return false;
    }
  }

  MS_VLOG(VL_PRINT_DUMP_V1) << "Channel " << channel_name_
                            << " call acltdtReceiveTensor begin to receive dataset, current receive state is "
                            << GetRecvStateString(receive_state_);
  aclError status = CALL_ASCEND_API(acltdtReceiveTensor, acl_handle_, ds_info->Get(), timeout_);
  MS_VLOG(VL_PRINT_DUMP_V1) << "Channel " << channel_name_ << " call acltdtReceiveTensor end with return code "
                            << status;
  if (status != ACL_SUCCESS) {
    if (status == ACL_ERROR_RT_QUEUE_EMPTY) {
      return true;
    }
    MS_LOG(ERROR) << "Channel " << channel_name_ << " failed to receive tensor. Error code is " << status;
    return false;
  }

  // update receive state
  (void)UpdateReceiveState(ds_info);
  acltdtDataItem *item = GetLastDataItem(ds_info);
  // drop sync data item
  if (item != nullptr && CALL_ASCEND_API(acltdtGetTensorTypeFromItem, item) == ACL_TENSOR_DATA_END_OF_SEQUENCE) {
    return true;
  }

  std::unique_lock<std::mutex> lock(queue_mutex_);
  dataset_queue_.emplace(ds_info);
  queue_cond_var_.notify_one();
  if (receive_state_ == RecvDataState::kWaitForSync) {
    auto wait_time = common::GetDumpWaitTime();
    auto op_debug_conf = OpDebugConf::GetInstance();
    MS_EXCEPTION_IF_NULL(op_debug_conf);
    uint32_t op_timeout = op_debug_conf->execute_timeout();
    // NOTE: Now CANN not support wait_time > op_timeout, change this condition when cann changed this limitation.
    if (wait_time == 0 || wait_time > static_cast<int>(op_timeout)) {
      wait_time = static_cast<int>(op_timeout);
      MS_VLOG(VL_PRINT_DUMP_V0) << "Set wait time to value of mindspore context variable `op_timeout`: " << wait_time;
    }
    if (wait_time > 1) {
      // the `wait_time` the second stage wait time of underlying aicpu operator, the host wait time must be less than
      // device
      wait_time -= 1;
    }
    MS_VLOG(VL_PRINT_DUMP_V0) << "Channel " << channel_name_ << " wait data sync to output dev, wait time is "
                              << wait_time;
    bool is_timeout = (queue_cond_var_.wait_for(lock, std::chrono::seconds(wait_time)) == std::cv_status::timeout);
    MS_VLOG(VL_PRINT_DUMP_V0) << "Channel " << channel_name_ << " wait data sync to output dev "
                              << (is_timeout ? "timeout" : "done") << ".";
    if (is_timeout) {
      MS_LOG(WARNING) << "Channel " << channel_name_
                      << " wait data sync to output dev timeout, current size of dataset queue is "
                      << dataset_queue_.size() << ", all items in dataset queue will be dropped.";
      while (!dataset_queue_.empty()) {
        dataset_queue_.pop();
      }
    }
  }
  // unlock ahead
  lock.unlock();
  ds_info = nullptr;
  MS_VLOG(VL_PRINT_DUMP_V0) << "Channel " << channel_name_ << " enqueue received dataset successfully.";

  return true;
}

void MbufDataHandler::ReceiveData() {
  MS_VLOG(VL_PRINT_DUMP_V0) << "Channel " << channel_name_ << " starts receiving data.";

  AclDatasetInfoPtr ds_info = nullptr;
  bool error_flag = false;
  while (!stop_receive_.load()) {
    if (!ReceiveOneDataset(&ds_info)) {
      error_flag = true;
      break;
    }
  }

  while (!error_flag) {
    size_t channel_size = 0;
    if (!QueryChannelSize(&channel_size)) {
      error_flag = true;
      break;
    }
    if (channel_size == 0) {
      break;
    }
    if (!ReceiveOneDataset(&ds_info)) {
      error_flag = true;
      break;
    }
  }

  done_receive_data_ = true;
  // the data processing thread may be blocked on queue_cond_var_, so notify it
  queue_cond_var_.notify_one();
  MS_VLOG(VL_PRINT_DUMP_V0) << "Channel " << channel_name_ << " finished receiving data "
                            << (error_flag ? "with" : "without") << " error occurred.";
}

void MbufDataHandler::ProcessData() {
  MS_VLOG(VL_PRINT_DUMP_V0) << "Channel " << channel_name_ << " starts processing data.";
  auto scoped_ds = std::make_unique<ScopeAclTdtDataset>();
  while (true) {
    std::unique_lock<std::mutex> lock(queue_mutex_);
    queue_cond_var_.wait(lock, [this] { return !dataset_queue_.empty() || done_receive_data_; });
    if (dataset_queue_.empty() && done_receive_data_) {
      break;
    }
    auto ds_info = dataset_queue_.front();
    dataset_queue_.pop();
    // unlock manually ahead, let receive thread continue to receive data
    lock.unlock();

    MS_VLOG(VL_PRINT_DUMP_V0) << "Channel " << channel_name_ << " process data start";
    if (ds_info->IsFirstSlice()) {
      scoped_ds->Reset(true);
    }
    auto output_finished = scoped_ds->ProcessDataset(ds_info->Get());
    MS_VLOG(VL_PRINT_DUMP_V0) << "Channel " << channel_name_ << " process data end with current output finished "
                              << std::boolalpha << output_finished;
    if (output_finished) {
      MS_VLOG(VL_PRINT_DUMP_V0) << "Channel " << channel_name_ << " write data to disk or stdout start";
      func_(scoped_ds->GetDatasetName(), scoped_ds->GetDataItems());
      MS_VLOG(VL_PRINT_DUMP_V0) << "Channel " << channel_name_ << " write data to disk or stdout end";
      scoped_ds->Reset();
    }
    if (dataset_queue_.empty() && common::GetDumpSliceSize() > 0) {
      // after writing data to output dev finished, notify the data receive thread
      queue_cond_var_.notify_one();
    }
  }
  MS_VLOG(VL_PRINT_DUMP_V0) << "Channel " << channel_name_ << " finished processing data.";
}
}  // namespace mindspore::device::ascend
