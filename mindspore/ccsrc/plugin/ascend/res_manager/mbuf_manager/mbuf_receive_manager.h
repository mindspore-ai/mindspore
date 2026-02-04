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

#ifndef MINDSPORE_CCSRC_PLUGIN_RES_MANAGER_ASCEND_MBUF_MANAGER_MBUF_RECEIVE_MANAGER_H_
#define MINDSPORE_CCSRC_PLUGIN_RES_MANAGER_ASCEND_MBUF_MANAGER_MBUF_RECEIVE_MANAGER_H_

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <future>
#include <map>
#include <queue>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <utility>
#include <vector>
#include <variant>
#include "ir/tensor.h"
#include "plugin/ascend/res_manager/symbol_interface/acl_tdt_symbol.h"
#include "plugin/ascend/res_manager/symbol_interface/symbol_utils.h"
#include "include/utils/utils.h"
#include "plugin/ascend/res_manager/visible.h"

#ifndef SECUREC_MEM_MAX_LEN
#define SECUREC_MEM_MAX_LEN 0x7fffffffUL
#endif

namespace mindspore::device::ascend {

class ScopeAclTdtDataset;
using DataItem = std::variant<std::string, mindspore::tensor::TensorPtr>;

using MbufFuncType = std::function<void(const std::string &, const std::vector<DataItem> &)>;

const std::map<aclDataType, TypeId> kAclDataTypeMap = {{ACL_INT8, TypeId::kNumberTypeInt8},
                                                       {ACL_UINT8, TypeId::kNumberTypeUInt8},
                                                       {ACL_INT16, TypeId::kNumberTypeInt16},
                                                       {ACL_UINT16, TypeId::kNumberTypeUInt16},
                                                       {ACL_INT32, TypeId::kNumberTypeInt32},
                                                       {ACL_UINT32, TypeId::kNumberTypeUInt32},
                                                       {ACL_INT64, TypeId::kNumberTypeInt64},
                                                       {ACL_UINT64, TypeId::kNumberTypeUInt64},
#ifdef EXPERIMENT_A5
                                                       {ACL_HIFLOAT8, TypeId::kNumberTypeHiFloat8},
                                                       {ACL_FLOAT8_E5M2, TypeId::kNumberTypeFloat8E5M2},
                                                       {ACL_FLOAT8_E4M3FN, TypeId::kNumberTypeFloat8E4M3FN},
#endif
                                                       {ACL_FLOAT16, TypeId::kNumberTypeFloat16},
                                                       {ACL_BF16, TypeId::kNumberTypeBFloat16},
                                                       {ACL_FLOAT, TypeId::kNumberTypeFloat32},
                                                       {ACL_DOUBLE, TypeId::kNumberTypeFloat64},
                                                       {ACL_COMPLEX64, TypeId::kNumberTypeComplex64},
                                                       {ACL_COMPLEX128, TypeId::kNumberTypeComplex128},
                                                       {ACL_BOOL, TypeId::kNumberTypeBool}};

struct SlicedTensor {
  SlicedTensor(size_t slice_num, aclDataType type, const ShapeVector &shape)
      : slice_id_(0), slice_num_(slice_num), data_type_(type), tensor_shape_(shape) {}
  SlicedTensor(const SlicedTensor &) = delete;
  SlicedTensor &operator=(const SlicedTensor &) = delete;
  ~SlicedTensor() = default;

  // the id of current slice of tensor
  size_t slice_id_{0};
  // the number of total slices of tensor
  size_t slice_num_{0};
  // tensor's data type and shape
  aclDataType data_type_;
  ShapeVector tensor_shape_;
  // buffer for storing contents of sliced tensor
  std::ostringstream buffer_;
};

class ScopeAclTdtDataset {
 public:
  ScopeAclTdtDataset() { Reset(); }
  ~ScopeAclTdtDataset() {}

  void Reset(bool force = false) {
    if (force && tensor_type_ != ACL_TENSOR_DATA_UNDEFINED) {
      MS_LOG(WARNING) << "Clear the incomplete data been processed.";
    }
    sliced_tensor_ = nullptr;
    sliced_string_ = nullptr;
    dataset_name_ = "";
    tensor_type_ = ACL_TENSOR_DATA_UNDEFINED;
    data_items_.clear();
  }

  const std::vector<DataItem> &GetDataItems() const { return data_items_; }

  const std::string &GetDatasetName() const { return dataset_name_; }

  // process full tensor(i.e. the content of tensor is in only one acltdtDataItem)
  // return true when success, otherwise false
  bool ProcessFullTensor(acltdtDataItem *item);

  // process sliced tensor(i.e. the content of tensor spans multiple acltdtDataItems)
  // return true when success, otherwise false
  bool ProcessSliceTensor(acltdtDataItem *item);

  // call this function when received last piece of slice tensor, return true when success, otherwise false
  bool FinishSliceTensor();

  // return true when encounter the end of OutfeedEnqueueOpV2's output, otherwise false
  bool ProcessDataset(acltdtDataset *acl_dataset);

  // set and check consistency of tensor types of data items, return true when success, otherwise false
  bool CheckAndSetTensorType(acltdtTensorType tensor_type);

 private:
  // acl tdt dataset for receiving data, created once, used many times
  acltdtDataset *acl_dataset_{nullptr};

  // structure for connecting tensor slices to a full tensor
  std::shared_ptr<SlicedTensor> sliced_tensor_{nullptr};
  // structure for connecting string slices to a full string
  std::shared_ptr<std::ostringstream> sliced_string_{nullptr};

  // ONLY the FIRST dataset containing the dataset name when the outputs of OutfeedEnqueueOpV2 span multiple datasets
  std::string dataset_name_;
  // NOTE: the data items of output of one OutfeedEnqueueOpV2 must be all with type ACL_TENSOR_DATA_TENSOR, or all with
  // type ACL_TENSOR_DATA_SLICE_TENSOR(ACL_TENSOR_DATA_END_TENSOR is also indicating type ACL_TENSOR_DATA_SLICE_TENSOR)
  acltdtTensorType tensor_type_{ACL_TENSOR_DATA_UNDEFINED};
  // vector for buffering outputs of OutfeedEnqueueOpV2 at a time
  std::vector<DataItem> data_items_;
};

class AclDatasetInfo {
 public:
  AclDatasetInfo(acltdtDataset *dataset, bool first_slice) : acl_dataset_(dataset), first_slice_(first_slice) {}
  ~AclDatasetInfo() {
    if (acl_dataset_ == nullptr) {
      return;
    }
    if (CALL_ASCEND_API(acltdtDestroyDataset, acl_dataset_) != ACL_SUCCESS) {
      MS_LOG(ERROR) << "AcltdtDestroyDataset failed.";
    } else {
      MS_LOG(INFO) << "AcltdtDestroyDataset succeeded.";
    }
  }

  acltdtDataset *Get() { return acl_dataset_; }
  bool IsFirstSlice() { return first_slice_; }

 private:
  acltdtDataset *acl_dataset_ = nullptr;
  bool first_slice_ = false;
};
using AclDatasetInfoPtr = std::shared_ptr<AclDatasetInfo>;

enum class RecvDataState : uint8_t {
  kReadyToRecv,  // ready to receive a new tensor
  kWaitForEnd,   // wait for the end of sliced tensor
  kWaitForSync   // wait for a synchronization dataset
};

class ASCEND_RES_MANAGER_EXPORT MbufDataHandler {
 public:
  MbufDataHandler(MbufFuncType func, uint32_t device_id, string channel_name, string op_name = "",
                  size_t capacity = 128, int32_t timeout = 800);
  ~MbufDataHandler();
  string GetChannelName() { return channel_name_; }
  uint32_t GetDeviceId() { return device_id_; }
  size_t GetCapacity() { return capacity_; }
  void StopReceive() { stop_receive_.store(true, std::memory_order_release); }
  void CleanChannel();

 private:
  MbufFuncType func_;
  uint32_t device_id_;
  std::string channel_name_;
  std::string prim_name_;
  size_t capacity_;
  int32_t timeout_;
  std::atomic_bool stop_receive_{false};
  acltdtChannelHandle *acl_handle_;

  // record the current receive state for data receiving thread
  RecvDataState receive_state_ = RecvDataState::kReadyToRecv;
  // whether the data receive thread has finished receiving all data
  bool done_receive_data_ = false;
  // thread for receiving data from device to host and put it on data queue
  std::thread thread_receive_;
  // thread for processing data on data queue and write data to file or stdout
  std::thread thread_process_;
  // mutex and cond_var for thread synchronization between thread_receive and thread_process
  std::mutex queue_mutex_;
  std::condition_variable queue_cond_var_;
  // data queue for sharing data between thread_receive and thread_process
  std::queue<AclDatasetInfoPtr> dataset_queue_;

  AclDatasetInfoPtr CreateAclDatasetInfo(bool first_slice);
  // receive one acltdtDataset, return false when encounter error, otherwise return true
  bool ReceiveOneDataset(AclDatasetInfoPtr *ds_info_ptr);
  // Update receive_state_ according to the input dataset and return the value of receive_state_ before update
  RecvDataState UpdateReceiveState(const AclDatasetInfoPtr &ds_info);
  // thread main function for receiving data
  void ReceiveData();
  // thread main function for processing data
  void ProcessData();

  bool QueryChannelSize(size_t *queue_size);
};

class ASCEND_RES_MANAGER_EXPORT MbufDataHandlerManager {
 public:
  static MbufDataHandlerManager &GetInstance() {
    static MbufDataHandlerManager instance;
    return instance;
  }
  ~MbufDataHandlerManager() = default;
  MbufDataHandlerManager(const MbufDataHandlerManager &) = delete;
  MbufDataHandlerManager &operator=(const MbufDataHandlerManager &) = delete;

  void AddHandler(std::unique_ptr<MbufDataHandler> handler) { handles_.push_back(std::move(handler)); }

  void CleanChannels() {
    for (auto &handle : handles_) {
      handle->CleanChannel();
    }
  }

  void DestoryPrintHandler() {
    for (auto iter = handles_.begin(); iter != handles_.end(); iter++) {
      if ((*iter)->GetChannelName() == kChannelNameNpuLog) {
        (*iter)->StopReceive();
        handles_.erase(iter);
        break;
      }
    }
  }

  void DestoryHandler() {
    for (auto &handle : handles_) {
      handle->StopReceive();
    }
    while (!handles_.empty()) {
      MS_LOG(INFO) << "The thread of " << handles_.back()->GetChannelName() << " channel is being destroyed.";
      handles_.pop_back();
    }
  }

 private:
  MbufDataHandlerManager() = default;
  std::vector<std::unique_ptr<MbufDataHandler>> handles_;
};
}  // namespace mindspore::device::ascend
#endif  // MINDSPORE_CCSRC_PLUGIN_RES_MANAGER_ASCEND_MBUF_MANAGER_MBUF_RECEIVE_MANAGER_H_
