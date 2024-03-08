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
#include "plugin/device/ascend/hal/device/mbuf_receive_manager.h"
#include <ctime>
#include <fstream>
#include <memory>
#include <string>
#include <thread>
#include "include/common/utils/utils.h"
#include "ir/tensor.h"
#include "plugin/device/ascend/hal/device/ascend_data_queue.h"
#include "utils/file_utils.h"
#include "utils/ms_context.h"
#include "utils/shape_utils.h"
#include "transform/symbol/acl_tdt_symbol.h"
#include "transform/symbol/symbol_utils.h"

namespace mindspore::device::ascend {

namespace {
// Data may not be received when the process exits; reserve a timeout period.
constexpr std::chrono::milliseconds stop_time_out{100};

bool CopyDataToTensor(const uint8_t *src_addr, mindspore::tensor::TensorPtr tensor_ptr, const size_t size) {
  MS_EXCEPTION_IF_NULL(src_addr);
  MS_EXCEPTION_IF_NULL(tensor_ptr);
  auto *dst_addr = reinterpret_cast<uint8_t *>(tensor_ptr->data_c());
  MS_EXCEPTION_IF_NULL(dst_addr);
  size_t dst_size = static_cast<size_t>(tensor_ptr->data().nbytes());
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
}  // namespace

mindspore::tensor::TensorPtr acltdtDataItemToTensorPtr(acltdtDataItem *item) {
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
  auto tensor_ptr = std::make_shared<mindspore::tensor::Tensor>(type_id, tensor_shape);
  if (acl_data_size == 0) {
    return tensor_ptr;
  }
  if (CopyDataToTensor(acl_data, tensor_ptr, acl_data_size)) {
    return tensor_ptr;
  }
  return nullptr;
}

MbufDataHandler::MbufDataHandler(MbufFuncType func, uint32_t device_id, string channel_name, string prim_name,
                                 size_t capacity, int32_t timeout)
    : func_(func),
      device_id_(device_id),
      channel_name_(channel_name),
      prim_name_(prim_name),
      capacity_(capacity),
      timeout_(timeout) {
  MS_LOG(INFO) << "Channel " << channel_name_ << " begins the construction process.";
  acl_handle_ = CALL_ASCEND_API(acltdtCreateChannelWithCapacity, device_id_, channel_name_.c_str(), capacity_);
  if (acl_handle_ == nullptr) {
    string warning_info = "The creation of " + channel_name_ + " channel failed";
    if (!prim_name_.empty()) {
      warning_info += " and the corresponding " + prim_name_ + " Primitive cannot be used in GRAPH_MODE";
    }
    MS_LOG(WARNING) << warning_info;
    return;
  }
  thread_ = std::thread(&MbufDataHandler::HandleData, this);
}

MbufDataHandler::~MbufDataHandler() {
  MS_LOG(INFO) << "Channel " << channel_name_ << " begins the destruction process.";
  // Stop the child thread from receiving data
  stop_receive_.store(true, std::memory_order_acq_rel);
  if (thread_.joinable()) {
    thread_.join();
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

bool MbufDataHandler::ReceiveAndProcessData(const ScopeAclTdtDataset &scope_acl_dataset) {
  aclError status = CALL_ASCEND_API(acltdtReceiveTensor, acl_handle_, scope_acl_dataset.Get(), timeout_);
  if (status != ACL_ERROR_RT_QUEUE_EMPTY && status != ACL_SUCCESS) {
    MS_LOG(ERROR) << "Channel " << channel_name_ << " failed to receive tensor. Error code is " << status;
    return false;
  }

  if (status == ACL_SUCCESS) {
    func_(scope_acl_dataset.Get());
  }
  return true;
}

bool MbufDataHandler::QueryChannelSize(size_t *size) {
  aclError status = CALL_ASCEND_API(acltdtQueryChannelSize, acl_handle_, size);
  if (status != ACL_SUCCESS) {
    MS_LOG(ERROR) << "Channel " << channel_name_ << " failed to QueryChannelSize. Error code is " << status;
    return false;
  }
  return true;
}

void MbufDataHandler::HandleData() {
  MS_LOG(INFO) << "Channel " << channel_name_ << " starts executing HandleData.";
  ScopeAclTdtDataset scope_acl_dataset;
  if (scope_acl_dataset.Get() == nullptr) {
    MS_LOG(ERROR) << "Channel " << channel_name_ << " failed to create aclDateaset.";
    return;
  }

  while (!stop_receive_.load()) {
    if (!ReceiveAndProcessData(scope_acl_dataset)) {
      return;
    }
  }

  size_t channel_size = 0;
  auto start = std::chrono::high_resolution_clock::now();
  if (!QueryChannelSize(&channel_size)) {
    return;
  }
  while (channel_size > 0 && std::chrono::high_resolution_clock::now() - start < stop_time_out) {
    if (!ReceiveAndProcessData(scope_acl_dataset)) {
      return;
    }
    if (!QueryChannelSize(&channel_size)) {
      return;
    }
  }
  if (channel_size > 0) {
    MS_LOG(ERROR) << "Channel " << channel_name_ << " has stopped receiving data, and " << channel_size
                  << " pieces of data have not been received.";
    return;
  }
}

}  // namespace mindspore::device::ascend
