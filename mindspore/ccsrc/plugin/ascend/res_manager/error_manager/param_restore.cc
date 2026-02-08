/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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

#include "plugin/ascend/res_manager/error_manager/param_restore.h"
#include <cstddef>
#include <algorithm>
#include <vector>
#include "plugin/ascend/res_manager/stream_manager/ascend_stream_manager.h"
#include "plugin/ascend/res_manager/collective/ascend_collective_comm_lib.h"
#include "plugin/ascend/res_manager/hccl_adapter/hccl_adapter.h"
#include "include/utils/utils.h"
#include "plugin/ascend/res_manager/symbol_interface/acl_rt_symbol.h"
#include "plugin/ascend/res_manager/symbol_interface/symbol_utils.h"
#include "include/runtime/hardware_abstract/collective/collective_manager.h"
#include "utils/log_adapter.h"

namespace mindspore {
namespace device {
namespace ascend {
// In order to do parameters replication, do the following steps:
// 1. Calculate the free memory.
// 2. Calculate number of total parameters and size in bytes of each parameter, whether the device address of the
//    parameter is allocated.
// 3. Exchange free device memory and info of parameters between send/receive size.
// 4. Calculate send/receive buffer size and allocate device memory to send/receive parameters.
// 5. Execute send/receive to copy parameters.
// The format for exchanging free device memory and info of parameters between send/receive size is as below:
// +-----------------------------------------------------------------------+
// + uint64_t   | uint64_t    | uint64_t   | uint64_t   | ... | uint64_t   |
// + freeMemory | numOfParams | sizeParam1 | sizeParam2 | ... | sizeParamN |
// +-----------------------------------------------------------------------+
// If parameter K's device address is not allocated, the `sizeParamK` is filled with `uint64_t::max`.

namespace {
constexpr size_t kDefaultStreamId = 0;
constexpr char kHcclWorldGroup[] = "hccl_world_group";
constexpr size_t kMegaByte = 1024 * 1024;
constexpr size_t kUceParamsMaxMemThresh = 512 * kMegaByte;       // 512MB
constexpr size_t kUceDeviceFreeMemTresh = 8 * 1024 * kMegaByte;  // 8GB

uint64_t AlignSize(uint64_t num) {
  constexpr uint64_t align_value = 64;
  return (num + align_value - 1) / align_value * align_value;
}

uint64_t GetExchangeBufferSize(const DataExchangeInfo &local_info, const DataExchangeInfo &remote_info) {
  uint64_t min_free_mem_size = std::min({local_info.GetFreeDevMem(), remote_info.GetFreeDevMem()});
  if (local_info.GetSizeMax() > min_free_mem_size) {
    MS_LOG(INFO) << "The free device memory size local(" << local_info.GetFreeDevMem() << ") and remote("
                 << remote_info.GetFreeDevMem() << ") are less than the largest aligned size of parameter "
                 << local_info.GetSizeMax();
    return DataExchangeInfo::kInvalidParamSize;
  }

  return std::min({min_free_mem_size, local_info.GetSizeSum()});
}

bool UceCanUseBatchCopy(size_t params_size_in_bytes, size_t device_free_mem_size) {
  return params_size_in_bytes < kUceParamsMaxMemThresh && device_free_mem_size > kUceDeviceFreeMemTresh;
}
}  // namespace

DataExchangeInfo::DataExchangeInfo(const std::vector<tensor::TensorPtr> &params, size_t device_free_size) {
  data_.resize(params.size() + kDataExchangeInfoHeadSize);
  data_[0] = device_free_size;
  data_[1] = params.size();

  for (size_t i = 0; i < params.size(); ++i) {
    auto &tensor = params[i];
    auto tensor_size = (tensor->device_address() == nullptr || tensor->device_address()->GetMutablePtr() == nullptr)
                         ? kInvalidParamSize
                         : static_cast<uint64_t>(tensor->Size());
    data_[kDataExchangeInfoHeadSize + i] = tensor_size;
    if (tensor_size == kInvalidParamSize) {
      continue;
    }
    auto aligned_value = AlignSize(tensor_size);
    size_sum_ += aligned_value;
    if (aligned_value > size_max_) {
      size_max_ = aligned_value;
    }
  }
}

void ParamReplication::Init() {
  stream_id_ = kDefaultStreamId;
  stream_ = AscendStreamMng::GetInstance().GetStream(stream_id_);
  MS_EXCEPTION_IF_NULL(stream_);
  MS_LOG(INFO) << "Use stream " << stream_id_ << " for parameter replication.";
  comm_ = AscendCollectiveCommLib::GetInstance().GetHcomByGroup(kHcclWorldGroup);
  MS_EXCEPTION_IF_NULL(comm_);
  rank_id_ = static_cast<int>(distributed::collective::CollectiveManager::instance()->GetRankId(kHcclWorldGroup));
}

struct ExchangeDevAddr {
  explicit ExchangeDevAddr(const AscendResManager *res_mgr) : res_mgr_(res_mgr) {}
  ~ExchangeDevAddr() {
    if (send_dev_addr != nullptr && res_mgr_ != nullptr) {
      res_mgr_->FreeMemory(send_dev_addr);
    }
    if (recv_dev_addr != nullptr && res_mgr_ != nullptr) {
      res_mgr_->FreeMemory(recv_dev_addr);
    }
  }
  void *send_dev_addr = nullptr;
  void *recv_dev_addr = nullptr;
  const AscendResManager *res_mgr_;
};

int ParamReplication::DoParamInfoExchange(DataExchangeInfo *local_info, DataExchangeInfo *remote_info, int src_rank,
                                          int dst_rank) {
  MS_EXCEPTION_IF_NULL(local_info);
  MS_EXCEPTION_IF_NULL(remote_info);
  size_t xchg_info_size = local_info->GetSize() * sizeof(int64_t);
  ExchangeDevAddr addr(res_mgr_);
  addr.send_dev_addr = res_mgr_->AllocateMemory(xchg_info_size, stream_id_);
  addr.recv_dev_addr = res_mgr_->AllocateMemory(xchg_info_size, stream_id_);
  if (addr.send_dev_addr == nullptr || addr.recv_dev_addr == nullptr) {
    MS_LOG(ERROR) << "Allocate device memory of size " << xchg_info_size << " failed.";
    return 1;
  }

  // copy local free device memory and memory info from host to device
  if (CALL_ASCEND_API(aclrtMemcpy, addr.send_dev_addr, xchg_info_size, local_info->GetData(), xchg_info_size,
                      ACL_MEMCPY_HOST_TO_DEVICE) != ACL_SUCCESS) {
    MS_LOG(ERROR) << "Copy exchange info from host to device fail.";
    return 1;
  }

  // exchange info parameter info
  if (rank_id_ == src_rank) {
    // send node
    // 1. send local device memory and param info to remote
    hccl::HcclAdapter::GetInstance().HcclSend(addr.send_dev_addr, xchg_info_size, HCCL_DATA_TYPE_INT8, dst_rank,
                                              stream_, comm_);
    // 2. receive device memory and param info from remote to local
    hccl::HcclAdapter::GetInstance().HcclRecv(addr.recv_dev_addr, xchg_info_size, HCCL_DATA_TYPE_INT8, dst_rank,
                                              stream_, comm_);
  } else {
    // receive node
    // 1. receive device memory and param info from remote to local
    hccl::HcclAdapter::GetInstance().HcclRecv(addr.recv_dev_addr, xchg_info_size, HCCL_DATA_TYPE_INT8, src_rank,
                                              stream_, comm_);
    // 2. send local device memory and param info to remote
    hccl::HcclAdapter::GetInstance().HcclSend(addr.send_dev_addr, xchg_info_size, HCCL_DATA_TYPE_INT8, src_rank,
                                              stream_, comm_);
  }
  (void)res_mgr_->SyncStream(stream_id_);

  // copy remote free device memory and memory info from device to host
  if (CALL_ASCEND_API(aclrtMemcpy, remote_info->GetData(), xchg_info_size, addr.recv_dev_addr, xchg_info_size,
                      ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) {
    MS_LOG(ERROR) << "Copy exchange info from device to host fail.";
    return 1;
  }

  if (!local_info->IsParamInfoSame(*remote_info)) {
    MS_LOG(ERROR) << "Sizes of parameters of local and remote are not same, can not do parameter replication.";
    for (size_t i = 0; i < local_info->GetSize(); ++i) {
      MS_LOG(INFO) << "rank " << rank_id_ << " [" << i << "]=" << local_info->GetData()[i] << "("
                   << remote_info->GetData()[i] << ")";
    }
    return 1;
  }

  return 0;
}

int ParamReplication::CopyParamsInBatches(const std::vector<tensor::TensorPtr> &params, int src_rank, int dst_rank,
                                          void *xchg_buf_addr, size_t xchg_buf_size) {
  auto copy_param = [&params, &xchg_buf_addr, &xchg_buf_size, this](size_t beg_idx, bool is_send) -> size_t {
    size_t index = beg_idx;
    uint64_t sum_size = 0;
    while (index < params.size()) {
      auto &tensor = params[index];
      if (tensor->device_address() == nullptr || tensor->device_address()->GetMutablePtr() == nullptr ||
          tensor->device_address()->GetDeviceType() == device::DeviceType::kCPU) {
        index += 1;
        continue;
      }
      auto aligned_size = AlignSize(tensor->Size());
      if (sum_size + aligned_size > xchg_buf_size) {
        break;
      }

      void *dst_addr =
        is_send ? reinterpret_cast<uint8_t *>(xchg_buf_addr) + sum_size : tensor->device_address()->GetMutablePtr();
      void *src_addr =
        is_send ? tensor->device_address()->GetMutablePtr() : reinterpret_cast<uint8_t *>(xchg_buf_addr) + sum_size;

      if (CALL_ASCEND_API(aclrtMemcpyAsync, dst_addr, tensor->Size(), src_addr, tensor->Size(),
                          ACL_MEMCPY_DEVICE_TO_DEVICE, stream_) != ACL_SUCCESS) {
        MS_LOG(EXCEPTION) << "Async copy data from device to device fail, dst:" << dst_addr
                          << " size:" << tensor->Size() << " src:" << src_addr << " size:" << tensor->Size();
      }
      sum_size += aligned_size;
      index += 1;
    }
    return index;
  };

  size_t index = 0;
  while (index < params.size()) {
    MS_LOG(INFO) << "Copy parameter begin, index = " << index << "/" << params.size()
                 << (rank_id_ == src_rank ? " send" : " recv");
    if (rank_id_ == src_rank) {
      index = copy_param(index, true);
      hccl::HcclAdapter::GetInstance().HcclSend(xchg_buf_addr, xchg_buf_size, HCCL_DATA_TYPE_INT8, dst_rank, stream_,
                                                comm_);
    } else {
      hccl::HcclAdapter::GetInstance().HcclRecv(xchg_buf_addr, xchg_buf_size, HCCL_DATA_TYPE_INT8, src_rank, stream_,
                                                comm_);

      index = copy_param(index, false);
    }
    (void)res_mgr_->SyncStream(stream_id_);
    MS_LOG(INFO) << "Copy parameter end, index = " << index << "/" << params.size()
                 << (rank_id_ == src_rank ? " send" : " recv");
  }

  return 0;
}

int ParamReplication::CopyParamsOneByOne(const std::vector<tensor::TensorPtr> &params, int src_rank, int dst_rank) {
  if (rank_id_ != src_rank && rank_id_ != dst_rank) {
    MS_LOG(EXCEPTION) << "Neither source rank(" << src_rank << ") nor destination rank(" << dst_rank
                      << ") matches current rank id(" << rank_id_ << "), ignore it.";
  }

  for (size_t index = 0; index < params.size(); ++index) {
    auto &tensor = params[index];
    if (tensor->device_address() == nullptr || tensor->device_address()->GetMutablePtr() == nullptr ||
        tensor->device_address()->GetDeviceType() == device::DeviceType::kCPU) {
      MS_LOG(INFO) << "Device address is nullptr, skip copying parameter index = " << index << "/" << params.size()
                   << " " << (rank_id_ == src_rank ? " send" : " recv");
      continue;
    }
    MS_LOG(INFO) << "Copy parameter begin, index = " << index << "/" << params.size() << " "
                 << (rank_id_ == src_rank ? " send" : " recv");
    if (rank_id_ == src_rank) {
      hccl::HcclAdapter::GetInstance().HcclSend(tensor->device_address()->GetMutablePtr(), tensor->Size(),
                                                HCCL_DATA_TYPE_INT8, dst_rank, stream_, comm_);
    } else {
      hccl::HcclAdapter::GetInstance().HcclRecv(tensor->device_address()->GetMutablePtr(), tensor->Size(),
                                                HCCL_DATA_TYPE_INT8, src_rank, stream_, comm_);
    }
    MS_LOG(INFO) << "Copy parameter end, index = " << index << "/" << params.size() << " "
                 << (rank_id_ == src_rank ? " send" : " recv");
  }
  (void)res_mgr_->SyncStream(stream_id_);

  return 0;
}

// return 0 when success, otherwise return 1
int ParamReplication::SendRecv(const std::vector<tensor::TensorPtr> &params, int src_rank, int dst_rank,
                               bool use_batch) {
  MS_LOG(INFO) << "Copy parameters start.";

  if (rank_id_ != src_rank && rank_id_ != dst_rank) {
    MS_LOG(EXCEPTION) << "Neither source rank(" << src_rank << ") nor destination rank(" << dst_rank
                      << ") matches current rank id(" << rank_id_ << "), ignore it.";
  }

  if (!res_mgr_->BindDeviceToCurrentThread(false)) {
    MS_LOG(ERROR) << "Bind context to current thread failed";
    return 1;
  }

  size_t device_hbm_free_size;
  size_t device_hbm_total_size;
  auto ret = CALL_ASCEND_API(aclrtGetMemInfo, ACL_HBM_MEM, &device_hbm_free_size, &device_hbm_total_size);
  if (ret != ACL_SUCCESS || device_hbm_total_size == 0) {
    MS_LOG(EXCEPTION) << "Internal Error: Get Device MOC memory size failed, ret = " << ret
                      << ", total MOC size :" << device_hbm_total_size;
  }
  MS_LOG(INFO) << "device_moc_free_size=" << device_hbm_free_size / kMegaByte
               << "MB, device_moc_total_size=" << device_hbm_total_size / kMegaByte << "MB";

  DataExchangeInfo local_info(params, device_hbm_free_size);
  DataExchangeInfo remote_info(params.size());
  if (DoParamInfoExchange(&local_info, &remote_info, src_rank, dst_rank) != 0) {
    return 1;
  }

  size_t xchg_buf_size = GetExchangeBufferSize(local_info, remote_info);
  void *xchg_buf_addr = nullptr;
  if (xchg_buf_size != DataExchangeInfo::kInvalidParamSize &&
      UceCanUseBatchCopy(xchg_buf_size, local_info.GetFreeDevMem()) && use_batch) {
    xchg_buf_addr = res_mgr_->AllocateMemory(xchg_buf_size, stream_id_);
  }

  int ret_value = 0;
  if (xchg_buf_addr == nullptr) {
    MS_LOG(INFO) << "Copy parameters one by one.";
    ret_value = CopyParamsOneByOne(params, src_rank, dst_rank);
  } else {
    MS_LOG(INFO) << "Copy parameters on batches.";
    ret_value = CopyParamsInBatches(params, src_rank, dst_rank, xchg_buf_addr, xchg_buf_size);
    res_mgr_->FreeMemory(xchg_buf_addr);
  }

  MS_LOG(INFO) << "Copy parameters finish.";
  return ret_value;
}
}  // namespace ascend
}  // namespace device
}  // namespace mindspore
