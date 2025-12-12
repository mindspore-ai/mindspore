/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_RUNTIME_FRAMEWORK_ACTOR_MEM_COUNTED_CACHE_H_
#define MINDSPORE_CCSRC_RUNTIME_FRAMEWORK_ACTOR_MEM_COUNTED_CACHE_H_

#include <memory>
#include <unordered_map>
#include "include/runtime/hardware_abstract/kernel_base/kernel_tensor.h"
#include "include/runtime/hardware_abstract/device_context/device_context.h"
#include "backend/ms_backend/runtime/actors/hyper_offload/mem_action_mgr.h"
#include "backend/ms_backend/runtime/actors/hyper_offload/hyper_offload_utils.h"

#include "include/runtime/utils/runtime_conf/runtime_conf.h"

namespace mindspore {
namespace runtime {

enum class TensorStatus { kReady, kCopyingToDevice, kCopyingToHost };
static const std::string horizon = common::GetEnv("MS_DEV_REMOTE_HORIZON");
static const std::string copy_out_wait_num = common::GetEnv("MS_DEV_REMOTE_COPY_OUT_WAIT_NUM");
static const std::string buffer_size = common::GetEnv("MS_DEV_REMOTE_BUFFER");
static const std::string copy_out_op_num = common::GetEnv("MS_DEV_REMOTE_COPY_OUT_OP_NUM");
static const size_t kInitHorizon = horizon.empty() ? 1 : static_cast<size_t>(std::stoull(horizon));
static const size_t kCopyOutWaitNum =
  copy_out_wait_num.empty() ? 5 : static_cast<size_t>(std::stoull(copy_out_wait_num));
static const float kBufferSizeScalar = buffer_size.empty() ? 0.1 : std::stof(buffer_size);
static const size_t kCopyOutWaitOpNum = copy_out_op_num.empty() ? 5 : static_cast<size_t>(std::stoull(copy_out_op_num));

struct TensorInfo {
  KernelTensorPtr kernel_tensor;
  size_t next_use_idx;
  bool need_free;
  TensorStatus status;
  KernelTensorPtr old_kernel_tensor;
  size_t start_wait_idx;

  TensorInfo(const KernelTensorPtr &kernel_tensor, size_t next_use_idx, bool need_free, TensorStatus status,
             const KernelTensorPtr &original_kernel = nullptr)
      : kernel_tensor(kernel_tensor),
        next_use_idx(next_use_idx),
        need_free(need_free),
        status(status),
        old_kernel_tensor(original_kernel),
        start_wait_idx(kCopyOutWaitNum) {}

  bool operator<(const TensorInfo &other) const {
    if (next_use_idx == other.next_use_idx) {
      return kernel_tensor.get() < other.kernel_tensor.get();
    }
    return next_use_idx < other.next_use_idx;
  }
};

using TensorInfoPtr = std::shared_ptr<TensorInfo>;
using TensorInfoPtrList = std::vector<TensorInfoPtr>;
using MemActionMgrPtr = std::shared_ptr<MemActionMgr>;

struct PtrLess {
  bool operator()(const TensorInfoPtr &a, const TensorInfoPtr &b) const {
    if (!a || !b) return static_cast<bool>(b) < static_cast<bool>(a);
    return *a < *b;
  }
};

using TensorInfoPtrSet = std::set<TensorInfoPtr, PtrLess>;

class MemCountedCache {
 public:
  MemCountedCache() : mem_action_mgr_(std::make_shared<MemActionMgr>()) {
    horizon_ = kInitHorizon;
    float max_size = runtime::RuntimeConf::GetInstance()->mem_max_size();
    max_mem_ = FloatToSize(max_size * kGBToByte);
    buffer_size_ = kBufferSizeScalar * max_mem_;
  }

  ~MemCountedCache() { ClearMCCInstance(); }

  void SetCopyStreamId(size_t copy_out_stream_id, size_t copy_in_stream_id);

  void SetHorizon(size_t horizon) { horizon_ = horizon; }

  size_t GetHorizon() { return horizon_; }

  void SetConditionSwitchIdxs(const std::queue<size_t> &conditionswitch_idxs) {
    conditionswitch_idxs_ = conditionswitch_idxs;
    next_conditionswitch_idx_ = conditionswitch_idxs_.front();
  }

  // Check if all inputs in device, if not, insert Wait Events
  // Args:
  //  idx: Current execution index
  //  kernel_tensors: inputs kernel tensors and next user idx
  void CheckInputsAvailable(size_t idx, const std::vector<std::pair<KernelTensorPtr, size_t>> &kernel_tensors,
                            const device::DeviceContext *device_context);

  KernelTensorPtrPairList CheckOutputsEnough(size_t idx,
                                             const std::vector<std::pair<KernelTensorPtr, size_t>> &kernel_tensors,
                                             const device::DeviceContext *device_context);

  // After finish execution, analyze D2H and H2D actions
  // Args:
  //  device_context: device context used to insert events
  //  stream id: current execution stream id
  //  kernel_tensors_info: kernel tensor and next user idx
  // Return:
  //  return copydata pair {orignal_kernel_tenor, new_kernel_tensor}
  KernelTensorPtrPairList ProcessOutput(const device::DeviceContext *device_context, uint32_t stream_id, size_t idx,
                                        const std::vector<std::pair<KernelTensorPtr, size_t>> &kernel_tensors_info);

  // Update tensor after copy task
  // Args:
  //  kernel_tensor: kernel tensor need refresh info
  //  event_type: which task finished
  void UpdateTensorStatus(const KernelTensorPtr &kernel_tensor, HyperOffloadEventType event_type);

  void ClearMCCInstance() {
    MS_VLOG(VL_HYPER_OFFLOAD_INFO) << "Clear MCCInstance";
    device_cache_.clear();
    host_cache_.clear();
    cur_inputs_.clear();
    mem_action_mgr_->ClearMAMInstance();
  }

 private:
  void CleanExpiredDeviceTensors(size_t current_idx);
  void InsertNewTensorsToDevice(size_t idx, const std::vector<std::pair<KernelTensorPtr, size_t>> &kernel_tensors_info);
  KernelTensorPtrPairList LoadAllWithinHorizon(size_t current_idx, size_t horizon, uint32_t stream_id,
                                               const device::DeviceContext *device_context);
  KernelTensorPtrPairList OffloadAndPrefetchTensors(size_t current_idx, size_t horizon, uint32_t stream_id,
                                                    size_t need_load_size, size_t available_mem,
                                                    const device::DeviceContext *device_context);

  KernelTensorPtrPairList Offload(size_t first_offloadable_idx, size_t need_size, size_t stream_id,
                                  const device::DeviceContext *device_context);
  size_t GetIdleMemSize();
  size_t GetDeviceAvailableMemSize();
  size_t GetUsedMemSize();
  size_t GetActualHorizon(size_t cur_idx);
  TensorInfoPtrList GetCanOffloadTensorInfo(size_t first_offload_idx);

  MemActionMgrPtr mem_action_mgr_{nullptr};

  size_t max_mem_;
  size_t buffer_size_;

  size_t to_host_stream_;
  size_t to_device_stream_;

  TensorInfoPtrSet device_cache_;
  TensorInfoPtrSet host_cache_;

  std::vector<KernelTensorPtr> cur_inputs_;

  // Handle condition switch
  size_t horizon_;
  size_t next_conditionswitch_idx_ = SIZE_MAX;
  std::queue<size_t> conditionswitch_idxs_;
};

}  // namespace runtime
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_RUNTIME_FRAMEWORK_ACTOR_MEM_COUNTED_CACHE_H_
