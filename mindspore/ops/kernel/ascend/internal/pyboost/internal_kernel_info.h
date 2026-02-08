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
#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_INTERNAL_PYBOOST_INTERNAL_KERNEL_INFO_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_INTERNAL_PYBOOST_INTERNAL_KERNEL_INFO_H_

#include <memory>
#include <vector>
#include <string>
#include <optional>
#include <utility>
#include <unordered_map>

#include "include/runtime/hardware_abstract/kernel_base/kernel.h"
#include "include/internal.h"
#include "kernel/ascend/internal/tiling_mem_mgr.h"
#include "include/runtime/hardware_abstract/kernel_base/ms_factory.h"
#include "tools/profiler/profiler.h"
#include "tools/profiler/profiling.h"

#include "kernel/ascend/internal/internal_tiling_cache.h"
#include "kernel/ascend/internal/internal_spinlock.h"
#include "kernel/ascend/internal/internal_kernel_in_out_map.h"
#include "kernel/ascend/internal/internal_helper.h"
#include "kernel/ascend/internal/internal_kernel_build.h"
#include "kernel/ascend/kernel_plugin.h"
#include "kernel/ascend/internal/pyboost/internal_pyboost_utils.h"
#include "include/pynative/utils/pyboost/op_runner.h"
#include "include/pynative/utils/pyboost/pyboost_utils.h"
#include "kernel/ascend/aclnn/pyboost_impl/aclnn_utils.h"
#include "include/runtime/pipeline/pipeline.h"
#include "include/pynative/utils/runtime/op_executor.h"

namespace mindspore {
namespace kernel {
using TensorPtr = tensor::TensorPtr;
using TensorPtrList = tensor::TensorPtrList;
// 线程安全
class InternalKernelInfo {
 public:
  explicit InternalKernelInfo(std::string &&op_name) : kernel_name_(std::move(op_name)) {}
  virtual ~InternalKernelInfo() = default;

  void GetOrCreateKernel(const std::shared_ptr<pyboost::OpRunner> &op, const uint64_t &op_key,
                         const uint64_t &tiling_key, const TensorPtrList &inputs, const TensorPtrList &outputs);

  static void UpdateAddr(std::vector<internal::RawDeviceAddr> *addrlist, const TensorPtrList &tensorlist) {
    addrlist->resize(tensorlist.size());
    for (size_t i = 0; i < tensorlist.size(); i++) {
      if (tensorlist[i] == nullptr) {
        addrlist->at(i) = nullptr;
      } else {
        addrlist->at(i) = tensorlist[i]->device_address()->GetMutablePtr();
      }
    }
  }

  static void MallocWorkspace(const device::DeviceContext *device_context, size_t stream_id,
                              const internal::InternalOpPtr &internal_op, internal::WsAddrList *internal_wss_addr) {
    auto workspace_size_list = internal_op->GetWorkspaceSize();
    internal_wss_addr->resize(workspace_size_list.size());
    for (size_t i = 0; i < workspace_size_list.size(); i++) {
      auto work_ptr = std::make_shared<kernel::pyboost::MemBlock>(device_context, workspace_size_list[i], stream_id);
      internal_wss_addr->at(i) = work_ptr->ptr_;
    }
  }

  static void FreeWorkspace(const device::DeviceContext *device_context, internal::WsAddrList *internal_wss_addr) {
    for (size_t i = 0; i < internal_wss_addr->size(); i++) {
      device_context->device_res_manager_->FreeMemory(internal_wss_addr->at(i));
      internal_wss_addr->at(i) = nullptr;
    }
  }

 protected:
  bool IsInternalDtypeSupport(const TensorPtrList *ms_inputs, const TensorPtrList *ms_outputs);
  virtual uint64_t GetOrGenerateOpKey(const uint64_t &op_key) const { return op_key; }
  virtual uint64_t GetOrGenerateOpTilingKey(const uint64_t &tiling_key) const { return tiling_key; }
  virtual bool UpdateParam() { return true; }
  TilingCacheItemPtr GetOrGenerateTiling(const std::shared_ptr<pyboost::OpRunner> &op, const uint64_t &tiling_key);
  virtual internal::InternalOpPtr CreateKernel(const internal::InputsImmutableInfoList &inputs,
                                               const internal::OutputsImmutableInfoList &outputs) = 0;
  void TransInternalShapes(internal::ShapeInfoList *shapelist, const TensorPtrList &tensorlist, bool is_input = false);
  void TransInternalShapes(const TensorPtrList &inputs, const TensorPtrList &outputs);

  std::string kernel_name_;
  internal::InternalOpPtr internal_op_{nullptr};
  inline static std::unordered_map<uint64_t, internal::InternalOpPtr> hash_map_;
  internal::DtypeInfoList internal_inputs_dtype_;
  internal::DtypeInfoList internal_outputs_dtype_;
  internal::ShapeInfoList internal_inputs_shape_;
  internal::ShapeInfoList internal_outputs_shape_;
  internal::InputsImmutableInfoList inputs_ii_;
  internal::OutputsImmutableInfoList outputs_ii_;
  TilingCacheItemPtr tiling_info_{nullptr};

 private:
  void UpdateArgImmutableInfo(internal::ArgImmutableInfo *arginfo, const TensorPtr &tensor, internal::DataType dtype);
  void UpdateArgImmutableInfo(std::vector<internal::ArgImmutableInfo> *arginfos, const TensorPtrList &tensorlist,
                              bool is_input = false);
  SimpleSpinLock lock_;
};
using InternalKernelInfoPtr = std::shared_ptr<InternalKernelInfo>;

#define MS_INTERNAL_KERNEL_INFO_FACTORY_REG(PRIM_NAME_STR, DERIVE) \
  MS_KERNEL_FACTORY_REG_WITH_NAME_PARAM(InternalKernelInfo, PRIM_NAME_STR, DERIVE)

#define LAUNCH_INTERNAL_KERNEL(op, internal_op, device_context, tiling_ptr, inputs_addr, outputs_addr,             \
                               internal_wss_addr, kernel_name)                                                     \
  do {                                                                                                             \
    runtime::OpExecutor::DispatchLaunchTask(                                                                       \
      [op, internal_op, device_context, tiling_ptr, inputs_addr, outputs_addr, internal_wss_addr, kernel_name]() { \
        MS_LOG(DEBUG) << "Launch InternalKernel " << kernel_name << " start";                                      \
        auto device_ctx = op->device_context();                                                                    \
        device_ctx->device_res_manager_->BindDeviceToCurrentThread(false);                                         \
        internal_op->SetTilingInfo(tiling_ptr->tiling_info_);                                                      \
        auto stream_ptr = device_context->device_res_manager_->GetStream(op->stream_id());                         \
        auto &internal_wss_addr_ = const_cast<internal::WsAddrList &>(internal_wss_addr);                          \
        internal::InternalStatus status =                                                                          \
          internal_op->Launch(inputs_addr, outputs_addr, internal_wss_addr_, stream_ptr, kernel_name);             \
        InternalTilingCache::GetInstance().Unbind(tiling_ptr);                                                     \
        if (status != internal::InternalStatus::kInternalOk) {                                                     \
          MS_LOG(EXCEPTION) << "Launch InternalKernel failed, kernel_name: " << kernel_name;                       \
        }                                                                                                          \
        MS_LOG(DEBUG) << "Launch InternalKernel " << kernel_name << " end";                                        \
      });                                                                                                          \
  } while (false)

#define LAUNCH_INTERNAL(kernel_name_, op, internal_op_, inputs, outputs, tiling_info_)                                \
  do {                                                                                                                \
    const std::string kernel_name = kernel_name_;                                                                     \
    internal::InternalOpPtr internal_op = internal_op_;                                                               \
    TilingCacheItemPtr tiling_ptr = tiling_info_;                                                                     \
    auto device_context = op->device_context();                                                                       \
    pyboost::PyBoostUtils::MallocInternalOpInputs(device_context, inputs);                                            \
    pyboost::PyBoostUtils::MallocOpOutputs(device_context, outputs);                                                  \
    internal::InputsAddrList inputs_addr;                                                                             \
    internal::OutputsAddrList outputs_addr;                                                                           \
    InternalKernelInfo::UpdateAddr(&inputs_addr, inputs);                                                             \
    InternalKernelInfo::UpdateAddr(&outputs_addr, outputs);                                                           \
    internal::WsAddrList internal_wss_addr;                                                                           \
    InternalKernelInfo::MallocWorkspace(device_context, op->stream_id(), internal_op, &internal_wss_addr);            \
    LAUNCH_INTERNAL_KERNEL(op, internal_op, device_context, tiling_ptr, inputs_addr, outputs_addr, internal_wss_addr, \
                           kernel_name);                                                                              \
  } while (false)

}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_INTERNAL_PYBOOST_INTERNAL_KERNEL_INFO_H_
