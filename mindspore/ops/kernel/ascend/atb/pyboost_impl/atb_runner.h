/**
 * Copyright 2025 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License")
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

#ifndef MINDSPORE_OPS_KERNEL_ASCEND_PYBOOST_ATB_RUNNER_H_
#define MINDSPORE_OPS_KERNEL_ASCEND_PYBOOST_ATB_RUNNER_H_

#include <vector>
#include <string>
#include <map>
#include <memory>
#include "atb/operation.h"
#include "include/pynative/utils/runtime/op_executor.h"
#include "kernel/ascend/atb/kernel_mod_impl/atb_adapter.h"
#include "kernel/ascend/atb/pyboost_impl/atb_runner_base.h"

namespace mindspore::kernel::pyboost {
class ATBOpCache {
 public:
  ~ATBOpCache() { Clear(); }
  static ATBOpCache &GetInstance() {
    static ATBOpCache instance;
    return instance;
  }

  template <typename ParamType>
  atb::Operation *GetOp(const ParamType &param, const std::string &atb_name) {
    // if param is related to input, then 'GatherHash' in atb_adapter.cc needed to be overloaded.
    auto hash_id = device::ascend::AtbHash(param, atb_name);
    {
      std::lock_guard<std::mutex> lock(mutex_);
      auto iter = atb_op_cache_.find(hash_id);
      if (iter == atb_op_cache_.end()) {
        atb::Operation *op;
        auto ret = atb::CreateOperation(param, &op);
        if (ret) {
          MS_LOG(ERROR) << "Create atb operation failed, ret: " << ret;
        }
        atb_op_cache_[hash_id] = op;
        return op;
      }
      return iter->second;
    }
  }

  void Clear() {
    for (auto &item : atb_op_cache_) {
      auto ret = atb::DestroyOperation(item.second);
      if (ret) {
        MS_LOG(ERROR) << "Destroy atb operation failed, ret: " << ret;
      }
    }
  }

 private:
  ATBOpCache() = default;
  std::map<uint64_t, atb::Operation *> atb_op_cache_;
  std::mutex mutex_;
};

#define MS_ATB_RUNNER_REG(atb_name, atb_param_type)                                                         \
  class atb_name##ATBRunner : public ATBRunnerBase {                                                        \
   public:                                                                                                  \
    atb_name##ATBRunner() = default;                                                                        \
    void InitProcess(const std::string &atb_name, void *param_ptr) override {                               \
      auto param = reinterpret_cast<atb_param_type *>(param_ptr);                                           \
      op_name_ = #atb_name;                                                                                 \
      atb_op_ = ATBOpCache::GetInstance().GetOp(*param, op_name_);                                          \
    }                                                                                                       \
                                                                                                            \
    void GetWorkSpaceInfo(const device::DeviceContext *device_context, uint32_t stream_id,                  \
                          const std::vector<tensor::TensorPtr> &inputs,                                     \
                          const std::vector<tensor::TensorPtr> &outputs) override {                         \
      param_setter_.Clear();                                                                                \
      for (const auto input : inputs) {                                                                     \
        param_setter_.Input(input);                                                                         \
      }                                                                                                     \
      for (const auto output : outputs) {                                                                   \
        param_setter_.Output(output);                                                                       \
      }                                                                                                     \
      stream_ptr_ = device_context->device_res_manager_->GetStream(stream_id);                              \
      ws_size_ = device::ascend::GetWorkSpaceSize(atb_op_, param_setter_.variant_pack, stream_ptr_);        \
    }                                                                                                       \
                                                                                                            \
    void Run(uint32_t stream_id, const device::DeviceContext *device_context) {                             \
      auto work_ptr = std::make_shared<kernel::pyboost::MemBlock>(device_context, ws_size_, stream_id);     \
      runtime::OpExecutor::DispatchLaunchTask([this, device_context, work_ptr]() {                          \
        runtime::ProfilerRecorder profiler(runtime::ProfilerModule::kPynative,                              \
                                           runtime::ProfilerEvent::kPyNativeLaunchTask, op_name_, false);   \
        MS_LOG(DEBUG) << "launch task start, " << op_name_;                                                 \
        device_context->device_res_manager_->BindDeviceToCurrentThread(false);                              \
        device::ascend::Launch(atb_op_, param_setter_.variant_pack, work_ptr->ptr_, ws_size_, stream_ptr_); \
        MS_LOG(DEBUG) << "launch task end, " << op_name_;                                                   \
      });                                                                                                   \
      auto sync = runtime::RuntimeConf::GetInstance()->launch_blocking();                                   \
      if (sync) {                                                                                           \
        if (!device::ascend::AscendStreamMng::GetInstance().SyncAllStreams()) {                             \
          MS_LOG(EXCEPTION) << "SyncStream failed for op " << op_name_;                                     \
        }                                                                                                   \
      }                                                                                                     \
    }                                                                                                       \
                                                                                                            \
   private:                                                                                                 \
    std::string op_name_;                                                                                   \
    atb::Operation *atb_op_;                                                                                \
    void *stream_ptr_;                                                                                      \
    uint64_t ws_size_;                                                                                      \
    device::ascend::ParamSetter param_setter_;                                                              \
  };                                                                                                        \
  MS_ATB_RUNNER_FACTORY_REG(atb_name##ATBRunner, atb_name##ATBRunner)
}  // namespace mindspore::kernel::pyboost

#endif  // MINDSPORE_OPS_KERNEL_ASCEND_PYBOOST_ATB_RUNNER_H_
