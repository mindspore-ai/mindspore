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

#include "include/pynative/utils/pyboost/custom/pyboost_extension.h"

#ifndef _MSC_VER
#include <cxxabi.h>
#endif

#include <tuple>
#include <string>
#include <vector>
#include <memory>

#include "utils/stream_guard.h"
#include "include/pynative/utils/pyboost/custom/tensor.h"
#include "include/utils/tensor_utils.h"
#include "include/runtime/hardware_abstract/device_context/device_context.h"
#include "include/pynative/utils/runtime/op_runner.h"
#include "mindspore/core/include/utils/ms_utils.h"
#include "include/backend/common/device_address_utils.h"
#include "mindspore/ccsrc/include/runtime/utils/runtime_conf/runtime_conf.h"
#include "mindspore/core/include/utils/ms_context.h"

namespace ms {
using PyBoostUtils = mindspore::kernel::pyboost::PyBoostUtils;

namespace inner {
std::string GetFunctionName(const char *name) {
#ifdef _MSC_VER
  return name;
#else
  int status = -1;
  std::unique_ptr<char, void (*)(void *)> res{abi::__cxa_demangle(name, nullptr, nullptr, &status), std::free};
  return (status == 0) ? res.get() : name;
#endif
}

void SetPromise(const std::string &, const std::tuple<mindspore::stub::StubNodePtr> &tuple, const ms::Tensor &output) {
  mindspore::tensor::SetPromise(tuple, output.tensor());
}

/**
 * @brief Memory block structure for managing device memory allocation.
 */
struct MemBlock {
  /**
   * @brief Constructs a MemBlock and allocates memory on the device.
   * @param device_context The device context for memory allocation.
   * @param size The size of the memory block to allocate.
   * @param stream_id The stream ID for the memory allocation.
   * @throws If memory allocation fails.
   */
  MemBlock(const mindspore::device::DeviceContext *device_context, size_t size, uint32_t stream_id) {
    ptr_ = device_context->device_res_manager_->AllocateMemory(size, stream_id);
    if (ptr_ == nullptr) {
      MS_LOG(EXCEPTION) << "Alloc workspace failed, size:" << size << ", stream_id:" << stream_id;
    }
    device_context_ = device_context;
  }

  /**
   * @brief Destructor for MemBlock. Frees the allocated memory.
   */
  ~MemBlock() { device_context_->device_res_manager_->FreeMemory(ptr_); }

  // Pointer to the allocated memory block.
  void *ptr_;
  // The device context used for allocation.
  const mindspore::device::DeviceContext *device_context_;
};
using MemBlockPtr = std::shared_ptr<MemBlock>;

mindspore::device::DeviceType GetDeviceTarget() { return mindspore::DeviceManagerConf::GetInstance()->device_type(); }
}  // namespace inner

namespace pynative {
PyboostRunner::PyboostRunner(const std::string &op_name) : _op_name_(op_name) {
  _device_context_ = mindspore::runtime::OpRunner::GetDeviceContext(inner::GetDeviceTarget());
}

void PyboostRunner::Run(const std::vector<Tensor> &inputs, const std::vector<Tensor> &outputs) {
  _inputs_ = inputs;
  _outputs_ = outputs;
  this->_Run();
}

void PyboostRunner::_Run() {
  this->_PrepareStream();
  this->_PrepareDeviceAddress();
  PyBoostUtils::DispatchRun(std::make_shared<mindspore::runtime::PyBoostDeviceTask>([runner = shared_from_this()]() {
    static auto simu = mindspore::common::IsCompileSimulation();
    if (simu) {
      return;
    }
    runner->_MallocDeviceAddress();
    runner->_MallocWorkspace();
    runner->_DispatchLaunchTask();
  }));
}

void PyboostRunner::_PrepareStream() {
  _stream_id_ = static_cast<size_t>(mindspore::CurrentStream::id());
  _stream_ = _device_context_->device_res_manager_->GetStream(_stream_id_);
}

void PyboostRunner::_PrepareDeviceAddress() {
  for (size_t i = 0; i < _inputs_.size(); i++) {
    if (!_inputs_[i].is_defined()) {
      continue;
    }
    mindspore::runtime::DeviceAddressUtils::CreateInputTensorAddress(_device_context_, _stream_id_, i,
                                                                     _inputs_[i].tensor());
  }
  std::vector<mindspore::tensor::TensorPtr> outs;
  outs.reserve(_outputs_.size());
  for (auto &out : _outputs_) {
    if (out.tensor() != nullptr && out.tensor()->device_address() == nullptr) {
      (void)outs.emplace_back(out.tensor());
    }
  }
  mindspore::runtime::DeviceAddressUtils::CreateOutputTensorAddress(_device_context_, _stream_id_, outs);
}

void PyboostRunner::_MallocDeviceAddress() {
  {
    // input tensors
    mindspore::runtime::ProfilerRecorder profiler(mindspore::runtime::ProfilerModule::kPynative,
                                                  mindspore::runtime::ProfilerEvent::kPyBoostMallocInput,
                                                  mindspore::runtime::ProfilerRecorder::kNoName, false);
    for (auto &inp : _inputs_) {
      if (!inp.is_defined()) {
        continue;
      }
      mindspore::kernel::pyboost::PyBoostUtils::MallocForInput(_device_context_, inp.tensor());
    }
  }
  {
    // output tensors
    mindspore::runtime::ProfilerRecorder profiler(mindspore::runtime::ProfilerModule::kPynative,
                                                  mindspore::runtime::ProfilerEvent::kPyBoostMallocOutput,
                                                  mindspore::runtime::ProfilerRecorder::kNoName, false);
    std::vector<mindspore::tensor::TensorPtr> outs;
    outs.reserve(_outputs_.size());
    for (auto &out : _outputs_) {
      if (out.tensor() != nullptr) {
        (void)outs.emplace_back(out.tensor());
      }
    }
    mindspore::runtime::DeviceAddressUtils::MallocForOutputs(_device_context_, outs);
  }
}

void PyboostRunner::_MallocWorkspace() {
  // calculate and alloc workspace
  inner::MemBlockPtr ws_mng;
  auto workspace_size = this->CalcWorkspace();
  if (workspace_size > 0) {
    ws_mng = std::make_shared<inner::MemBlock>(_device_context_, workspace_size, _stream_id_);
    this->_workspace_ptr_ = ws_mng->ptr_;
  } else {
    this->_workspace_ptr_ = nullptr;
  }
  this->ProcessWithWorkspace();
}

void PyboostRunner::ProcessCrossStreamAddress() {
  mindspore::tensor::TensorPtrList tensors;
  tensors.reserve(_inputs_.size() + _outputs_.size());
  for (auto &t : _inputs_) {
    if (t.is_defined()) {
      (void)tensors.emplace_back(t.tensor());
    }
  }
  for (auto &t : _outputs_) {
    if (t.is_defined()) {
      (void)tensors.emplace_back(t.tensor());
    }
  }
  mindspore::runtime::DeviceAddressUtils::ProcessCrossStreamAddress(_op_name_, _device_context_, _stream_id_, tensors);
}

void PyboostRunner::_DispatchLaunchTask() {
  mindspore::runtime::OpExecutor::DispatchLaunchTask([runner = shared_from_this()]() {
    mindspore::runtime::ProfilerRecorder profiler(mindspore::runtime::ProfilerModule::kPynative,
                                                  mindspore::runtime::ProfilerEvent::kPyNativeLaunchTask,
                                                  runner->op_name(), false);
    runner->_device_context_->device_res_manager_->BindDeviceToCurrentThread(false);
    runner->LaunchKernel();
    if (mindspore::runtime::RuntimeConf::GetInstance()->launch_blocking()) {
      if (!runner->_device_context_->device_res_manager_->SyncAllStreams()) {
        MS_LOG(EXCEPTION) << "SyncStream failed for op " << runner->op_name();
      }
    } else {
      runner->ProcessCrossStreamAddress();
    }
  });
}
}  // namespace pynative
}  // namespace ms
