/**
 * Copyright 2024-2025 Huawei Technologies Co., Ltd
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

#include "kernel/ascend/dvm/pyboost_impl/lazy_fusion_kernel.h"
#include <utility>
#include <string>
#include <memory>
#include "kernel/ascend/dvm/pyboost_impl/lazy_fusion_flags.h"
#include "plugin/ascend/res_manager/stream_manager/ascend_stream_manager.h"
#include "tools/profiler/profiling.h"
#include "include/runtime/pipeline/pipeline.h"
#include "utils/file_utils.h"
#include "include/runtime/hardware_abstract/stream/multi_stream_controller.h"

namespace mindspore {
namespace kernel {
namespace {
static constexpr uint32_t kLoadFlag = 0x10000u;
static constexpr uint32_t kIndexFlag = 0xffffu;

void *WsAllocCallback(uint64_t size, void *user_data) {
  auto kernel = static_cast<LazyFusionKernelAscend *>(user_data);
  MS_LOG(INFO) << "Alloc workspace for dvm kernel, kernel id is " << kernel->id() << " " << kernel << " size: " << size;
  return kernel->AllocWorkspace(size);
}

size_t GetOffset(const TensorPtr &tensor) {
  const auto &storage_info = tensor->storage_info();
  return storage_info == nullptr ? 0 : storage_info->storage_offset * tensor->DataItemSize();
}
}  // namespace

void LazyFusionQueue::Push(const runtime::AsyncTaskPtr &task) {
  // When the task type is kBpropTask, this task is only for facilitating destruction and does not involve actual
  // operator offloading, thus no synchronization is required
  if (task->task_type() != runtime::kBpropTask) {
    FlushLazyFusion();
  }
  AsyncRQueue::Push(task);
}

void LazyFusionQueue::Wait() {
  auto current_level = GetCurrentLevel();
  if (current_level >= wait_level_) {
    MS_LOG(DEBUG) << "No need to wait, current level " << current_level << " AsyncQueue name " << name_;
    // Only need to wait the low level thread.
    return;
  }
  FlushLazyFusion();
  AsyncRQueue::Wait();
}

bool LazyFusionQueue::Empty() {
  // This function only been called by OpExecutor::RunQueueEmpty, which only be called in non-pyboost sync running.
  // In case async running + sync running in the same process, AsyncRQueue::Empty does not means the queue is really
  // empty, maybe the dvm kernel has not been enqueued.
  if (!runtime::AsyncRQueue::Empty()) {
    return false;
  }
  // if LazyFusionManager::current_ is not null, means LazyFusionManager::Flush has not been called.
  return g_lazy_fusion_manager.Empty();
}

void LazyFusionQueue::WorkerJoin() {
  // If the process exit without calling asnumpy()/sync(), the atexit function will call WorkerJoin()
  // first, then call Wait(). The WorkerJoin function will exit the thread, then when call Wait(), it
  // push a dvm task to the queue, and will stuck in the dead loop because the dvm task will never be
  // executed as the thread already exit. So we need to push dvm task to the queue inside WorkerJoin() first.
  FlushLazyFusion();
  runtime::AsyncRQueue::WorkerJoin();
}

runtime::kThreadWaitLevel LazyFusionQueue::GetCurrentLevel() {
  runtime::kThreadWaitLevel current_level{runtime::kThreadWaitLevel::kLevelUnknown};
  auto thread_id = std::this_thread::get_id();
  std::unique_lock<std::mutex> lock(level_mutex_);
  auto iter = thread_id_to_wait_level_.find(thread_id);
  if (iter != thread_id_to_wait_level_.end()) {
    current_level = iter->second;
  }
  return current_level;
}

LazyFusionManager g_lazy_fusion_manager;

LazyFusionManager::~LazyFusionManager() {
  while (!pool_.empty()) {
    auto top = pool_.front();
    delete top;
    pool_.pop();
  }
}

LazyFusionKernelAscend *LazyFusionManager::Get(const device::DeviceContext *context, size_t stream) {
  static bool runtime_init = false;
  if (!runtime_init) {
    auto ms_context = MsContext::GetInstance();
    MS_EXCEPTION_IF_NULL(ms_context);
    bool enable_deterministic = ms_context->get_param<std::string>(MS_CTX_DETERMINISTIC) == "ON";
    dvm::SetDeterministic(enable_deterministic);
    MS_LOG(INFO) << "Set dvm deterministic " << (enable_deterministic ? "on" : "off");
    bool enable_tuning = LazyFusionFlags::GetInstance().online_tuning;
    if (enable_deterministic) {
      enable_tuning = false;
      MS_LOG(WARNING) << "online tuning can not be enabled if set deterministic";
    }
    dvm::SetOnlineTuning(enable_tuning);
    MS_LOG(INFO) << "Set dvm online tuning " << (enable_tuning ? "on" : "off");
    runtime_init = true;
  }
  if (current_ != nullptr) {
    if (current_->stream_id() == stream) {
      return current_;
    }
    current_->Flush();
  }
  current_ = NewKernel();
  current_->Reset(context, stream);
  current_->set_id(id_.fetch_add(1, std::memory_order_relaxed));
  return current_;
}

void LazyFusionManager::Flush() {
  if (current_ != nullptr) {
    current_->Flush();
    current_ = nullptr;
  }
}

LazyFusionKernelAscend *LazyFusionManager::NewKernel() {
  {
    std::lock_guard<std::mutex> guard(mutex_);
    if (!pool_.empty()) {
      auto k = pool_.front();
      pool_.pop();
      return k;
    }
  }
  return new LazyFusionKernelAscend();
}

LazyFusionKernelAscend::LazyFusionKernelAscend() { EagerReset(WsAllocCallback, this); }

LazyFusionKernelAscend::~LazyFusionKernelAscend() {
  for (auto load : inputs_) {
    delete load;
  }
}

dvm::ShapeRef *LazyFusionKernelAscend::GetShapeRef(const ShapeVector &shape) {
  auto &item = cached_shape_.emplace_back(shape, nullptr);
  item.second = std::make_shared<dvm::ShapeRef>(item.first);
  return item.second.get();
}

dvm::NDObject *LazyFusionKernelAscend::Input(const TensorPtr &x, bool enable_cast,
                                             const std::optional<ShapeVector> &shape) {
  auto input_type = TransType(x->data_type());
  bool cast_to_fp32 = (enable_cast && input_type == dvm::DType::kBFloat16);
  auto device_addr = x->device_address();
  MS_EXCEPTION_IF_NULL(device_addr);
  auto xp = device_addr.get();
  // ops_map_ uses device_address as key, because TensorPtr is not continuous, e.g. A is use by B, TensorPtr
  // of A may be different from TensorPtr of B's input, which will affect the relationship of dvm NDObject.
  auto iter = ops_map_.find(xp);
  if (iter == ops_map_.end()) {
    if (input_used_ == inputs_.size()) {
      inputs_.push_back(new Load());
    }
    auto load = inputs_[input_used_++];
    if (shape == std::nullopt) {
      load->shape = cached_shape_.emplace_back(x->shape(), nullptr).first;
    } else {
      load->shape = cached_shape_.emplace_back(shape.value(), nullptr).first;
    }
    auto load_op = dvm::Kernel::Load(nullptr, &(load->shape), input_type);
    auto op = cast_to_fp32 ? Cast(load_op, dvm::DType::kFloat32) : load_op;
    load->op = load_op;
    load->tensor = x;
    ops_map_[xp] = op;
    return op;
  }
  auto op = iter->second;
  op = cast_to_fp32 ? Cast(op, dvm::DType::kFloat32) : op;
  return op;
}

void LazyFusionKernelAscend::Output(const TensorPtr &tensor, dvm::NDObject *obj, bool inplace) {
  auto tensor_type = TransType(tensor->data_type());
  if (GetDType(obj) != tensor_type) {
    obj = Cast(obj, tensor_type);
  }
  void *dev_addr = tensor->device_address().get();
  if (inplace && ops_map_.find(dev_addr) != ops_map_.end()) {
    // %0 = Mul(p0, p1)
    // %1 = InplaceAddExt(%0, p2, 1)
    // here Inplace op's first input come from another op, should not emit Store for %0
    for (int64_t i = SizeToLong(outputs_.size()) - 1; i >= 0; --i) {
      auto idx = LongToSize(i);
      if (outputs_[idx].tensor->device_address().get() == dev_addr) {
        outputs_[idx].skip = true;
        break;
      }
    }
  }
  (void)outputs_.emplace_back(obj, tensor, false, inplace);
  ops_map_[dev_addr] = obj;
}

void *LazyFusionKernelAscend::AllocWorkspace(uint64_t size) {
  auto mem = std::make_shared<kernel::pyboost::MemBlock>(device_context_, size, stream_id_);
  workspace_.push_back(mem);
  return mem->ptr_;
}

void LazyFusionKernelAscend::Launch() {
  MS_LOG(INFO) << "Run launch task dvm kernel start, kernel id is " << id() << " " << this;
  runtime::ProfilerRecorder profiler(runtime::ProfilerModule::kPynative, runtime::ProfilerEvent::kPyNativeLaunchTask,
                                     "FlushEager", false);
  device_context_->device_res_manager_->BindDeviceToCurrentThread(false);
  auto stream_ptr = device_context_->device_res_manager_->GetStream(stream_id_);
  if (profiler::Profiler::GetInstance(kAscendDevice)->GetEnableFlag()) {
    EagerMsProfLaunch(stream_ptr);
  } else {
    EagerLaunch(stream_ptr);
  }
  if (LazyFusionFlags::GetInstance().synchronize && !device::ascend::AscendStreamMng::GetInstance().SyncAllStreams()) {
    MS_LOG(EXCEPTION) << "SyncStream failed for dvm kernel, kernel id is " << id() << " " << this;
  }
  ClearKernel();
  MS_LOG(INFO) << "Run launch task dvm kernel end, kernel id is " << id() << " " << this;
}

void LazyFusionKernelAscend::Flush() {
  if (outputs_.empty()) {
    Clear();
    return;
  }
  // Async
  auto task = std::make_shared<runtime::PyBoostDeviceTask>([this]() {
    MS_LOG(INFO) << "Run device task dvm kernel start, kernel id is " << id() << " " << this;
    {
      runtime::ProfilerRecorder profiler(runtime::ProfilerModule::kPynative, runtime::ProfilerEvent::kPyBoostDeviceTask,
                                         "MallocIO", false);
      reloc_entry_.reserve(input_used_);
      // Malloc for input tensors
      for (size_t i = 0; i < input_used_; ++i) {
        auto input = inputs_[i];
        pyboost::PyBoostUtils::MallocForInput(device_context_, input->tensor);
        auto device_address = std::static_pointer_cast<device::DeviceAddress>(input->tensor->device_address());
        MS_EXCEPTION_IF_NULL(device_address);
        auto offset = GetOffset(input->tensor);
        auto dev_mem = device_address->GetMutablePtr();
        reloc_entry_.emplace_back(input->op, static_cast<void *>(static_cast<uint8_t *>(dev_mem) + offset));
        auto stream_id = device_address->stream_id();
        if (stream_id_ != stream_id) {  // to do: public and use runtime::DeviceAddressUtils::GetCrossStreamAddressInfo
          cross_stream_addrs_.emplace_back(stream_id, dev_mem);
        }
      }
      // Malloc for output tensors
      bool has_store = false;
      for (auto &out : outputs_) {
        if (out.skip) {
          continue;
        }
        auto offset = GetOffset(out.tensor);
        std::weak_ptr<DeviceAddress> dev_addr(out.tensor->device_address());
        out.tensor.reset();
        auto device_address = dev_addr.lock();
        if (device_address == nullptr) {
          out.skip = true;
          continue;
        }
        if (device_address->GetPtr() == nullptr) {
          device::tracker::CALL_MEMORY_TRACKER_WITH_FILE(AddMemInfo, "PyNative",
                                                         memory::mem_pool::MemType::kPyNativeOutput,
                                                         device_address->GetSize(), device_address.get());
          if (!device_context_->device_res_manager_->AllocateMemory(device_address.get())) {
            MS_LOG(EXCEPTION) << "Allocate memory failed for dvm kernel output, kernel id is " << id() << " " << this;
          }
        }
        auto dev_mem = device_address->GetMutablePtr();
        auto store = dvm::Kernel::Store(static_cast<void *>(static_cast<uint8_t *>(dev_mem) + offset), out.op);
        if (out.inplace) {
          dvm::Kernel::SetStoreInplace(store);
        }
        has_store = true;
      }
      if (LazyFusionFlags::GetInstance().dump_as_text) {
        DumpGraph();
      }
      if (!has_store) {
        MS_LOG(INFO) << "Skip launch task dvm kernel, kernel id is " << id() << " " << this
                     << " output size: " << outputs_.size();
        Clear();
        return;
      }
    }
    static auto simu = !common::GetEnv(kSimulationLevel).empty();
    if (!simu) {
      // Codegen
      {
        runtime::ProfilerRecorder profiler(runtime::ProfilerModule::kPynative,
                                           runtime::ProfilerEvent::kPyBoostDeviceTask, "CodeGen", false);
        if (LazyFusionFlags::GetInstance().dump_as_text) {
          dump_buf_ << "[lazy_fusion before split] kernel id: " << id() << "\n";
          dump_buf_ << Dump() << "\n";
          LazyFusionDump::Instance().DumpKernelInfo(&dump_buf_);
          EagerCodeGen(reloc_entry_.data(), reloc_entry_.size());
          dump_buf_ << "[lazy_fusion after split]\n";
          dump_buf_ << Dump() << "\n";
          dump_buf_ << Das() << "\n";
          LazyFusionDump::Instance().DumpKernelInfo(&dump_buf_);
        } else {
          EagerCodeGen(reloc_entry_.data(), reloc_entry_.size());
        }
      }
      // Launch
      ClearGraph();
      runtime::OpExecutor::DispatchLaunchTask([this]() { Launch(); });
      if (!cross_stream_addrs_.empty()) {
        MS_LOG(INFO) << "cross_stream_addrs_ size: " << cross_stream_addrs_.size() << " kernel id is " << id() << " "
                     << this;
        auto &ms = device::DeviceContextManager::GetInstance().GetMultiStreamController(
          device_context_->device_context_key().device_type_);
        ms->Refresh();
        auto task_id_on_stream = ms->LaunchTaskIdOnStream(stream_id_);
        ms->RecordEvent(task_id_on_stream, stream_id_, cross_stream_addrs_);
      }
    }
    MS_LOG(INFO) << "Run device task dvm kernel end, kernel id is " << id() << " " << this;
  });
  runtime::ProfilerAnalyzer::GetInstance().RecordFlowData(task->task_id());
  runtime::Pipeline::Get().backend_stage()->runtime::AsyncRQueue::Push(task);  // No flush here
}

std::pair<bool, uint32_t> LazyFusionKernelAscend::GetInputIdx(const TensorPtr &tensor) {
  if (tensor == nullptr) {
    return std::make_pair(false, 0u);
  }
  for (int64_t i = SizeToLong(input_used_) - 1; i >= 0; --i) {
    if (inputs_[LongToSize(i)]->tensor == tensor) {
      return std::make_pair(true, kLoadFlag | static_cast<uint32_t>(i));
    }
  }
  return std::make_pair(false, 0u);
}

std::pair<bool, uint32_t> LazyFusionKernelAscend::GetOutputIdx(const TensorPtr &tensor) {
  if (tensor == nullptr) {
    return std::make_pair(false, 0u);
  }
  auto dev_addr = tensor->device_address();
  for (size_t i = 0; i < outputs_.size(); ++i) {
    if (outputs_[i].tensor->device_address() == dev_addr) {
      return std::make_pair(true, static_cast<uint32_t>(i));
    }
  }
  return std::make_pair(false, 0u);
}

void LazyFusionKernelAscend::DumpGraph() {
  dump_buf_ << "kernel id: " << id() << " " << this << "\n";
  for (size_t i = 0; i < input_used_; ++i) {
    auto input_tensor = inputs_[i]->tensor;
    dump_buf_ << "p" << i << ": " << LazyFusionDump::Instance().ToString(input_tensor) << "\n";
  }
  dump_buf_ << "lazy_fusion_graph";
  for (const auto &op : dump_ops_) {
    dump_buf_ << "_" << op.name;
  }
  dump_buf_ << "() {\n";
  size_t output_idx = 0;
  for (const auto &op : dump_ops_) {
    dump_buf_ << "  ";
    // op outputs
    for (size_t i = 0; i < op.output_num; ++i) {
      if (i != 0) {
        dump_buf_ << ", ";
      }
      dump_buf_ << "%" << output_idx;
      output_idx += 1;
    }
    // op name
    dump_buf_ << " = " << op.name << "(";
    // op inputs
    for (size_t i = 0; i < op.inputs.size(); ++i) {
      if (i != 0) {
        dump_buf_ << ", ";
      }
      auto idx = op.inputs[i].first;
      if (idx < 0) {
        dump_buf_ << op.inputs[i].second;
      } else {
        auto idx_u = static_cast<uint32_t>(idx);
        auto real_idx = idx_u & kIndexFlag;
        dump_buf_ << (static_cast<bool>(idx_u & kLoadFlag) ? "p" : "%") << real_idx;
      }
    }
    dump_buf_ << ")\n";
  }
  dump_buf_ << "  return(";
  output_idx = 0;
  for (size_t i = 0; i < outputs_.size(); ++i) {
    if (!outputs_[i].skip) {
      if (output_idx != 0) {
        dump_buf_ << ", ";
      }
      dump_buf_ << "%" << i;
      output_idx += 1;
    }
  }
  dump_buf_ << ")\n}\n";
  LazyFusionDump::Instance().DumpGraphInfo(&dump_buf_);
}
}  // namespace kernel
}  // namespace mindspore
