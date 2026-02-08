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
#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_KERNEL_DVM_LAZY_FUSION_KERNEL_H
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_KERNEL_DVM_LAZY_FUSION_KERNEL_H

#include <memory>
#include <vector>
#include <string>
#include <unordered_map>
#include <queue>
#include <mutex>
#include <utility>
#include "ir/tensor_new.h"
#include "kernel/ascend/dvm/dvm.h"
#include "mindspore/core/include/ir/tensor.h"
#include "include/pynative/utils/pyboost/op_runner.h"
#include "kernel/ascend/aclnn/pyboost_impl/aclnn_utils.h"
#include "include/pynative/utils/runtime/lazy_fusion.h"
#include "kernel/ascend/dvm/pyboost_impl/lazy_fusion_dump.h"

namespace mindspore {
namespace kernel {
using ShapeRefPtr = std::shared_ptr<dvm::ShapeRef>;
using TensorPtr = tensor::TensorPtr;
using OpRunnerPtr = std::shared_ptr<pyboost::OpRunner>;

class LazyFusionQueue : public runtime::AsyncRQueue {
 public:
  LazyFusionQueue(const string &name, runtime::kThreadWaitLevel waitLevel) : AsyncRQueue(name, waitLevel) {}

  void Push(const runtime::AsyncTaskPtr &task) override;
  void Wait() override;
  bool Empty() override;
  void WorkerJoin() override;
  runtime::kThreadWaitLevel GetCurrentLevel();
};

class LazyFusionKernelAscend;
class LazyFusionManager {
 public:
  LazyFusionManager() = default;
  ~LazyFusionManager();

  LazyFusionKernelAscend *Get(const device::DeviceContext *context, size_t stream);

  void Flush();
  bool Empty() { return current_ == nullptr; }

  void FreeKernel(LazyFusionKernelAscend *k) {
    std::lock_guard<std::mutex> guard(mutex_);
    pool_.push(k);
  }

 private:
  LazyFusionKernelAscend *NewKernel();

  std::queue<LazyFusionKernelAscend *> pool_;
  LazyFusionKernelAscend *current_{nullptr};
  std::mutex mutex_;
  std::atomic<size_t> id_{0};
};

extern LazyFusionManager g_lazy_fusion_manager;

class LazyFusionKernelAscend : public dvm::Kernel {
 public:
  LazyFusionKernelAscend();
  ~LazyFusionKernelAscend();
  void Flush();

  void Reset(const device::DeviceContext *context, size_t stream_id) {
    device_context_ = context;
    stream_id_ = stream_id;
  }
  const device::DeviceContext *device_context() const { return device_context_; }
  size_t stream_id() const { return stream_id_; }
  void set_id(size_t id) { id_ = id; }
  size_t id() const { return id_; }

  dvm::NDObject *Input(const TensorPtr &x, bool enable_cast = true,
                       const std::optional<ShapeVector> &shape = std::nullopt);
  void Output(const TensorPtr &tensor, dvm::NDObject *obj, bool inplace = false);

  TensorPtr Output(dvm::NDObject *obj, TypeId dtype, const ShapeVector &shape) {
    auto tensor = tensor::from_spec(dtype, shape, device::DeviceType::kNone);
    runtime::DeviceAddressUtils::CreateOutputTensorAddress(device_context_, stream_id_, tensor,
                                                           LongToSize(tensor->DataNBytes()));
    Output(tensor, obj);
    return tensor;
  }

  ShapeVector GetShape(dvm::NDObject *obj) {
    auto shape_ref = dvm::Kernel::GetShape(obj);
    return ShapeVector(shape_ref->data, shape_ref->data + shape_ref->size);
  }

  dvm::ShapeRef *GetShapeRef(const ShapeVector &shape);

  dvm::DType TransType(TypeId type) {
    switch (type) {
      case kNumberTypeBool:
        return dvm::DType::kBool;
      case kNumberTypeInt32:
        return dvm::DType::kInt32;
      case kNumberTypeFloat16:
        return dvm::DType::kFloat16;
      case kNumberTypeFloat32:
        return dvm::DType::kFloat32;
      case kNumberTypeBFloat16:
        return dvm::DType::kBFloat16;
      default:
        return dvm::DType::kTypeEnd;
    }
  }

  void *AllocWorkspace(uint64_t size);

  struct Op {
    std::string name;
    std::vector<std::pair<int64_t, std::string>> inputs;
    size_t output_num;
  };

  template <typename T>
  std::pair<bool, uint32_t> GetInputIdx(const T &) {
    return std::make_pair(false, 0u);
  }

  template <typename T>
  std::pair<bool, uint32_t> GetOutputIdx(const T &) {
    return std::make_pair(false, 0u);
  }

  std::pair<bool, uint32_t> GetInputIdx(const TensorPtr &tensor);
  std::pair<bool, uint32_t> GetOutputIdx(const TensorPtr &tensor);

  template <typename T>
  void DumpOpInput(Op *op, const T &t) {
    MS_EXCEPTION_IF_NULL(op);
    auto [found, idx] = GetInputIdx(t);
    if (!found) {
      auto res = GetOutputIdx(t);
      found = res.first;
      idx = res.second;
    }
    if (found) {
      op->inputs.emplace_back(static_cast<int64_t>(idx), "");
    } else {
      op->inputs.emplace_back(-1, LazyFusionDump::Instance().ToString(t));
    }
  }

  template <typename T>
  void DumpOpInput(Op *op, const std::optional<T> &t) {
    MS_EXCEPTION_IF_NULL(op);
    if (!t.has_value()) {
      op->inputs.emplace_back(-1, "None");
    } else {
      DumpOpInput(op, t.value());
    }
  }

  template <typename... Args>
  void DumpOp(const std::string &op_name, const Args &...inputs) {
    auto &op = dump_ops_.emplace_back();
    op.name = op_name;
    op.output_num = outputs_.size() - dump_idx_;
    (DumpOpInput(&op, inputs), ...);
    dump_idx_ = outputs_.size();
  }

  void DumpGraph();

 private:
  void Launch();

  void ClearGraph() {
    for (size_t i = 0; i < input_used_; i++) {
      inputs_[i]->tensor.reset();
    }
    ops_map_.clear();
    input_used_ = 0;
    outputs_.clear();
    reloc_entry_.clear();
    workspace_.clear();
    dump_idx_ = 0;
    dump_ops_.clear();
  }

  void ClearKernel() {
    cached_shape_.clear();
    cross_stream_addrs_.clear();
    EagerClear();
    g_lazy_fusion_manager.FreeKernel(this);
  }

  void Clear() {
    ClearGraph();
    ClearKernel();
  }

  struct Load {
    Load() = default;
    dvm::ShapeRef shape;
    dvm::NDObject *op;
    TensorPtr tensor;
  };

  struct Store {
    Store() = default;
    Store(dvm::NDObject *p, const TensorPtr &t, bool is_skip, bool is_inplace)
        : op(p), tensor(t), skip(is_skip), inplace(is_inplace) {}
    dvm::NDObject *op;
    TensorPtr tensor;
    bool skip{false};
    bool inplace{false};
  };

  std::unordered_map<void *, dvm::NDObject *> ops_map_;
  std::vector<Load *> inputs_;
  std::vector<Store> outputs_;
  std::vector<dvm::RelocEntry> reloc_entry_;
  std::vector<kernel::pyboost::MemBlockPtr> workspace_;
  std::vector<std::pair<uint32_t, void *>> cross_stream_addrs_;
  std::vector<std::pair<ShapeVector, ShapeRefPtr>> cached_shape_;
  size_t input_used_{0};
  std::stringstream dump_buf_;
  const device::DeviceContext *device_context_;
  size_t stream_id_;
  size_t id_{0};
  size_t dump_idx_{0};
  std::vector<Op> dump_ops_;
};

static inline void FlushLazyFusion() { g_lazy_fusion_manager.Flush(); }
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_KERNEL_DVM_LAZY_FUSION_KERNEL_H
