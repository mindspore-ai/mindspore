/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_LITE_SRC_SUB_GRAPH_H
#define MINDSPORE_LITE_SRC_SUB_GRAPH_H

#include <atomic>
#include <utility>
#include <string>
#include <vector>
#include <map>
#include "src/lite_kernel.h"
#include "src/executor.h"
#include "src/common/log_adapter.h"
#ifdef ENABLE_ARM64
#include "src/common/utils.h"
#endif

namespace mindspore::kernel {
// store origin data and allocator of input tensor of subgraph for PreProcess and PostProcess
struct DataStore {
  void *data_ = nullptr;
  mindspore::Allocator *allocator_ = nullptr;
  static DataStore *CreateDataStore(void *data = nullptr, mindspore::Allocator *data_allocator = nullptr,
                                    mindspore::Allocator *allocator = nullptr) {
    DataStore *data_store = nullptr;
    if (allocator == nullptr) {
      data_store = static_cast<DataStore *>(malloc(sizeof(DataStore)));
    } else {
      data_store = static_cast<DataStore *>(allocator->Malloc(sizeof(DataStore)));
    }
    if (data_store == nullptr) {
      MS_LOG(ERROR) << "Malloc data_store failed";
      return nullptr;
    }
    data_store->data_ = data;
    data_store->allocator_ = data_allocator;
    return data_store;
  }
};

class SubGraphKernel : public LiteKernel {
 public:
  SubGraphKernel(const std::vector<lite::Tensor *> &inputs, const std::vector<lite::Tensor *> &outputs,
                 std::vector<LiteKernel *> in_kernels, std::vector<LiteKernel *> out_kernels,
                 std::vector<LiteKernel *> nodes, const lite::InnerContext *ctx)
      : LiteKernel(nullptr, inputs, outputs, ctx),
        nodes_(std::move(nodes)),
        in_nodes_(std::move(in_kernels)),
        out_nodes_(std::move(out_kernels)) {
    subgraph_type_ = kCpuFP32SubGraph;
  }

  ~SubGraphKernel() override {
    for (auto *node : nodes_) {
      delete node;
    }
    nodes_.clear();
  }

  void FindInoutKernels(const std::vector<kernel::LiteKernel *> &scope_kernels) override {
    LiteKernel::FindInoutKernels(scope_kernels);
    std::vector<kernel::LiteKernel *> new_scope_kernels = {};
    new_scope_kernels.insert(new_scope_kernels.end(), this->in_kernels().begin(), this->in_kernels().end());
    new_scope_kernels.insert(new_scope_kernels.end(), this->out_kernels().begin(), this->out_kernels().end());
    new_scope_kernels.insert(new_scope_kernels.end(), nodes_.begin(), nodes_.end());
    for (auto *node : nodes_) {
      node->FindInoutKernels(new_scope_kernels);
    }
  }

  bool IsReady(const std::vector<lite::Tensor *> &scope_tensors) override {
    return std::all_of(this->in_nodes_.begin(), this->in_nodes_.end(),
                       [&](LiteKernel *kernel) { return kernel->IsReady(scope_tensors); });
  }

  // called while compiling graph. Call node->Prepare() by default.
  int Prepare() override;
  // called before Run
  int PreProcess() override { return mindspore::lite::RET_OK; }

  int Run() override;

  int Run(const KernelCallBack &before, const KernelCallBack &after) override;
  // called after Run
  int PostProcess() override { return mindspore::lite::RET_OK; }

  int ReSize() override;

  int ReSize(bool is_interrupt);

  void InitOutTensorInitRefCount() override;

  int Init() override { return mindspore::lite::RET_OK; }

  std::string ToString() const override;

  std::vector<LiteKernel *> nodes() { return this->nodes_; }

 protected:
  std::vector<LiteKernel *> nodes_{};
  // entry nodes in nodes
  std::vector<LiteKernel *> in_nodes_{};
  // exit nodes in nodes
  std::vector<LiteKernel *> out_nodes_{};
  mindspore::lite::Executor *executor_ = nullptr;
};

class CpuSubGraph : public SubGraphKernel {
 public:
  CpuSubGraph(const std::vector<lite::Tensor *> &inputs, const std::vector<lite::Tensor *> &outputs,
              std::vector<LiteKernel *> in_kernels, std::vector<LiteKernel *> out_kernels,
              std::vector<LiteKernel *> nodes, const lite::InnerContext *ctx)
      : SubGraphKernel(inputs, outputs, std::move(in_kernels), std::move(out_kernels), std::move(nodes), ctx) {
    subgraph_type_ = kCpuFP32SubGraph;
  }

  ~CpuSubGraph() override { delete this->executor_; }
  int Prepare() override;
  int Init() override { return SubGraphKernel::Init(); }
  int PreProcess() override { return SubGraphKernel::PreProcess(); }
  int Run() override { return SubGraphKernel::Run(); }
  int Run(const KernelCallBack &before, const KernelCallBack &after) override {
    return SubGraphKernel::Run(before, after);
  };
  int PostProcess() override { return SubGraphKernel::PostProcess(); }
};

class CpuFp32SubGraph : public CpuSubGraph {
 public:
  CpuFp32SubGraph(const std::vector<lite::Tensor *> &inputs, const std::vector<lite::Tensor *> &outputs,
                  std::vector<LiteKernel *> in_kernels, std::vector<LiteKernel *> out_kernels,
                  std::vector<LiteKernel *> nodes, const lite::InnerContext *ctx)
      : CpuSubGraph(inputs, outputs, std::move(in_kernels), std::move(out_kernels), std::move(nodes), ctx) {
    subgraph_type_ = kCpuFP32SubGraph;
    static std::atomic_int index = 0;
    this->name_ = "CpuFP32SubGraph" + std::to_string(index++);
  }

  ~CpuFp32SubGraph() override = default;
  int Init() override { return CpuSubGraph::Init(); }
  int PreProcess() override { return CpuSubGraph::PreProcess(); }
  int Run() override { return CpuSubGraph::Run(); }
  int Run(const KernelCallBack &before, const KernelCallBack &after) override {
    return CpuSubGraph::Run(before, after);
  };
  int PostProcess() override { return CpuSubGraph::PostProcess(); }
};

#ifdef ENABLE_FP16
class CpuFp16SubGraph : public CpuSubGraph {
 public:
  CpuFp16SubGraph(const std::vector<lite::Tensor *> &inputs, const std::vector<lite::Tensor *> &outputs,
                  std::vector<LiteKernel *> in_kernels, std::vector<LiteKernel *> out_kernels,
                  std::vector<LiteKernel *> nodes, const lite::InnerContext *ctx)
      : CpuSubGraph(inputs, outputs, std::move(in_kernels), std::move(out_kernels), std::move(nodes), ctx) {
    subgraph_type_ = kCpuFP16SubGraph;
    static std::atomic_int index = 0;
    this->name_ = "CpuFP16SubGraph" + std::to_string(index++);
  }

  ~CpuFp16SubGraph() override = default;
  int Init() override { return CpuSubGraph::Init(); }
  int PreProcess() override;
  int Run() override { return CpuSubGraph::Run(); }
  int Run(const KernelCallBack &before, const KernelCallBack &after) override {
    return CpuSubGraph::Run(before, after);
  };
  int PostProcess() override;

 private:
  void FreeOriginInputData();
  int Float32TensorToFloat16Tensor(lite::Tensor *tensor);
  int Float16TensorToFloat32Tensor(lite::Tensor *tensor);

 private:
  std::map<lite::Tensor *, DataStore *> origin_input_data_;
};
#endif
}  // namespace mindspore::kernel
#endif  // MINDSPORE_LITE_SRC_SUB_GRAPH_H
