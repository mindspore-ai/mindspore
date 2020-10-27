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

#include <utility>
#include <string>
#include <vector>
#include "src/lite_kernel.h"
#include "src/executor.h"

namespace mindspore::kernel {
class SubGraphKernel : public LiteKernel {
 public:
  explicit SubGraphKernel(const std::vector<lite::Tensor *> &inputs, const std::vector<lite::Tensor *> &outputs,
                          const std::vector<LiteKernel *> &in_kernels, const std::vector<LiteKernel *> &out_kernels,
                          std::vector<LiteKernel *> nodes, const lite::InnerContext *ctx)
      : LiteKernel(nullptr, inputs, outputs, ctx, nullptr), nodes_(std::move(nodes)) {
    in_kernels_ = in_kernels;
    out_kernels_ = out_kernels;
    subgraph_type_ = kCpuFP32SubGraph;
  }

  ~SubGraphKernel() override {
    for (auto *node : nodes_) {
      delete node;
    }
    nodes_.clear();
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

  std::string ToString() const override;

  std::vector<LiteKernel *> nodes() { return this->nodes_; }

 protected:
  std::vector<LiteKernel *> nodes_;
  mindspore::lite::Executor *executor_ = nullptr;
};

class CpuSubGraph : public SubGraphKernel {
 public:
  explicit CpuSubGraph(const std::vector<lite::Tensor *> &inputs, const std::vector<lite::Tensor *> &outputs,
                       const std::vector<LiteKernel *> &in_kernels, const std::vector<LiteKernel *> &out_kernels,
                       const std::vector<LiteKernel *> &nodes, const lite::InnerContext *ctx)
      : SubGraphKernel(inputs, outputs, in_kernels, out_kernels, nodes, ctx) {
    subgraph_type_ = kCpuFP32SubGraph;
    this->executor_ = new mindspore::lite::Executor;
  }

  ~CpuSubGraph() override { delete this->executor_; }

  int Prepare() override;
  int Init() override { return SubGraphKernel::Init(); }
  int PreProcess() override { return SubGraphKernel::PreProcess(); }
  int Run() override { return SubGraphKernel::Run(); }
  int Run(const KernelCallBack &before, const KernelCallBack &after) override {
    return SubGraphKernel::Run(before, after);
  };
  int PostProcess() override { return mindspore::lite::RET_OK; }
};

class CpuFp32SubGraph : public CpuSubGraph {
 public:
  explicit CpuFp32SubGraph(const std::vector<lite::Tensor *> &inputs, const std::vector<lite::Tensor *> &outputs,
                           const std::vector<LiteKernel *> &in_kernels, const std::vector<LiteKernel *> &out_kernels,
                           const std::vector<LiteKernel *> &nodes, const lite::InnerContext *ctx)
      : CpuSubGraph(inputs, outputs, in_kernels, out_kernels, nodes, ctx) {
    subgraph_type_ = kCpuFP32SubGraph;
    this->name_ = "CpuFP32SubGraph";
  }

  ~CpuFp32SubGraph() override = default;
  int Init() override { return mindspore::lite::RET_ERROR; }
  int PreProcess() override;
  int Run() override { return CpuSubGraph::Run(); }
  int Run(const KernelCallBack &before, const KernelCallBack &after) override {
    return CpuSubGraph::Run(before, after);
  };
  int PostProcess() override { return mindspore::lite::RET_OK; }
};

class CpuFp16SubGraph : public CpuSubGraph {
 public:
  explicit CpuFp16SubGraph(const std::vector<lite::Tensor *> &inputs, const std::vector<lite::Tensor *> &outputs,
                           const std::vector<LiteKernel *> &in_kernels, const std::vector<LiteKernel *> &out_kernels,
                           const std::vector<LiteKernel *> &nodes, const lite::InnerContext *ctx)
      : CpuSubGraph(inputs, outputs, in_kernels, out_kernels, nodes, ctx) {
    subgraph_type_ = kCpuFP16SubGraph;
    this->name_ = "CpuFP16SubGraph";
  }

  ~CpuFp16SubGraph() override = default;
  int Init() override { return mindspore::lite::RET_ERROR; }
  int PreProcess() override;
  int Run() override { return CpuSubGraph::Run(); }
  int Run(const KernelCallBack &before, const KernelCallBack &after) override {
    return CpuSubGraph::Run(before, after);
  };
  int PostProcess() override;
};
}  // namespace mindspore::kernel
#endif  // MINDSPORE_LITE_SRC_SUB_GRAPH_H
