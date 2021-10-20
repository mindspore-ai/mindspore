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

#ifndef MINDSPORE_LITE_SRC_SUB_GRAPH_KERNEL_H_
#define MINDSPORE_LITE_SRC_SUB_GRAPH_KERNEL_H_

#include <atomic>
#include <utility>
#include <string>
#include <vector>
#include <map>
#include <memory>
#include "src/lite_kernel.h"
#include "src/executor.h"
#include "src/common/log_adapter.h"
#include "src/common/version_manager.h"
#include "src/cpu_info.h"
#if defined(ENABLE_ARM) && defined(ENABLE_FP16)
#include "nnacl/constant_of_shape_parameter.h"
#endif

namespace mindspore::kernel {
// store origin data and allocator of input tensor of subgraph for PreProcess and PostProcess
struct DataStore {
  void *data_ = nullptr;
  Allocator *allocator_ = nullptr;
  bool own_data_ = true;
  static DataStore *CreateDataStore(void *data = nullptr, bool own_data = true, Allocator *data_allocator = nullptr,
                                    Allocator *allocator = nullptr) {
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
    data_store->own_data_ = own_data;
    data_store->allocator_ = data_allocator;
    return data_store;
  }
};

class SubGraphKernel : public LiteKernel {
 public:
  SubGraphKernel(std::vector<LiteKernel *> in_kernels, std::vector<LiteKernel *> out_kernels,
                 std::vector<LiteKernel *> nodes, Kernel *kernel)
      : LiteKernel(std::shared_ptr<Kernel>(kernel)),
        nodes_(std::move(nodes)),
        in_nodes_(std::move(in_kernels)),
        out_nodes_(std::move(out_kernels)) {
    subgraph_type_ = kCpuFP32SubGraph;
    desc_.data_type = kNumberTypeFloat32;
  }

  ~SubGraphKernel() override {
    for (auto *node : nodes_) {
      delete node;
    }
    nodes_.clear();
  }

  bool IsReady(const std::vector<lite::Tensor *> &scope_tensors) override {
    return std::all_of(this->in_nodes_.begin(), this->in_nodes_.end(),
                       [&](LiteKernel *kernel) { return kernel->IsReady(scope_tensors); });
  }

  // called while compiling graph. Call node->Prepare() by default.
  int Prepare() override;
  // called before Run
  int Execute() override { return Execute(nullptr, nullptr); }

  int Execute(const KernelCallBack &before, const KernelCallBack &after) override;

  // called after Run
  int ReSize() override;

  void InitOutTensorInitRefCount(const std::vector<LiteKernel *> *mask_kernels) override;

  void InitInputTensorInitRefCount();

  int Init() override { return mindspore::lite::RET_OK; }

  std::string ToString() const override;

  std::vector<LiteKernel *> &nodes() { return this->nodes_; }

  void DropNode(LiteKernel *node);

  std::vector<LiteKernel *> in_nodes() { return this->in_nodes_; }

  std::vector<LiteKernel *> out_nodes() { return this->out_nodes_; }

  void SetSchemaVersion(int schema_version) { schema_version_ = schema_version; }

 protected:
  std::vector<LiteKernel *> nodes_{};
  // entry nodes in nodes
  std::vector<LiteKernel *> in_nodes_{};
  // exit nodes in nodes
  std::vector<LiteKernel *> out_nodes_{};
  mindspore::lite::Executor *executor_ = nullptr;
  int schema_version_ = lite::SCHEMA_VERSION::SCHEMA_CUR;
};

class CpuSubGraph : public SubGraphKernel {
 public:
  CpuSubGraph(std::vector<LiteKernel *> in_kernels, std::vector<LiteKernel *> out_kernels,
              std::vector<LiteKernel *> nodes, Kernel *kernel)
      : SubGraphKernel(std::move(in_kernels), std::move(out_kernels), std::move(nodes), kernel) {
    subgraph_type_ = kCpuFP32SubGraph;
    desc_.arch = kernel::KERNEL_ARCH::kCPU;
  }

  ~CpuSubGraph() override { delete this->executor_; }
  int Prepare() override;
  int Init() override { return SubGraphKernel::Init(); }
  int Execute() override { return Execute(nullptr, nullptr); }
  int Execute(const KernelCallBack &before, const KernelCallBack &after) override;
};

class CpuFp32SubGraph : public CpuSubGraph {
 public:
  CpuFp32SubGraph(std::vector<LiteKernel *> in_kernels, std::vector<LiteKernel *> out_kernels,
                  std::vector<LiteKernel *> nodes, Kernel *kernel)
      : CpuSubGraph(std::move(in_kernels), std::move(out_kernels), std::move(nodes), kernel) {
    subgraph_type_ = kCpuFP32SubGraph;
    static std::atomic_int index = {0};
    this->set_name("CpuFP32SubGraph" + std::to_string(index++));
    desc_.data_type = kNumberTypeFloat32;
  }
  ~CpuFp32SubGraph() override = default;
};

#if defined(ENABLE_ARM) && defined(ENABLE_FP16)
class CpuFp16SubGraph : public CpuSubGraph {
 public:
  CpuFp16SubGraph(std::vector<LiteKernel *> in_kernels, std::vector<LiteKernel *> out_kernels,
                  std::vector<LiteKernel *> nodes, Kernel *kernel)
      : CpuSubGraph(std::move(in_kernels), std::move(out_kernels), std::move(nodes), kernel) {
    subgraph_type_ = kCpuFP16SubGraph;
    static std::atomic_int index = 0;
    this->set_name("CpuFP16SubGraph" + std::to_string(index++));
    desc_.data_type = kNumberTypeFloat16;
  }

  ~CpuFp16SubGraph() override = default;
  int Init() override {
    const auto *context = this->Context();
    MS_ASSERT(context != nullptr);
    support_fp16_ = context->device_and_pkg_support_fp16();
    return CpuSubGraph::Init();
  }

  int Prepare() override {
    auto ret = CpuSubGraph::Prepare();
    if (ret != RET_OK) {
      return ret;
    }
    for (auto &node : this->nodes_) {
      if (node->type() == schema::PrimitiveType_Cast) {
        auto inputs = node->in_tensors();
        MS_ASSERT(inputs.size() >= 2);
        auto dst_tensor = inputs[1];
        MS_ASSERT(dst_tensor != nullptr);
        MS_ASSERT(dst_tensor->data_type() == kNumberTypeInt32);
        MS_ASSERT(dst_tensor->data() != nullptr);
        MS_ASSERT(dst_tensor->ElementsNum() == 1);
        auto *dst_data = reinterpret_cast<int32_t *>(dst_tensor->data());
        if (dst_data[0] == kNumberTypeFloat32) {
          dst_data[0] = kNumberTypeFloat16;
        }
        auto outputs = node->out_tensors();
        MS_ASSERT(outputs.size() == 1);
        auto output = outputs.front();
        MS_ASSERT(output != nullptr);
        if (output->data_type() == kNumberTypeFloat32) {
          output->set_data_type(kNumberTypeFloat16);
        }
      } else if (node->type() == schema::PrimitiveType_ConstantOfShape) {
        auto param = node->op_parameter();
        MS_ASSERT(param != nullptr);
        if (static_cast<TypeId>(reinterpret_cast<ConstantOfShapeParameter *>(param)->data_type_ ==
                                kNumberTypeFloat32)) {
          reinterpret_cast<ConstantOfShapeParameter *>(param)->data_type_ = kNumberTypeFloat16;
        }
        auto outputs = node->out_tensors();
        MS_ASSERT(outputs.size() == 1);
        auto output = outputs.front();
        MS_ASSERT(output != nullptr);
        if (output->data_type() == kNumberTypeFloat32) {
          output->set_data_type(kNumberTypeFloat16);
        }
      }
    }
    return RET_OK;
  }

 private:
  bool support_fp16_ = false;
};
#endif

class CustomSubGraph : public SubGraphKernel {
 public:
  CustomSubGraph(std::vector<LiteKernel *> in_kernels, std::vector<LiteKernel *> out_kernels,
                 std::vector<LiteKernel *> nodes, Kernel *kernel)
      : SubGraphKernel(std::move(in_kernels), std::move(out_kernels), std::move(nodes), kernel) {
    subgraph_type_ = kCustomSubGraph;
    desc_.arch = kernel::KERNEL_ARCH::kCustom;
  }

  ~CustomSubGraph() override { delete this->executor_; }
  int Prepare() override;
  int Init() override { return SubGraphKernel::Init(); }
  int Execute() override { return Execute(nullptr, nullptr); }
  int Execute(const KernelCallBack &before, const KernelCallBack &after) override;
};
}  // namespace mindspore::kernel
#endif  // MINDSPORE_LITE_SRC_SUB_GRAPH_KERNEL_H_
