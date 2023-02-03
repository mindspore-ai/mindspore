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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_SUB_GRAPH_KERNEL_H_
#define MINDSPORE_LITE_SRC_RUNTIME_SUB_GRAPH_KERNEL_H_

#include <atomic>
#include <utility>
#include <string>
#include <vector>
#include <map>
#include <memory>
#include "src/litert/kernel_exec.h"
#include "src/litert/executor.h"
#include "src/common/log_adapter.h"
#include "src/common/version_manager.h"
#include "src/litert/cpu_info.h"
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

typedef struct KernelsArray {
  struct KernelsArrayUnit {
    std::vector<KernelExec *> kernels = {};
    std::vector<size_t> input_indexs = {};
    std::vector<size_t> output_indexs = {};
  };
  std::vector<KernelsArrayUnit> units = {};
  std::vector<size_t> graph_input = {};
} KernelsArray;

class SubGraphKernel : public KernelExec {
 public:
  SubGraphKernel(std::vector<KernelExec *> in_kernels, std::vector<KernelExec *> out_kernels,
                 std::vector<KernelExec *> nodes, Kernel *kernel)
      : KernelExec(std::shared_ptr<Kernel>(kernel)),
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
                       [&](KernelExec *kernel) { return kernel->IsReady(scope_tensors); });
  }

  // called while compiling graph. Call node->Prepare() by default.
  int Prepare() override { return RET_OK; };
  // called before Run
  int Execute() override { return Execute(nullptr, nullptr); }

  int Execute(const KernelCallBack &before, const KernelCallBack &after) override;

  int ReSize() override;

  virtual int MallocNodesOutputSpace();

  virtual int MallocSubgraphInputs();

  void InitOutTensorInitRefCount(const std::vector<KernelExec *> *mask_kernels) override;

  void InitInputTensorInitRefCount();

  virtual int SetFp16Attr() { return mindspore::lite::RET_OK; }

  std::string ToString() const override;

  void set_nodes(const std::vector<KernelExec *> &node) { this->nodes_ = node; }

  std::vector<KernelExec *> &nodes() { return this->nodes_; }

  void DropNode(KernelExec *node);

  std::vector<KernelExec *> in_nodes() { return this->in_nodes_; }

  std::vector<KernelExec *> out_nodes() { return this->out_nodes_; }

  void SetInNodes(const std::vector<KernelExec *> &in_nodes) { in_nodes_ = in_nodes; }

  void SetOutNodes(const std::vector<KernelExec *> &out_nodes) { out_nodes_ = out_nodes; }

  void SetSchemaVersion(int schema_version) { schema_version_ = schema_version; }

  int TopologicalSortNodes();

  void InsertInEdge(KernelExec *kernel, KernelExec *replace_kernel, const size_t &tensor_index);

  void InsertOutEdge(KernelExec *kernel, KernelExec *replace_kernel, const size_t &tensor_index);

  void UpdateInOutKernels(KernelExec *in_kernel, std::vector<KernelExec *> out_kernels, KernelExec *in_post_kernel,
                          KernelExec *out_pre_kernel);

  int UpdateInOutTensors(KernelExec *in_kernel, std::vector<KernelExec *> out_kernels, lite::Tensor *in_tensor,
                         lite::Tensor *out_tensor, bool keep_input);

  int DeleteSingleWayNode(KernelExec *kernel, bool keep_input);

  inline bool GetGraphChanged() const { return graph_changed_; }

  inline void SetGraphChanged(bool flag) { graph_changed_ = flag; }

  int SubGraphSplitByOperator(KernelsArray *out_kernels);

 protected:
  std::vector<KernelExec *> nodes_{};
  // entry nodes in nodes
  std::vector<KernelExec *> in_nodes_{};
  // exit nodes in nodes
  std::vector<KernelExec *> out_nodes_{};
  mindspore::lite::Executor *executor_ = nullptr;
  int schema_version_ = lite::SCHEMA_VERSION::SCHEMA_CUR;
  bool graph_changed_ = false;
};

class CpuSubGraph : public SubGraphKernel {
 public:
  CpuSubGraph(std::vector<KernelExec *> in_kernels, std::vector<KernelExec *> out_kernels,
              std::vector<KernelExec *> nodes, Kernel *kernel)
      : SubGraphKernel(std::move(in_kernels), std::move(out_kernels), std::move(nodes), kernel) {
    subgraph_type_ = kCpuFP32SubGraph;
    desc_.arch = kernel::KERNEL_ARCH::kCPU;
  }

  ~CpuSubGraph() override { delete this->executor_; }
  int Prepare() override;
  int SetFp16Attr() override { return SubGraphKernel::SetFp16Attr(); }
  int Execute() override { return Execute(nullptr, nullptr); }
  int Execute(const KernelCallBack &before, const KernelCallBack &after) override;
};

class CpuFp32SubGraph : public CpuSubGraph {
 public:
  CpuFp32SubGraph(std::vector<KernelExec *> in_kernels, std::vector<KernelExec *> out_kernels,
                  std::vector<KernelExec *> nodes, Kernel *kernel)
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
  CpuFp16SubGraph(std::vector<KernelExec *> in_kernels, std::vector<KernelExec *> out_kernels,
                  std::vector<KernelExec *> nodes, Kernel *kernel)
      : CpuSubGraph(std::move(in_kernels), std::move(out_kernels), std::move(nodes), kernel) {
    subgraph_type_ = kCpuFP16SubGraph;
    static std::atomic_int index = 0;
    this->set_name("CpuFP16SubGraph" + std::to_string(index++));
    desc_.data_type = kNumberTypeFloat16;
  }

  ~CpuFp16SubGraph() override = default;
  int SetFp16Attr() override {
    const auto *context = this->Context();
    MS_ASSERT(context != nullptr);
    support_fp16_ = context->device_and_pkg_support_fp16_;
    return CpuSubGraph::SetFp16Attr();
  }

  int Prepare() override {
    auto ret = CpuSubGraph::Prepare();
    if (ret != RET_OK) {
      return ret;
    }
    for (auto &node : this->nodes_) {
      if (node->type() == schema::PrimitiveType_Cast) {
        auto inputs = node->in_tensors();
#define kCaseInputNum 2
        MS_ASSERT(inputs.size() >= 2);
#undef kCaseInputNum
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
  CustomSubGraph(std::vector<KernelExec *> in_kernels, std::vector<KernelExec *> out_kernels,
                 std::vector<KernelExec *> nodes, Kernel *kernel)
      : SubGraphKernel(std::move(in_kernels), std::move(out_kernels), std::move(nodes), kernel) {
    subgraph_type_ = kCustomSubGraph;
    desc_.arch = kernel::KERNEL_ARCH::kCustom;
  }

  ~CustomSubGraph() override { delete this->executor_; }
  int Prepare() override;
  int Execute() override { return Execute(nullptr, nullptr); }
  int Execute(const KernelCallBack &before, const KernelCallBack &after) override;
};
}  // namespace mindspore::kernel
#endif  // MINDSPORE_LITE_SRC_RUNTIME_SUB_GRAPH_KERNEL_H_
