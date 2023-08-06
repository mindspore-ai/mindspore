/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_LITE_SRC_EXTENDRT_KERNEL_ASCEND_NATIVE_COMPOSITE_KERNEL_H_
#define MINDSPORE_LITE_SRC_EXTENDRT_KERNEL_ASCEND_NATIVE_COMPOSITE_KERNEL_H_

#include <string>
#include <set>
#include <memory>
#include <vector>
#include "extendrt/kernel/ascend_native/ascend_native_base_kernel.h"
#include "infer/context.h"

namespace mindspore::kernel {
class AscendNativeCompositeKernel : public AscendNativeBaseKernel {
 public:
  // AscendNativeCompositeKernel = delete;

  AscendNativeCompositeKernel(const std::vector<InferTensor *> &inputs, const std::vector<InferTensor *> &outputs,
                              InferPrimitive prim, std::shared_ptr<kernel::InferContext> *ctx, const void *stream,
                              std::string name)
      : AscendNativeBaseKernel(inputs, outputs, prim, ctx, stream, name) {}

  int Prepare() override;
  int Execute() override;

  bool IsWeightInputHanledInner() const override { return true; }

  void set_func_graph(const FuncGraphPtr &func_graph) { func_graph_ = func_graph; }

 private:
  std::shared_ptr<kernel::BaseKernel> CreateKernel(const AnfNodePtr &node);
  void CreateInputKernelTensors(const CNodePtr &cnode, std::vector<kernel::InferTensor *> *input_tensors);
  void CreateOutputKernelTensors(const CNodePtr &cnode, std::vector<kernel::InferTensor *> *output_tensors);
  Status FindGraphInputs(const std::vector<AnfNodePtr> &node_list, const std::vector<AnfNodePtr> &graph_inputs,
                         const std::vector<std::shared_ptr<kernel::BaseKernel>> &kernels);
  Status FindGraphOutputs(const std::vector<AnfNodePtr> &node_list, const AnfNodePtr &graph_output,
                          const std::vector<std::shared_ptr<kernel::BaseKernel>> &kernels);
  Status AllocateGraphTensors();
  Status AllocTensors();
  void InitializeTensorRefrenceCnt();
  void FreeDevice(void *ptr);

  FuncGraphPtr func_graph_;
  std::vector<KernelWithIndexAndTensor> kernel_list_;
  std::vector<std::shared_ptr<kernel::BaseKernel>> kernels_;
  void *device_memory_base_addr_ = nullptr;
  size_t device_mem_size_ = 0;
  std::set<kernel::InferTensor *> allocated_tensors_;
};
}  // namespace mindspore::kernel
#endif  // MINDSPORE_LITE_SRC_EXTENDRT_KERNEL_ASCEND_NATIVE_COMPOSITE_KERNEL_H_
