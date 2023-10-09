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

#include "extendrt/delegate/ascend_native/ascend_native_base_kernel.h"
#include <string>
#include <set>
#include <memory>
#include <vector>
#include <unordered_map>
#include "infer/context.h"

namespace mindspore::kernel {
class AscendNativeCompositeKernel : public AscendNativeBaseKernel {
 public:
  // AscendNativeCompositeKernel = delete;

  AscendNativeCompositeKernel(const std::vector<InferTensor *> &inputs, const std::vector<InferTensor *> &outputs,
                              InferPrimitive prim, const InferContext *ctx, const void *stream, std::string name)
      : AscendNativeBaseKernel(inputs, outputs, prim, ctx, stream, name) {}

  int Prepare() override;
  int Run() override;
  int PostProcess() override;
  int PreProcess() override;
  int InferShape() override;
  int ReSize() override;

  bool IsWeightInputHanledInner() const override { return true; }

  void set_func_graph(const FuncGraphPtr &func_graph) { func_graph_ = func_graph; }

 private:
  std::shared_ptr<kernel::AscendNativeBaseKernel> CreateKernel(const AnfNodePtr &node);
  void CreateInputKernelTensors(const CNodePtr &cnode, std::vector<kernel::InferTensor *> *input_tensors);
  void CreateOutputKernelTensors(const CNodePtr &cnode, std::vector<kernel::InferTensor *> *output_tensors);
  int GetIdxFromString(std::string str);
  int AllocateGraphTensors();
  int AllocTensors();
  void InitializeTensorRefrenceCnt();
  void FreeDevice();
  int ReAllocTensors();
  int AllocateGraphWorkspace(size_t size);
  size_t get_workspace_size() const override { return ws_size_; }
  void set_workspace_size(size_t size) { ws_size_ = size; }
  FuncGraphPtr func_graph_;
  std::vector<KernelWithIndexAndTensor> kernel_list_;
  std::vector<std::shared_ptr<kernel::AscendNativeBaseKernel>> kernels_;
  void *device_memory_base_addr_ = nullptr;
  size_t device_mem_size_ = 0;
  std::set<kernel::InferTensor *> allocated_tensors_;
  std::unordered_map<kernel::InferTensor *, size_t> offset_map_;
  size_t ws_size_{0};

  static constexpr size_t max_ws_size_ = static_cast<size_t>(2100) * (1 << 20);
};
}  // namespace mindspore::kernel
#endif  // MINDSPORE_LITE_SRC_EXTENDRT_KERNEL_ASCEND_NATIVE_COMPOSITE_KERNEL_H_
