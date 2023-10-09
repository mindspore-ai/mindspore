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
#ifndef MINDSPORE_LITE_SRC_EXTENDRT_DELEGATE_ASCEND_NATIVE_DELEGATE_H_
#define MINDSPORE_LITE_SRC_EXTENDRT_DELEGATE_ASCEND_NATIVE_DELEGATE_H_

#include <memory>
#include <vector>
#include <string>
#include "extendrt/delegate/type.h"
#include "src/extendrt/utils/func_graph_utils.h"
#include "extendrt/delegate/ascend_native/ascend_native_base_kernel.h"
#include "extendrt/delegate/ascend_native/sub_graph_helper.h"
#include "extendrt/delegate/ascend_native/delegate_allocator.h"

namespace mindspore {
class AscendNativeDelegate : public ExtendDelegate {
 public:
  static AscendNativeDelegate &Instance() {
    static AscendNativeDelegate instance;
    return instance;
  }
  AscendNativeDelegate() = default;
  virtual ~AscendNativeDelegate() = default;

  void ReplaceNodes(const std::shared_ptr<FuncGraph> &graph) override;

  bool IsDelegateNode(const std::shared_ptr<AnfNode> &node) override;

  std::shared_ptr<kernel::BaseKernel> CreateKernel(const std::shared_ptr<AnfNode> &node) override;
  std::shared_ptr<kernel::BaseKernel> CreateKernel(const kernel::KernelSpec &spec,
                                                   const std::vector<InferTensor *> &inputs,
                                                   const std::vector<InferTensor *> &outputs,
                                                   const InferContext *ctx) const override;

  void set_ascend_native_ctx(std::shared_ptr<kernel::InferContext> ascend_native_ctx) {
    this->ascend_native_ctx_ = ascend_native_ctx;
  }

 private:
  void CreateInputKernelTensors(const CNodePtr &cnode, std::vector<kernel::InferTensor *> *input_tensors,
                                std::shared_ptr<DelegateAllocator> allocator);
  void CreateOutputKernelTensors(const CNodePtr &cnode, std::vector<kernel::InferTensor *> *output_tensors,
                                 std::shared_ptr<DelegateAllocator> allocator);
  bool IsSupport(const CNodePtr &cnode);
  void ReplaceSubGraph(const std::shared_ptr<FuncGraph> &graph, int idx);
  std::vector<KernelWithIndexAndTensor> kernel_list_;
  std::shared_ptr<kernel::InferContext> ascend_native_ctx_ = nullptr;
  void DrawGraph(const std::string &file_name, const std::shared_ptr<FuncGraph> &graph);
  void CopyTensors(InferTensor *t_src, InferTensor *t_dst, const void *stream) const;
  std::shared_ptr<SubGraphHelper> helper_;
};

}  // namespace mindspore

#endif  // MINDSPORE_LITE_SRC_EXTENDRT_DELEGATE_ASCEND_NATIVE_DELEGATE_H_
