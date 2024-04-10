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
#include "src/common/config_infos.h"
#include "src/common/common.h"

namespace mindspore {
class AscendNativeDelegate : public ExtendDelegate {
 public:
  static AscendNativeDelegate &Instance(const std::shared_ptr<Context> &ctx, const ConfigInfos &config_infos) {
    static AscendNativeDelegate instance(ctx, config_infos);
    return instance;
  }
  AscendNativeDelegate(const std::shared_ptr<Context> &ctx, const ConfigInfos &config_infos)
      : context_(ctx), config_infos_(config_infos) {}
  virtual ~AscendNativeDelegate() = default;

  void ReplaceNodes(const std::shared_ptr<FuncGraph> &graph) override;

  bool IsDelegateNode(const std::shared_ptr<AnfNode> &node) override;

  std::shared_ptr<kernel::BaseKernel> CreateKernel(const kernel::KernelSpec &spec,
                                                   const std::vector<InferTensor *> &inputs,
                                                   const std::vector<InferTensor *> &outputs,
                                                   const InferContext *ctx) const override;

  void set_ascend_native_ctx(std::shared_ptr<kernel::InferContext> ascend_native_ctx) {
    this->ascend_native_ctx_ = ascend_native_ctx;
  }

  static bool init_delegate_;

 private:
  void CreateInputKernelTensors(const CNodePtr &cnode, std::vector<kernel::InferTensor *> *input_tensors,
                                std::shared_ptr<DelegateAllocator> allocator);
  void CreateOutputKernelTensors(const CNodePtr &cnode, std::vector<kernel::InferTensor *> *output_tensors,
                                 std::shared_ptr<DelegateAllocator> allocator);
  int ParseTransformerProfile();
  int ParseMaskTensorName();
  bool IsSupport(const CNodePtr &cnode);
  void ReplaceSubGraph(const std::shared_ptr<FuncGraph> &graph, int idx);
  std::vector<KernelWithIndexAndTensor> kernel_list_;
  std::shared_ptr<kernel::InferContext> ascend_native_ctx_ = nullptr;
  void DrawGraph(const std::string &file_name, const std::shared_ptr<FuncGraph> &graph);
  void CopyTensors(InferTensor *t_src, InferTensor *t_dst, const void *stream, const void *acl_ctx) const;
  void Init();
  int AddVsl(const std::shared_ptr<FuncGraph> &graph);
  std::shared_ptr<SubGraphHelper> helper_;
  mutable void *stream_;
  mutable void *acl_ctx_;
  std::shared_ptr<mindspore::Context> context_{nullptr};
  ConfigInfos config_infos_;
  bool pangu_sigma_{false};
  int out_num_vsl_{6};
  std::string mask_tensor_name_ = "";
};

}  // namespace mindspore

#endif  // MINDSPORE_LITE_SRC_EXTENDRT_DELEGATE_ASCEND_NATIVE_DELEGATE_H_
