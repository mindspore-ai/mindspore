/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_BACKEND_OPTIMIZER_ASCEND_FORMAT_TYPE_RECTIFY_DO_MASK_KERNEL_INFO_H
#define MINDSPORE_CCSRC_BACKEND_OPTIMIZER_ASCEND_FORMAT_TYPE_RECTIFY_DO_MASK_KERNEL_INFO_H
#include <map>
#include <string>
#include <memory>

#include "include/backend/optimizer/optimizer.h"
#include "plugin/device/ascend/optimizer/ascend_helper.h"

namespace mindspore {
namespace opt {
class RectifyDoMaskKernelInfo : public PatternProcessPass {
 public:
  explicit RectifyDoMaskKernelInfo(bool multigraph = true)
      : PatternProcessPass("rectify_do_mask_kernel_info", multigraph),
        kernel_selecter(std::make_shared<KernelSelect>()) {}
  ~RectifyDoMaskKernelInfo() override = default;

  const BaseRef DefinePattern() const override;
  const AnfNodePtr Process(const FuncGraphPtr &graph, const AnfNodePtr &node, const EquivPtr &) const override;

 private:
  void ReSelectChildNodeKernelInfo(const CNodePtr &cnode, const FuncGraphPtr &graph) const;
  KernelSelectPtr kernel_selecter;
};
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_OPTIMIZER_ASCEND_FORMAT_TYPE_RECTIFY_DO_MASK_KERNEL_INFO_H
