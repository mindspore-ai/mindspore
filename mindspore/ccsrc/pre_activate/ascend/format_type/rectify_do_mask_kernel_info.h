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

#ifndef MINDSPORE_CCSRC_PRE_ACTIVATE_ASCEND_FORMAT_TYPE_RECTIFY_DO_MASK_KERNEL_INFO_H
#define MINDSPORE_CCSRC_PRE_ACTIVATE_ASCEND_FORMAT_TYPE_RECTIFY_DO_MASK_KERNEL_INFO_H
#include <map>
#include <string>
#include <vector>

#include "pre_activate/common/optimizer.h"
namespace mindspore {
namespace opt {
class RectifyDoMaskKernelInfo : public PatternProcessPass {
 public:
  explicit RectifyDoMaskKernelInfo(bool multigraph = true)
      : PatternProcessPass("batch_norm_bert_fission", multigraph) {}
  ~RectifyDoMaskKernelInfo() override = default;
  const BaseRef DefinePattern() const override;
  const AnfNodePtr Process(const FuncGraphPtr &, const AnfNodePtr &, const EquivPtr &) const override;

 private:
  void RectifyKernelInfo(const std::vector<CNodePtr> &do_mask_node_list) const;
  std::string GetConvertFormat(const std::map<std::string, size_t> &format_counter) const;
  void RectifyDropOutDoMaskKernelInfo(const std::vector<CNodePtr> &do_mask_node_list, const std::string &format) const;
};
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PRE_ACTIVATE_ASCEND_FORMAT_TYPE_RECTIFY_DO_MASK_KERNEL_INFO_H
