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

#ifndef MINDSPORE_CCSRC_KERNEL_TBE_KERNEL_BROADCAST_SELECTER_H_
#define MINDSPORE_CCSRC_KERNEL_TBE_KERNEL_BROADCAST_SELECTER_H_

#include <vector>
#include <string>
#include <utility>
#include "ir/anf.h"
#include "kernel/tbe/tbe_kernel_select/common_utils.h"

namespace mindspore {
namespace kernel {
class TbeKernelBroadCastSelecter {
 public:
  explicit TbeKernelBroadCastSelecter(CNodePtr cnode_ptr) : cnode_ptr_(std::move(cnode_ptr)) {}
  ~TbeKernelBroadCastSelecter() = default;
  bool GetShapeInfo(SupportFormat *support_format);
  bool IsBroadCastSupport5HD(SupportFormat *support_format) const;
  bool IsBroadCastSupportFracZ(SupportFormat *support_format) const;
  bool IsBroadCastSupportC1HWNCoC0(SupportFormat *support_format) const;
  bool IsBroadCastSupportFracNZ(SupportFormat *support_format) const;
  bool IsBroadCastSupportNDC1HWC0(SupportFormat *support_format) const;

 private:
  bool IsSameShape() const;
  void PadScalarShape(std::vector<size_t> *shape) const;
  bool Is4DShape(const std::vector<size_t> &shape) const;
  bool IsScalarShape(const std::vector<size_t> &shape) const;
  bool HasScalarInput() const;
  void GenOutputSupportFormat(const std::string &support_format, SupportFormatItem *output_support_item) const;
  void AssignSupportFormat(const std::string &support_format_str, SupportFormat *support_format) const;
  // broadcast
  CNodePtr cnode_ptr_;
  size_t input_num_{};
  size_t output_num_{};
  std::vector<std::vector<size_t>> input_shapes_;
  std::vector<std::vector<size_t>> output_shapes_;
};

}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_TBE_KERNEL_BROADCAST_SELECTER_HELPER_H
