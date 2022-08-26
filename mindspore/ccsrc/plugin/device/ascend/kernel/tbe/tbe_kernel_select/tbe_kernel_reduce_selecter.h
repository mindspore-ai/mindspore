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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_SELECT_TBE_KERNEL_REDUCE_SELECTER_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_SELECT_TBE_KERNEL_REDUCE_SELECTER_H_
#include <utility>
#include <string>
#include <vector>
#include "ir/anf.h"
#include "plugin/device/ascend/kernel/tbe/tbe_kernel_select/tbe_select_utils.h"
namespace mindspore {
namespace kernel {
class TbeKernelReduceSelecter {
 public:
  explicit TbeKernelReduceSelecter(CNodePtr cnode_ptr) : cnode_ptr_(std::move(cnode_ptr)) {}
  ~TbeKernelReduceSelecter() = default;
  bool GetShapeInfo(SupportFormat *support_format);
  bool IsReduceSupport5HD(SupportFormat *support_format) const;
  bool IsReduceSupportNDC1HWC0(SupportFormat *support_format) const;
  bool IsReduceSupportFracZ(SupportFormat *support_format) const;
  bool IsReduceSupportC1HWNCoC0(SupportFormat *support_format) const;
  bool IsReduceSupportFracNZ(SupportFormat *support_format) const;

 private:
  bool IsFracZAndC1HWNCoC0Common(const std::string &format, SupportFormat *support_format) const;
  void GetReduceAttrKeepDim();
  void AssignSupportFormat(const std::string &support_format_str, SupportFormat *support_format) const;
  bool Is4DShape(const ShapeVector &shape) const;
  bool Is5DShape(const ShapeVector &shape) const;
  void PadScalarShape(ShapeVector *shape) const;
  CNodePtr cnode_ptr_;
  ShapeVector input_shape_{};
  ShapeVector output_shape_{};
  std::vector<int64_t> axis_{};
  bool keep_dims_ = false;
};
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_TBE_KERNEL_REDUCE_SELECTER_H
