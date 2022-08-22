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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_SELECT_TBE_KERNEL_PATTERN_BROADCAST_SELECTOR_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_SELECT_TBE_KERNEL_PATTERN_BROADCAST_SELECTOR_H_

#include <vector>
#include <string>
#include <utility>
#include "ir/anf.h"
#include "plugin/device/ascend/kernel/tbe/tbe_kernel_select/tbe_select_utils.h"

namespace mindspore::kernel {
class TbeKernelBroadcastSelector {
 public:
  explicit TbeKernelBroadcastSelector(CNodePtr cnode_ptr) : cnode_ptr_(std::move(cnode_ptr)) {}
  ~TbeKernelBroadcastSelector() = default;
  void GetSupportedFormatDType(SupportFormatDType *support_format_dtype);

 private:
  void GetBroadCastNodeInfo();
  void GetCheckInfo();
  void GetBroadcastSupport5HD(SupportFormat *support_format) const;
  void GetBroadcastSupportFracZ(SupportFormat *support_format) const;
  void GetBroadcastSupportC1HWNCoC0(SupportFormat *support_format) const;
  void GetBroadcastSupportFracNZ(SupportFormat *support_format) const;
  void GetBroadcastSupportNDC1HWC0(SupportFormat *support_format) const;
  void GetBroadcastSupportFracZ3D(SupportFormat *support_format) const;
  [[nodiscard]] inline bool IsSameShape() const;
  [[nodiscard]] static inline bool Is4DShape(const ShapeVector &shape);
  [[nodiscard]] static inline bool Is5DShape(const ShapeVector &shape);
  [[nodiscard]] static inline bool IsScalarShape(const ShapeVector &shape);
  [[nodiscard]] inline bool HasScalarInput() const;
  [[nodiscard]] inline bool IsInputBroadcastByChannel(size_t channel) const;
  void GenOutputSupportFormat(const std::string &support_format, SupportFormatItem *output_support_item) const;
  // broadcast
  CNodePtr cnode_ptr_ = nullptr;
  size_t input_num_ = 0;
  size_t output_num_ = 0;
  std::vector<ShapeVector> input_shapes_{};
  std::vector<ShapeVector> output_shapes_{};
  // check info
  bool is_same_shape_ = false;
  bool has_scalar_input_ = false;
};
}  // namespace mindspore::kernel
#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_SELECT_TBE_KERNEL_PATTERN_BROADCAST_SELECTOR_H_
