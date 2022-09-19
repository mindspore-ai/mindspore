/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_SELECT_TBE_KERNEL_PATTERN_REDUCE_SELECTOR_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_SELECT_TBE_KERNEL_PATTERN_REDUCE_SELECTOR_H_
#include <utility>
#include <string>
#include <vector>
#include "ir/anf.h"
#include "plugin/device/ascend/kernel/tbe/tbe_kernel_select/tbe_select_utils.h"

namespace mindspore::kernel {
class TbeKernelReduceSelector {
 public:
  explicit TbeKernelReduceSelector(CNodePtr cnode_ptr) : cnode_ptr_(std::move(cnode_ptr)) {}
  ~TbeKernelReduceSelector() = default;
  void GetSupportedFormatDType(SupportFormatDType *support_format_dtype);

 private:
  void GetReduceNodeInfo();
  void GetCheckInfo();
  void GetReduceSupport5HD(SupportFormat *support_format) const;
  void GetReduceSupportNDC1HWC0(SupportFormat *support_format) const;
  void GetReduceSupportFracZ(SupportFormat *support_format) const;
  void GetReduceSupportC1HWNCoC0(SupportFormat *support_format) const;
  void GetReduceSupportFracZ3D(SupportFormat *support_format) const;
  void GetReduceSupportFracNZ(SupportFormat *support_format) const;
  void GetReduceAttrKeepDim();
  static void FilterInvalidFormatDType(SupportFormatDType *support_format_dtype);
  [[nodiscard]] inline bool CheckOriginInputShapeDimEqual(size_t support_dim_size) const;
  [[nodiscard]] inline bool CheckOriginInputShapeDimLess(size_t support_min_dim_size) const;
  [[nodiscard]] inline bool CheckReduceContainChanel(int64_t channel_index) const;

  CNodePtr cnode_ptr_;
  std::vector<ShapeVector> input_shape_{};
  std::vector<ShapeVector> output_shape_{};
  std::vector<int64_t> axis_{};
  bool keep_dims_ = false;
  // check info
  bool is_shape_4_dims_ = false;
  bool is_shape_5_dims_ = false;
  bool is_shape_less_2_dims_ = false;
  bool is_shape_less_4_dims_ = false;
  bool is_shape_less_5_dims_ = false;
  bool is_reduce_n_channel_ = false;
  bool is_reduce_c_channel_ = false;
};
}  // namespace mindspore::kernel
#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_SELECT_TBE_KERNEL_PATTERN_REDUCE_SELECTOR_H_
