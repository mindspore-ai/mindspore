/**
 * Copyright 2022-2025 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CORE_OPS_DEFORMABLE_OFFSETS_GRAD_H_
#define MINDSPORE_CORE_OPS_DEFORMABLE_OFFSETS_GRAD_H_

#include <memory>
#include <string>
#include <vector>

#include "mindapi/base/types.h"
#include "ops/base_operator.h"

namespace mindspore {
namespace ops {
constexpr auto kNameDeformableOffsetsGrad = "DeformableOffsetsGrad";
class OPS_API DeformableOffsetsGrad : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(DeformableOffsetsGrad);
  DeformableOffsetsGrad() : BaseOperator(kNameDeformableOffsetsGrad) {
    InitIOName({"out_backprop", "input", "offsets"}, {"output"});
  }
  explicit DeformableOffsetsGrad(const std::string k_name) : BaseOperator(k_name) {
    InitIOName({"out_backprop", "input", "offsets"}, {"output"});
  }
  void Init(const std::vector<int64_t> &strides, const std::vector<int64_t> &pads, const std::vector<int64_t> &ksize,
            const std::vector<int64_t> &dilation, const std::string &data_format, int64_t deformable_groups,
            bool modulated);
  void set_strides(const std::vector<int64_t> &strides);
  void set_pads(const std::vector<int64_t> &pads);
  void set_kernel_size(const std::vector<int64_t> &ksize);
  void set_dilations(const std::vector<int64_t> &dilations);
  void set_format(const std::string &format);
  void set_deformable_groups(const int64_t deformable_groups);
  void set_modulated(bool modulated);

  std::vector<int64_t> get_strides() const;
  std::vector<int64_t> get_pads() const;
  std::vector<int64_t> get_kernel_size() const;
  std::vector<int64_t> get_dilations() const;
  std::string get_format() const;
  int64_t get_deformable_groups() const;
  bool get_modulated() const;
};
}  // namespace ops
}  // namespace mindspore
#endif
