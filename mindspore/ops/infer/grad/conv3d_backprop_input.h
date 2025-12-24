/**
 * Copyright 2020-2023 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CORE_OPS_CONV3D_BACKPROP_INPUT_H_
#define MINDSPORE_CORE_OPS_CONV3D_BACKPROP_INPUT_H_
#include <memory>
#include <string>
#include <vector>

#include "mindapi/base/format.h"
#include "mindapi/base/types.h"
#include "ops/base_operator.h"

namespace mindspore {
namespace ops {
constexpr auto kNameConv3DBackpropInput = "Conv3DBackpropInput";
class OPS_API Conv3DBackpropInput : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(Conv3DBackpropInput);
  Conv3DBackpropInput() : BaseOperator(kNameConv3DBackpropInput) {
    InitIOName({"filter", "out_backprop", "input_size"}, {"output"});
  }
  void Init(int64_t out_channel, const std::vector<int64_t> &kernel_size, int64_t mode = 1,
            const PadMode &pad_mode = VALID, const std::vector<int64_t> &pad = {0, 0, 0, 0, 0, 0},
            const std::vector<int64_t> &stride = {1, 1, 1, 1, 1},
            const std::vector<int64_t> &dilation = {1, 1, 1, 1, 1}, int64_t group = 1, const Format &format = NCDHW);
  void set_out_channel(const int64_t out_channel);
  void set_kernel_size(const std::vector<int64_t> &kernel_size);
  void set_mode(const int64_t mode);
  void set_pad_mode(const PadMode &pad_mode);
  void set_pad(const std::vector<int64_t> &pad);
  void set_stride(const std::vector<int64_t> &stride);
  void set_dilation(const std::vector<int64_t> &dilation);
  void set_group(const int64_t group);
  void set_format(const Format &format);
  void set_pad_list(const std::vector<int64_t> &pad_list);

  int64_t get_out_channel() const;
  std::vector<int64_t> get_kernel_size() const;
  int64_t get_mode() const;
  std::string get_pad_mode() const;
  std::vector<int64_t> get_pad() const;
  std::vector<int64_t> get_stride() const;
  std::vector<int64_t> get_dilation() const;
  int64_t get_group() const;
  Format get_format() const;
  std::vector<int64_t> get_pad_list() const;
};
}  // namespace ops
}  // namespace mindspore
#endif  // MINDSPORE_CORE_OPS_CONV3D_BACKPROP_INPUT_H_
