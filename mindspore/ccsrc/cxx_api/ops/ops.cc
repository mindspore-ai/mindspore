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
#include "include/api/ops/ops.h"

namespace mindspore {
Conv2D::Conv2D(int out_channel, const std::vector<int> &kernel_size, int mode, const std::string &pad_mode,
               const std::vector<int> &pad, const std::vector<int> &stride, const std::vector<int> &dilation, int group)
    : OpCell("Conv2D"),
      out_channel(out_channel),
      kernel_size(kernel_size),
      mode(mode),
      pad_mode(pad_mode),
      pad(pad),
      stride(stride),
      dilation(dilation),
      group(group) {}

Output Conv2D::operator()(const Input &input1, const Input &input2) const {
  return CellBase::operator()({input1, input2})[0];
}

std::vector<Output> Conv2D::Construct(const std::vector<Input> &inputs) {
  return {Output(shared_from_this(), inputs, 1)};
}
}  // namespace mindspore
