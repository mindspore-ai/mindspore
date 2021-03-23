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
#ifndef MINDSPORE_INCLUDE_API_OPS_OPS_H
#define MINDSPORE_INCLUDE_API_OPS_OPS_H

#include <string>
#include <vector>
#include <map>
#include <memory>
#include "include/api/status.h"
#include "include/api/types.h"
#include "include/api/cell.h"

namespace mindspore {
struct MS_API Conv2D : public OpCell<Conv2D> {
  Conv2D() : OpCell("Conv2D") {}
  ~Conv2D() override = default;
  std::vector<Output> Construct(const std::vector<Input> &inputs) override;
  Conv2D(int out_channel, const std::vector<int> &kernel_size, int mode = 1, const std::string &pad_mode = "valid",
         const std::vector<int> &pad = {0, 0, 0, 0}, const std::vector<int> &stride = {1, 1, 1, 1},
         const std::vector<int> &dilation = {1, 1, 1, 1}, int group = 1);

  Output operator()(const Input &, const Input &) const;

  int out_channel;
  std::vector<int> kernel_size;
  int mode = 1;
  std::string pad_mode = "valid";
  std::vector<int> pad = {0, 0, 0, 0};
  std::vector<int> stride = {1, 1, 1, 1};
  std::vector<int> dilation = {1, 1, 1, 1};
  int group = 1;
};
}  // namespace mindspore
#endif  // MINDSPORE_INCLUDE_API_OPS_OPS_H
