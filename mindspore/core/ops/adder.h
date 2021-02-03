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

#ifndef MINDSPORE_CORE_OPS_ADDER_H_
#define MINDSPORE_CORE_OPS_ADDER_H_

#include <map>
#include <vector>
#include <string>
#include <memory>
#include "ops/primitive_c.h"
#include "abstract/abstract_value.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace ops {
constexpr auto kNameAdder = "Adder";
class Adder : public PrimitiveC {
 public:
  explicit Adder(const std::string &k_name = kNameAdder) : PrimitiveC(k_name) {}
  ~Adder() = default;
  MS_DECLARE_PARENT(Adder, PrimitiveC);
  void Init(const int64_t in_channel, const int64_t out_channel, const std::vector<int64_t> &kernel_size,
            const PadMode &pad_mode, const std::vector<int64_t> &stride, const std::vector<int64_t> &pad_list,
            const std::vector<int64_t> &dilation, const int64_t group, const Format &format);
  void set_in_channel(const int64_t in_channel);
  void set_out_channel(const int64_t out_channel);
  void set_kernel_size(const std::vector<int64_t> &kernel_size);
  void set_pad_mode(const PadMode &pad_mode);
  void set_stride(const std::vector<int64_t> &stride);
  void set_pad_list(const std::vector<int64_t> &pad_list);
  void set_dilation(const std::vector<int64_t> &dilation);
  void set_group(const int64_t group);
  void set_format(const Format &format);

  int64_t get_in_channel() const;
  int64_t get_out_channel() const;
  std::vector<int64_t> get_kernel_size() const;
  PadMode get_pad_mode() const;
  std::vector<int64_t> get_stride() const;
  std::vector<int64_t> get_pad_list() const;
  std::vector<int64_t> get_dilation() const;
  int64_t get_group() const;
  Format get_format() const;
};
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_ADDER_H_
