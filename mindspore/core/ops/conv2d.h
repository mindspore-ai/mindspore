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

#ifndef MINDSPORE_CORE_OPS_CONV2D_H_
#define MINDSPORE_CORE_OPS_CONV2D_H_
#include <map>
#include <vector>
#include <string>
#include <memory>

#include "ops/op_utils.h"
#include "ops/primitive_c.h"
#include "abstract/abstract_value.h"
#include "utils/check_convert_utils.h"
namespace mindspore {
namespace ops {
constexpr auto kNameConv2D = "Conv2D";
class Conv2D : public PrimitiveC {
 public:
  Conv2D() : PrimitiveC(kNameConv2D) { InitIOName({"x", "w"}, {"output"}); }
  explicit Conv2D(const std::string k_name) : PrimitiveC(k_name) { InitIOName({"x", "w"}, {"output"}); }
  ~Conv2D() = default;
  MS_DECLARE_PARENT(Conv2D, PrimitiveC);
  void Init(int64_t out_channel, const std::vector<int64_t> &kernel_size, int64_t mode = 1,
            const PadMode &pad_mode = VALID, const std::vector<int64_t> &pad = {0, 0, 0, 0},
            const std::vector<int64_t> &stride = {1, 1, 1, 1}, const std::vector<int64_t> &dilation = {1, 1, 1, 1},
            int64_t group = 1, const Format &format = NCHW);
  void set_kernel_size(const std::vector<int64_t> &kernel_size);
  void set_stride(const std::vector<int64_t> &stride);
  void set_dilation(const std::vector<int64_t> &dilation);
  void set_pad_mode(const PadMode &pad_mode);
  void set_pad(const std::vector<int64_t> &pad);
  void set_mode(int64_t mode);
  void set_group(int64_t group);
  void set_out_channel(int64_t out_channel);
  void set_format(const Format &format);
  std::vector<int64_t> get_kernel_size() const;
  std::vector<int64_t> get_stride() const;
  std::vector<int64_t> get_dilation() const;
  PadMode get_pad_mode() const;
  std::vector<int64_t> get_pad() const;
  int64_t get_mode() const;
  int64_t get_group() const;
  int64_t get_out_channel() const;
  Format get_format() const;
};
AbstractBasePtr Conv2dInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                            const std::vector<AbstractBasePtr> &input_args);
using PrimConv2dPtr = std::shared_ptr<Conv2D>;
}  // namespace ops
}  // namespace mindspore
#endif  // MINDSPORE_CORE_OPS_CONV2D_H_
