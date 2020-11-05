/**
 * Copyright 2019-2020 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CORE_C_OPS_DEPTHWISE_CONV2D_H
#define MINDSPORE_CORE_C_OPS_DEPTHWISE_CONV2D_H
#include <map>
#include <vector>
#include <string>
#include <memory>
#include "c_ops/primitive_c.h"
#include "abstract/abstract_value.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
constexpr auto kNameDepthWiseConv2D = "DepthwiseConv2dNative";
class DepthWiseConv2D : public PrimitiveC {
 public:
  DepthWiseConv2D() : PrimitiveC(kNameDepthWiseConv2D) { InitIOName({"x", "w"}, {"output"}); }
  ~DepthWiseConv2D() = default;
  MS_DECLARE_PARENT(DepthWiseConv2D, PrimitiveC);
  void Init(int out_channel, const std::vector<int64_t> &kernel_size, int mode = 1,
            const std::string &pad_mode = "valid", const std::vector<int64_t> &pad = {0, 0, 0, 0},
            const std::vector<int64_t> &stride = {1, 1, 1, 1}, const std::vector<int64_t> &dilation = {1, 1, 1, 1},
            int group = 1);
  std::vector<int64_t> get_kernel_size() const;
  std::vector<int64_t> get_stride() const;
  std::vector<int64_t> get_dilation() const;
  std::string get_pad_mode() const;
  std::vector<int64_t> get_pad() const;
  std::vector<int64_t> get_pads() const;
  int get_mode() const;
  int get_group() const;
  int get_output_channel() const;
  void set_kernel_size(const std::vector<int64_t> &kernel_size);
  void set_stride(const std::vector<int64_t> &stride);
  void set_dilation(const std::vector<int64_t> &dilation);
  void set_pad_mode(const std::string &pad_mode);
  void set_pad(const std::vector<int64_t> &pad);
  void set_mode(int mode);
  void set_group(int group);
  void set_out_channel(int output_channel);
  void set_pads(const std::vector<int64_t> &pad_list);
};
AbstractBasePtr DepthWiseConv2DInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                     const std::vector<AbstractBasePtr> &input_args);
using PrimDepthWiseConv2DPtr = std::shared_ptr<DepthWiseConv2D>;
}  // namespace mindspore

#endif  // MINDSPORE_CORE_C_OPS_DEPTHWISE_CONV2D_H
