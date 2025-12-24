/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CORE_OPS_CONV2D_BACKPROP_INPUT_H_
#define MINDSPORE_CORE_OPS_CONV2D_BACKPROP_INPUT_H_
#include <map>
#include <memory>
#include <string>
#include <vector>
#include "mindapi/base/format.h"
#include "mindapi/base/types.h"
#include "ir/anf.h"
#include "ops/base_operator.h"

namespace mindspore {
namespace ops {
constexpr auto kNameConv2DBackpropInput = "Conv2DBackpropInput";
class OPS_API Conv2DBackpropInput : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(Conv2DBackpropInput);
  explicit Conv2DBackpropInput(const std::string &k_name = kNameConv2DBackpropInput) : BaseOperator(k_name) {
    InitIOName({"out_backprop", "filter", "input_sizes"}, {"output"});
  }
  void Init(int64_t out_channel, const std::vector<int64_t> &kernel_size, int64_t mode = 1,
            const PadMode &pad_mode = VALID, const std::vector<int64_t> &pad = {0, 0, 0, 0},
            const std::vector<int64_t> &stride = {1, 1, 1, 1}, const std::vector<int64_t> &dilation = {1, 1, 1, 1},
            int64_t group = 1, const Format &format = NCHW, const std::vector<int64_t> &pad_list = {0, 0, 0, 0});
  void set_kernel_size(const std::vector<int64_t> &kernel_size);
  void set_stride(const std::vector<int64_t> &stride);
  void set_dilation(const std::vector<int64_t> &dilation);
  void set_pad_mode(const PadMode &pad_mode);
  void set_pad(const std::vector<int64_t> &pad);
  void set_mode(int64_t mode);
  void set_group(int64_t group);
  void set_out_channel(int64_t out_channel);
  void set_format(const Format &format);
  void set_pad_list(const std::vector<int64_t> &pad_list);
  std::vector<int64_t> get_kernel_size() const;
  std::vector<int64_t> get_stride() const;
  std::vector<int64_t> get_dilation() const;
  PadMode get_pad_mode() const;
  std::vector<int64_t> get_pad() const;
  int64_t get_mode() const;
  int64_t get_group() const;
  int64_t get_out_channel() const;
  Format get_format() const;
  std::vector<int64_t> get_pad_list() const;
};
OPS_API abstract::AbstractBasePtr Conv2DBackpropInputInfer(const abstract::AnalysisEnginePtr &,
                                                           const PrimitivePtr &primitive,
                                                           const std::vector<abstract::AbstractBasePtr> &input_args);

BaseShapePtr Conv2DBackpropInputInferShape(const PrimitivePtr &primitive,
                                           const std::vector<AbstractBasePtr> &input_args);
TypePtr Conv2DBackpropInputInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args);
}  // namespace ops
}  // namespace mindspore
#endif  // MINDSPORE_CORE_OPS_CONV2D_BACKPROP_INPUT_H_
