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

#ifndef MINDSPORE_CORE_OPS_POOL_GRAD_H_
#define MINDSPORE_CORE_OPS_POOL_GRAD_H_
#include <map>
#include <memory>
#include <string>
#include <vector>
#include "mindapi/base/format.h"
#include "mindapi/base/types.h"
#include "ops/base_operator.h"

namespace mindspore {
namespace ops {
constexpr auto kNamePoolGrad = "PoolGrad";
class OPS_API PoolGrad : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(PoolGrad);
  PoolGrad() : BaseOperator(kNamePoolGrad) { InitIOName({"x_origin", "out_origin", "grad"}, {"output"}); }
  explicit PoolGrad(const std::string k_name) : BaseOperator(k_name) {
    InitIOName({"x_origin", "out_origin", "grad"}, {"output"});
  }
  virtual void Init(const std::vector<int64_t> &kernel_size = {1}, const std::vector<int64_t> &strides = {1},
                    const PadMode &pad_mode = VALID, const Format &format = NCHW);
  virtual void set_kernel_size(const std::vector<int64_t> &kernel_size);
  virtual void set_strides(const std::vector<int64_t> &strides);
  void set_pad_mode(const PadMode &pad_mode);
  void set_format(const Format &format);

  std::vector<int64_t> get_kernel_size() const;
  std::vector<int64_t> get_strides() const;
  PadMode get_pad_mode() const;
  Format get_format() const;
  std::vector<int64_t> _grad_check_vector(const std::string &arg_name, const std::vector<int64_t> arg_val,
                                          const std::string &op_name);
};

static std::map<std::string, int64_t> pad_map = {
  {"CALCULATED", PadMode::PAD},
  {"PAD", PadMode::PAD},
  {"SAME", PadMode::SAME},
  {"VALID", PadMode::VALID},
};

static std::map<std::string, int64_t> dataformat_map = {
  {"NCHW", Format::NCHW},
  {"NHWC", Format::NHWC},
  {"NCDHW", Format::NCDHW},
  {"NDHWC", Format::NDHWC},
};
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_POOL_GRAD_H_
