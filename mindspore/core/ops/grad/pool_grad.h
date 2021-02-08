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

#ifndef MINDSPORE_CORE_OPS_POOL_GRAD_H_
#define MINDSPORE_CORE_OPS_POOL_GRAD_H_
#include <map>
#include <vector>
#include <string>
#include <memory>
#include "ops/primitive_c.h"
#include "abstract/abstract_value.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace ops {
constexpr auto kNamePoolGrad = "PoolGrad";
class PoolGrad : public PrimitiveC {
 public:
  PoolGrad() : PrimitiveC(kNamePoolGrad) { InitIOName({"x_origin", "out_origin", "grad"}, {"output"}); }
  explicit PoolGrad(const std::string k_name) : PrimitiveC(k_name) {
    InitIOName({"x_origin", "out_origin", "grad"}, {"output"});
  }
  ~PoolGrad() = default;
  MS_DECLARE_PARENT(PoolGrad, PrimitiveC);
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
  std::vector<int64_t> _grad_check_vector(const std::string arg_name, const std::vector<int64_t> arg_val,
                                          const std::string op_name);
};
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_POOL_GRAD_H_
