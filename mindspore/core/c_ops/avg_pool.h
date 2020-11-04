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

#ifndef MINDSPORE_CORE_C_OPS_AVG_POOL_H_
#define MINDSPORE_CORE_C_OPS_AVG_POOL_H_

#include <map>
#include <vector>
#include <string>
#include <memory>
#include "c_ops/primitive_c.h"
#include "abstract/abstract_value.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
constexpr auto kNameAvgPool = "AvgPool";
class AvgPool : public PrimitiveC {
 public:
  AvgPool() : PrimitiveC(kNameAvgPool) { InitIOName({"x"}, {"output"}); }
  ~AvgPool() = default;
  MS_DECLARE_PARENT(AvgPool, PrimitiveC);
  void Init(const std::vector<int64_t> &kernel_size = {1}, const std::vector<int64_t> &stride = {1},
            const std::string &padding = "valid");
  void set_padding(const std::string &pad);
  void set_kernel_size(const std::vector<int64_t> &kernel_size);
  void set_strides(const std::vector<int64_t> &strides);
  std::vector<int64_t> get_kernel_size() const;
  std::vector<int64_t> get_strides() const;
  std::string get_padding() const;
};

AbstractBasePtr AvgPoolInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                             const std::vector<AbstractBasePtr> &input_args);
using PrimAvgPoolPtr = std::shared_ptr<AvgPool>;
}  // namespace mindspore

#endif  // MINDSPORE_CORE_C_OPS_AVG_POOL_H_
