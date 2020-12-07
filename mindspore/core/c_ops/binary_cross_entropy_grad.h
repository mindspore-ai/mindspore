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

#ifndef MINDSPORE_CORE_C_OPS_BINARY_CROSS_ENTROPY_GRAD_H_
#define MINDSPORE_CORE_C_OPS_BINARY_CROSS_ENTROPY_GRAD_H_
#include <vector>
#include <string>
#include <memory>

#include "c_ops/primitive_c.h"
#include "c_ops/op_utils.h"
#include "abstract/abstract_value.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
constexpr auto kNameBinaryCrossEntropyGrad = "BinaryCrossEntropyGrad";
class BinaryCrossEntropyGrad : public PrimitiveC {
 public:
  BinaryCrossEntropyGrad() : PrimitiveC(kNameBinaryCrossEntropyGrad) {}
  ~BinaryCrossEntropyGrad() = default;
  MS_DECLARE_PARENT(BinaryCrossEntropyGrad, PrimitiveC);
  void Init(const std::string &reduction = "mean");
  void set_reduction(const std::string &reduction);
  std::string get_reduction() const;
};
AbstractBasePtr BinaryCrossEntropyGradInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                            const std::vector<AbstractBasePtr> &input_args);
using PrimBinaryCrossEntropyGrad = std::shared_ptr<BinaryCrossEntropyGrad>;
}  // namespace mindspore
#endif  // MINDSPORE_CORE_C_OPS_BINARY_CROSS_ENTROPY_GRAD_H_
