/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CORE_OPS_SPLICE_H_
#define MINDSPORE_CORE_OPS_SPLICE_H_
#include <vector>
#include <memory>
#include "ops/primitive_c.h"
#include "abstract/abstract_value.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace ops {
constexpr auto kNameSplice = "Splice";
class Splice : public PrimitiveC {
 public:
  Splice() : PrimitiveC(kNameSplice) { InitIOName({"inputs"}, {"outputs"}); }
  ~Splice() = default;
  MS_DECLARE_PARENT(Splice, PrimitiveC);
  void Init(const std::vector<int64_t> &contexts, const std::vector<int64_t> &forward_indexes, int64_t output_dims);
  void set_context(const std::vector<int64_t> &contexts);
  void set_forward_indexes(const std::vector<int64_t> &forward_indexes);
  void set_output_dim(int64_t output_dim);

  std::vector<int64_t> get_context() const;
  std::vector<int64_t> get_forward_indexes() const;
  int64_t get_output_dim() const;
  AbstractBasePtr SpliceInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                              const std::vector<AbstractBasePtr> &input_args);
};
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_SPLICE_H_
