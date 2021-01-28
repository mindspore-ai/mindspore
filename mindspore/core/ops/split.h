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

#ifndef MINDSPORE_CORE_OPS_SPLIT_H_
#define MINDSPORE_CORE_OPS_SPLIT_H_
#include <vector>
#include <memory>

#include "ops/primitive_c.h"
#include "abstract/abstract_value.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace ops {
constexpr auto kNameSplit = "Split";
class Split : public PrimitiveC {
 public:
  Split() : PrimitiveC(kNameSplit) {}
  ~Split() = default;
  MS_DECLARE_PARENT(Split, PrimitiveC);
  void Init(const std::vector<int64_t> &size_splits, const int64_t axis, const int64_t output_num);
  void set_size_splits(const std::vector<int64_t> &size_splits);
  void set_axis(const int64_t axis);
  void set_output_num(const int64_t output_num);
  std::vector<int64_t> get_size_splits() const;
  int64_t get_axis() const;
  int64_t get_output_num() const;
};
AbstractBasePtr SplitInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                           const std::vector<AbstractBasePtr> &input_args);
using PrimSplit = std::shared_ptr<Split>;
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_SPLIT_H_
