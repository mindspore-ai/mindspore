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
#ifndef MINDSPORE_CORE_OPS_RFFT_H_
#define MINDSPORE_CORE_OPS_RFFT_H_
#include <memory>
#include <vector>

#include "ops/primitive_c.h"
#include "abstract/abstract_value.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace ops {
constexpr auto kNameRfft = "Rfft";
class Rfft : public PrimitiveC {
 public:
  Rfft() : PrimitiveC(kNameRfft) {}
  ~Rfft() = default;
  MS_DECLARE_PARENT(Rfft, PrimitiveC);
  void Init(const int64_t fft_length);
  void set_fft_length(const int64_t fft_length);
  int64_t get_fft_length() const;
};
AbstractBasePtr RfftInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args);
using PrimRfftPtr = std::shared_ptr<Rfft>;
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_RFFT_H_
