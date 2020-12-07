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

#ifndef MINDSPORE_CORE_C_OPS_DROPOUT_H_
#define MINDSPORE_CORE_C_OPS_DROPOUT_H_
#include "c_ops/primitive_c.h"
#include "abstract/abstract_value.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
constexpr auto kNameDropout = "Dropout";
class Dropout : public PrimitiveC {
 public:
  Dropout() : PrimitiveC(kNameDropout) {}
  ~Dropout() = default;
  MS_DECLARE_PARENT(Dropout, PrimitiveC);
  void Init(float keep_prob = 0.5);
  void set_keep_prob(float keep_prob);
  float get_keep_prob();
};
}  // namespace mindspore
#endif  // MINDSPORE_CORE_C_OPS_DROPOUT_H_
