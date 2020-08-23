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

#ifndef LITE_MINDSPORE_LITE_C_OPS_B_N_GRAD_INPUT_H_
#define LITE_MINDSPORE_LITE_C_OPS_B_N_GRAD_INPUT_H_

#include <vector>
#include <set>
#include <cmath>
#include "ir/dtype/type_id.h"
#include "src/ops/primitive_c.h"

namespace mindspore {
namespace lite {
class BNGradInput : public PrimitiveC {
 public:
#ifdef PRIMITIVE_WRITEABLE
  MS_DECLARE_PARENT(BNGradInput, PrimitiveC);
  BNGradInput() = default;
  explicit BNGradInput(schema::PrimitiveT *primitive) : PrimitiveC(primitive) {}
  void SetEps(float eps);
  void SetChannels(int channels);
#else
  explicit BNGradInput(schema::Primitive *primitive) : PrimitiveC(primitive) {}
#endif
  float GetEps() const;
  int GetChannels() const;
};
}  // namespace lite
}  // namespace mindspore

#endif  // LITE_MINDSPORE_LITE_C_OPS_B_N_GRAD_INPUT_H_
