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

#include <vector>
#include <set>
#include <cmath>
#include "ir/dtype/type_id.h"
#include "mindspore/lite/c_ops/primitive_c.h"
#ifdef PRIMITIVE_WRITEABLE
#include "schema/inner/model_generated.h"
#else
#include "schema/model_generated.h"
#endif

#ifndef LITE_MINDSPORE_LITE_C_OPS_LOCAL_RESPONSE_NORMALIZATION_H_
#define LITE_MINDSPORE_LITE_C_OPS_LOCAL_RESPONSE_NORMALIZATION_H_

namespace mindspore {
class LocalResponseNormalization : public PrimitiveC {
 public:
#ifdef PRIMITIVE_WRITEABLE
  explicit LocalResponseNormalization(schema::PrimitiveT *primitive) : PrimitiveC(primitive) {}
#else
  explicit LocalResponseNormalization(schema::Primitive *primitive) : PrimitiveC(primitive) {}
#endif
  int GetDepthRadius() const;
  float GetBias() const;
  float GetAlpha() const;
  float GetBeta() const;
  void SetDepthRadius(int depth_radius);
  void SetBias(float bias);
  void SetAlpha(float alpha);
  void SetBeta(float beta);
};
}  // namespace mindspore

#endif  // LITE_MINDSPORE_LITE_C_OPS_LOCAL_RESPONSE_NORMALIZATION_H_
