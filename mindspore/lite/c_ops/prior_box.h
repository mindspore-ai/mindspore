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

#ifndef LITE_MINDSPORE_LITE_C_OPS_PRIOR_BOX_H_
#define LITE_MINDSPORE_LITE_C_OPS_PRIOR_BOX_H_

namespace mindspore {
class PriorBox : public PrimitiveC {
 public:
#ifdef PRIMITIVE_WRITEABLE
  explicit PriorBox(schema::PrimitiveT *primitive) : PrimitiveC(primitive) {}
#else
  explicit PriorBox(schema::Primitive *primitive) : PrimitiveC(primitive) {}
#endif
  int InferShape(std::vector<lite::tensor::Tensor *> inputs_, std::vector<lite::tensor::Tensor *> outputs_) override;
  std::vector<int> GetMinSizes() const;
  std::vector<int> GetMaxSizes() const;
  std::vector<float> GetAspectRatios() const;
  std::vector<float> GetVariances() const;
  int GetImageSizeW() const;
  int GetImageSizeH() const;
  float GetStepW() const;
  float GetStepH() const;
  bool GetClip() const;
  bool GetFlip() const;
  float GetOffset() const;
  void SetMinSizes(const std::vector<int> &min_sizes);
  void SetMaxSizes(const std::vector<int> &max_sizes);
  void SetAspectRatios(const std::vector<float> &aspect_ratios);
  void SetVariances(const std::vector<float> &variances);
  void SetImageSizeW(int image_size_w);
  void SetImageSizeH(int image_size_h);
  void SetStepW(float step_w);
  void SetStepH(float step_h);
  void SetClip(bool clip);
  void SetFlip(bool flip);
  void SetOffset(float offset);
};
}  // namespace mindspore

#endif  // LITE_MINDSPORE_LITE_C_OPS_PRIOR_BOX_H_
