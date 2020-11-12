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

#ifndef LITE_MINDSPORE_LITE_C_OPS_PRIOR_BOX_H_
#define LITE_MINDSPORE_LITE_C_OPS_PRIOR_BOX_H_

#include <vector>
#include <set>
#include <cmath>
#include <memory>

#include "src/ops/primitive_c.h"

namespace mindspore {
namespace lite {
class PriorBox : public PrimitiveC {
 public:
  PriorBox() = default;
  ~PriorBox() = default;
#ifdef PRIMITIVE_WRITEABLE
  MS_DECLARE_PARENT(PriorBox, PrimitiveC);
  explicit PriorBox(schema::PrimitiveT *primitive) : PrimitiveC(primitive) {}
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
#else
  int UnPackToFlatBuilder(const schema::Primitive *primitive, flatbuffers::FlatBufferBuilder *fbb) override;
#endif
  int InferShape(std::vector<lite::Tensor *> inputs_, std::vector<lite::Tensor *> outputs_) override;
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
};
}  // namespace lite
}  // namespace mindspore

#endif  // LITE_MINDSPORE_LITE_C_OPS_PRIOR_BOX_H_
