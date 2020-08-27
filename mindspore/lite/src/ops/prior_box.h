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
#include "ir/dtype/type_id.h"
#include "src/ops/primitive_c.h"

namespace mindspore {
namespace lite {
class PriorBox : public PrimitiveC {
 public:
#ifdef PRIMITIVE_WRITEABLE
  MS_DECLARE_PARENT(PriorBox, PrimitiveC);
  PriorBox() = default;
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
  explicit PriorBox(schema::Primitive *primitive) : PrimitiveC(primitive) {}

  schema::Primitive *Init(schema::Primitive *primitive) {
    flatbuffers::FlatBufferBuilder fbb(1024);

    auto attr = primitive->value_as_PriorBox();
    MS_ASSERT(attr != nullptr);

    auto min_sizes = std::make_unique<std::vector<int32_t>>();
    for (int i = 0; i < static_cast<int>(attr->min_sizes()->size()); i++) {
      min_sizes->push_back(attr->min_sizes()->data()[i]);
    }
    auto max_sizes = std::make_unique<std::vector<int32_t>>();
    for (int i = 0; i < static_cast<int>(attr->max_sizes()->size()); i++) {
      max_sizes->push_back(attr->max_sizes()->data()[i]);
    }
    auto aspect_ratios = std::make_unique<std::vector<float>>();
    for (int i = 0; i < static_cast<int>(attr->aspect_ratios()->size()); i++) {
      aspect_ratios->push_back(attr->aspect_ratios()->data()[i]);
    }
    auto variances = std::make_unique<std::vector<float>>();
    for (int i = 0; i < static_cast<int>(attr->variances()->size()); i++) {
      variances->push_back(attr->variances()->data()[i]);
    }

    auto val_offset = schema::CreatePriorBoxDirect(fbb, min_sizes.release(), max_sizes.release(),
                                                   aspect_ratios.release(), variances.release());
    auto prim_offset = schema::CreatePrimitive(fbb, schema::PrimitiveType_PriorBox, val_offset.o);
    fbb.Finish(prim_offset);

    auto buf = fbb.GetBufferPointer();
    MS_ASSERT(buf != nullptr);
    auto buf_bak = new char[fbb.GetSize()];
    memcpy(buf_bak, buf, fbb.GetSize());

    auto root = flatbuffers::GetRoot<schema::Primitive>(buf_bak);
    auto prim = const_cast<schema::Primitive *>(root);

    delete[] buf_bak;
    fbb.Clear();
    return prim;
  }
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
};
}  // namespace lite
}  // namespace mindspore

#endif  // LITE_MINDSPORE_LITE_C_OPS_PRIOR_BOX_H_
