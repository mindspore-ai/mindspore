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

#include "src/ops/prior_box.h"

#ifndef PRIMITIVE_WRITEABLE
#include "src/ops/ops_register.h"
#endif

namespace mindspore {
namespace lite {
#ifdef PRIMITIVE_WRITEABLE
std::vector<int> PriorBox::GetMinSizes() const { return this->primitive_->value.AsPriorBox()->max_sizes; }
std::vector<int> PriorBox::GetMaxSizes() const { return this->primitive_->value.AsPriorBox()->max_sizes; }
std::vector<float> PriorBox::GetAspectRatios() const { return this->primitive_->value.AsPriorBox()->aspect_ratios; }
std::vector<float> PriorBox::GetVariances() const { return this->primitive_->value.AsPriorBox()->variances; }
int PriorBox::GetImageSizeW() const { return this->primitive_->value.AsPriorBox()->image_size_w; }
int PriorBox::GetImageSizeH() const { return this->primitive_->value.AsPriorBox()->image_size_h; }
float PriorBox::GetStepW() const { return this->primitive_->value.AsPriorBox()->step_w; }
float PriorBox::GetStepH() const { return this->primitive_->value.AsPriorBox()->step_h; }
bool PriorBox::GetClip() const { return this->primitive_->value.AsPriorBox()->clip; }
bool PriorBox::GetFlip() const { return this->primitive_->value.AsPriorBox()->flip; }
float PriorBox::GetOffset() const { return this->primitive_->value.AsPriorBox()->offset; }

void PriorBox::SetMinSizes(const std::vector<int> &min_sizes) {
  this->primitive_->value.AsPriorBox()->min_sizes = min_sizes;
}
void PriorBox::SetMaxSizes(const std::vector<int> &max_sizes) {
  this->primitive_->value.AsPriorBox()->max_sizes = max_sizes;
}
void PriorBox::SetAspectRatios(const std::vector<float> &aspect_ratios) {
  this->primitive_->value.AsPriorBox()->aspect_ratios = aspect_ratios;
}
void PriorBox::SetVariances(const std::vector<float> &variances) {
  this->primitive_->value.AsPriorBox()->variances = variances;
}
void PriorBox::SetImageSizeW(int image_size_w) { this->primitive_->value.AsPriorBox()->image_size_w = image_size_w; }
void PriorBox::SetImageSizeH(int image_size_h) { this->primitive_->value.AsPriorBox()->image_size_h = image_size_h; }
void PriorBox::SetStepW(float step_w) { this->primitive_->value.AsPriorBox()->step_w = step_w; }
void PriorBox::SetStepH(float step_h) { this->primitive_->value.AsPriorBox()->step_h = step_h; }
void PriorBox::SetClip(bool clip) { this->primitive_->value.AsPriorBox()->clip = clip; }
void PriorBox::SetFlip(bool flip) { this->primitive_->value.AsPriorBox()->flip = flip; }
void PriorBox::SetOffset(float offset) { this->primitive_->value.AsPriorBox()->offset = offset; }

#else

std::vector<int> PriorBox::GetMinSizes() const {
  auto fb_vector = this->primitive_->value_as_PriorBox()->min_sizes();
  return std::vector<int>(fb_vector->begin(), fb_vector->end());
}
std::vector<int> PriorBox::GetMaxSizes() const {
  auto fb_vector = this->primitive_->value_as_PriorBox()->max_sizes();
  return std::vector<int>(fb_vector->begin(), fb_vector->end());
}
std::vector<float> PriorBox::GetAspectRatios() const {
  auto fb_vector = this->primitive_->value_as_PriorBox()->aspect_ratios();
  return std::vector<float>(fb_vector->begin(), fb_vector->end());
}
std::vector<float> PriorBox::GetVariances() const {
  auto fb_vector = this->primitive_->value_as_PriorBox()->variances();
  return std::vector<float>(fb_vector->begin(), fb_vector->end());
}
int PriorBox::GetImageSizeW() const { return this->primitive_->value_as_PriorBox()->image_size_w(); }
int PriorBox::GetImageSizeH() const { return this->primitive_->value_as_PriorBox()->image_size_h(); }
float PriorBox::GetStepW() const { return this->primitive_->value_as_PriorBox()->step_w(); }
float PriorBox::GetStepH() const { return this->primitive_->value_as_PriorBox()->step_h(); }
bool PriorBox::GetClip() const { return this->primitive_->value_as_PriorBox()->clip(); }
bool PriorBox::GetFlip() const { return this->primitive_->value_as_PriorBox()->flip(); }
float PriorBox::GetOffset() const { return this->primitive_->value_as_PriorBox()->offset(); }

int PriorBox::UnPackToFlatBuilder(const schema::Primitive *primitive, flatbuffers::FlatBufferBuilder *fbb) {
  MS_ASSERT(nullptr != primitive);
  MS_ASSERT(nullptr != fbb);
  auto attr = primitive->value_as_PriorBox();
  if (attr == nullptr) {
    MS_LOG(ERROR) << "value_as_PriorBox return nullptr";
    return RET_ERROR;
  }
  std::vector<int32_t> min_sizes;
  if (attr->min_sizes() != nullptr) {
    for (int i = 0; i < static_cast<int>(attr->min_sizes()->size()); i++) {
      min_sizes.push_back(attr->min_sizes()->data()[i]);
    }
  }
  std::vector<int32_t> max_sizes;
  if (attr->max_sizes() != nullptr) {
    for (int i = 0; i < static_cast<int>(attr->max_sizes()->size()); i++) {
      max_sizes.push_back(attr->max_sizes()->data()[i]);
    }
  }
  std::vector<float> aspect_ratios;
  if (attr->aspect_ratios() != nullptr) {
    for (int i = 0; i < static_cast<int>(attr->aspect_ratios()->size()); i++) {
      aspect_ratios.push_back(attr->aspect_ratios()->data()[i]);
    }
  }
  std::vector<float> variances;
  if (attr->variances() != nullptr) {
    for (int i = 0; i < static_cast<int>(attr->variances()->size()); i++) {
      variances.push_back(attr->variances()->data()[i]);
    }
  }
  auto val_offset = schema::CreatePriorBoxDirect(*fbb, &min_sizes, &max_sizes, &aspect_ratios, &variances);
  auto prim_offset = schema::CreatePrimitive(*fbb, schema::PrimitiveType_PriorBox, val_offset.o);
  fbb->Finish(prim_offset);
  return RET_OK;
}

PrimitiveC *PriorBoxCreator(const schema::Primitive *primitive) {
  return PrimitiveC::NewPrimitiveC<PriorBox>(primitive);
}
Registry PriorBoxRegistry(schema::PrimitiveType_PriorBox, PriorBoxCreator);
#endif

namespace {
constexpr int kPriorBoxPoints = 4;
constexpr int kPriorBoxN = 1;
constexpr int kPriorBoxW = 1;
constexpr int kPriorBoxC = 2;
}  // namespace
int PriorBox::InferShape(std::vector<Tensor *> inputs_, std::vector<Tensor *> outputs_) {
  MS_ASSERT(this->primitive_ != nullptr);
  auto input = inputs_.at(0);
  MS_ASSERT(input != nullptr);
  auto output = outputs_.at(0);
  MS_ASSERT(output != nullptr);
  output->set_data_type(kNumberTypeFloat32);
  output->set_format(input->format());
  if (!infer_flag()) {
    return RET_INFER_INVALID;
  }
  std::vector<float> different_aspect_ratios{1.0f};
  auto aspect_ratios = GetAspectRatios();
  for (size_t i = 0; i < aspect_ratios.size(); i++) {
    float ratio = aspect_ratios[i];
    bool exist = std::any_of(different_aspect_ratios.begin(), different_aspect_ratios.end(),
                             [&](float v) { return abs(ratio - v) < 1e-6; });
    if (!exist) {
      different_aspect_ratios.emplace_back(ratio);
      if (GetFlip()) {
        different_aspect_ratios.emplace_back(1.0f / ratio);
      }
    }
  }
  int32_t num_priors_box = GetMinSizes().size() * different_aspect_ratios.size() + GetMaxSizes().size();
  int32_t h = input->Height() * input->Width() * num_priors_box * kPriorBoxPoints;
  std::vector<int> output_shape{kPriorBoxN, h, kPriorBoxW, kPriorBoxC};
  output->set_shape(output_shape);
  return RET_OK;
}
}  // namespace lite
}  // namespace mindspore
