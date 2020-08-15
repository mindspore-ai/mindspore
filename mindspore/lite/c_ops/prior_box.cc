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

#include "c_ops/prior_box.h"

namespace mindspore {
#ifdef PRIMITIVE_WRITEABLE
std::vector<int> PriorBox::GetMinSizes() const { return this->primitive->value.AsPriorBox()->max_sizes; }
std::vector<int> PriorBox::GetMaxSizes() const { return this->primitive->value.AsPriorBox()->max_sizes; }
std::vector<float> PriorBox::GetAspectRatios() const { return this->primitive->value.AsPriorBox()->aspect_ratios; }
std::vector<float> PriorBox::GetVariances() const { return this->primitive->value.AsPriorBox()->variances; }
int PriorBox::GetImageSizeW() const { return this->primitive->value.AsPriorBox()->image_size_w; }
int PriorBox::GetImageSizeH() const { return this->primitive->value.AsPriorBox()->image_size_h; }
float PriorBox::GetStepW() const { return this->primitive->value.AsPriorBox()->step_w; }
float PriorBox::GetStepH() const { return this->primitive->value.AsPriorBox()->step_h; }
bool PriorBox::GetClip() const { return this->primitive->value.AsPriorBox()->clip; }
bool PriorBox::GetFlip() const { return this->primitive->value.AsPriorBox()->flip; }
float PriorBox::GetOffset() const { return this->primitive->value.AsPriorBox()->offset; }

void PriorBox::SetMinSizes(const std::vector<int> &min_sizes) {
  this->primitive->value.AsPriorBox()->min_sizes = min_sizes;
}
void PriorBox::SetMaxSizes(const std::vector<int> &max_sizes) {
  this->primitive->value.AsPriorBox()->max_sizes = max_sizes;
}
void PriorBox::SetAspectRatios(const std::vector<float> &aspect_ratios) {
  this->primitive->value.AsPriorBox()->aspect_ratios = aspect_ratios;
}
void PriorBox::SetVariances(const std::vector<float> &variances) {
  this->primitive->value.AsPriorBox()->variances = variances;
}
void PriorBox::SetImageSizeW(int image_size_w) { this->primitive->value.AsPriorBox()->image_size_w = image_size_w; }
void PriorBox::SetImageSizeH(int image_size_h) { this->primitive->value.AsPriorBox()->image_size_h = image_size_h; }
void PriorBox::SetStepW(float step_w) { this->primitive->value.AsPriorBox()->step_w = step_w; }
void PriorBox::SetStepH(float step_h) { this->primitive->value.AsPriorBox()->step_h = step_h; }
void PriorBox::SetClip(bool clip) { this->primitive->value.AsPriorBox()->clip = clip; }
void PriorBox::SetFlip(bool flip) { this->primitive->value.AsPriorBox()->flip = flip; }
void PriorBox::SetOffset(float offset) { this->primitive->value.AsPriorBox()->offset = offset; }

#else

std::vector<int> PriorBox::GetMinSizes() const {
  auto fb_vector = this->primitive->value_as_PriorBox()->min_sizes();
  return std::vector<int>(fb_vector->begin(), fb_vector->end());
}
std::vector<int> PriorBox::GetMaxSizes() const {
  auto fb_vector = this->primitive->value_as_PriorBox()->max_sizes();
  return std::vector<int>(fb_vector->begin(), fb_vector->end());
}
std::vector<float> PriorBox::GetAspectRatios() const {
  auto fb_vector = this->primitive->value_as_PriorBox()->aspect_ratios();
  return std::vector<float>(fb_vector->begin(), fb_vector->end());
}
std::vector<float> PriorBox::GetVariances() const {
  auto fb_vector = this->primitive->value_as_PriorBox()->variances();
  return std::vector<float>(fb_vector->begin(), fb_vector->end());
}
int PriorBox::GetImageSizeW() const { return this->primitive->value_as_PriorBox()->image_size_w(); }
int PriorBox::GetImageSizeH() const { return this->primitive->value_as_PriorBox()->image_size_h(); }
float PriorBox::GetStepW() const { return this->primitive->value_as_PriorBox()->step_w(); }
float PriorBox::GetStepH() const { return this->primitive->value_as_PriorBox()->step_h(); }
bool PriorBox::GetClip() const { return this->primitive->value_as_PriorBox()->clip(); }
bool PriorBox::GetFlip() const { return this->primitive->value_as_PriorBox()->flip(); }
float PriorBox::GetOffset() const { return this->primitive->value_as_PriorBox()->offset(); }

void PriorBox::SetMinSizes(const std::vector<int> &min_sizes) {}
void PriorBox::SetMaxSizes(const std::vector<int> &max_sizes) {}
void PriorBox::SetAspectRatios(const std::vector<float> &aspect_ratios) {}
void PriorBox::SetVariances(const std::vector<float> &variances) {}
void PriorBox::SetImageSizeW(int image_size_w) {}
void PriorBox::SetImageSizeH(int image_size_h) {}
void PriorBox::SetStepW(float step_w) {}
void PriorBox::SetStepH(float step_h) {}
void PriorBox::SetClip(bool clip) {}
void PriorBox::SetFlip(bool flip) {}
void PriorBox::SetOffset(float offset) {}
#endif
namespace {
constexpr int kPriorBoxPoints = 4;
constexpr int kPriorBoxN = 1;
constexpr int kPriorBoxW = 1;
constexpr int kPriorBoxC = 2;
}  // namespace

int PriorBox::InferShape(std::vector<lite::tensor::Tensor *> inputs_, std::vector<lite::tensor::Tensor *> outputs_) {
  std::vector<float> different_aspect_ratios{1.0f};
  auto aspect_ratios = GetAspectRatios();
  MS_ASSERT(aspect_ratios != nullptr);
  for (auto i = 0; i < aspect_ratios.size(); i++) {
    float ratio = (aspect_ratios)[i];
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
  auto input = inputs_.at(0);
  MS_ASSERT(input != nullptr);
  int32_t h = input->Height() * input->Width() * num_priors_box * kPriorBoxPoints;

  std::vector<int> output_shape{kPriorBoxN, h, kPriorBoxW, kPriorBoxC};
  auto output = outputs_.at(0);
  MS_ASSERT(output != nullptr);

  output->set_shape(output_shape);
  output->set_data_type(kNumberTypeFloat32);
  output->SetFormat(input->GetFormat());
  return 0;
}
}  // namespace mindspore
