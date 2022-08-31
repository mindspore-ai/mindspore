/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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

#include "minddata/dataset/kernels/image/auto_augment_op.h"

#include "minddata/dataset/kernels/image/image_utils.h"
#include "minddata/dataset/util/random.h"

namespace mindspore {
namespace dataset {
AutoAugmentOp::AutoAugmentOp(AutoAugmentPolicy policy, InterpolationMode interpolation,
                             const std::vector<uint8_t> &fill_value)
    : policy_(policy), interpolation_(interpolation), fill_value_(fill_value) {
  rnd_.seed(GetSeed());
  transforms_ = GetTransforms(policy);
}

Transforms AutoAugmentOp::GetTransforms(AutoAugmentPolicy policy) {
  if (policy == AutoAugmentPolicy::kImageNet) {
    return {{{"Posterize", 0.4, 8}, {"Rotate", 0.6, 9}},    {{"Solarize", 0.6, 5}, {"AutoContrast", 0.6, -1}},
            {{"Equalize", 0.8, -1}, {"Equalize", 0.6, -1}}, {{"Posterize", 0.6, 7}, {"Posterize", 0.6, 6}},
            {{"Equalize", 0.4, -1}, {"Solarize", 0.2, 4}},  {{"Equalize", 0.4, -1}, {"Rotate", 0.8, 8}},
            {{"Solarize", 0.6, 3}, {"Equalize", 0.6, -1}},  {{"Posterize", 0.8, 5}, {"Equalize", 1.0, -1}},
            {{"Rotate", 0.2, 3}, {"Solarize", 0.6, 8}},     {{"Equalize", 0.6, -1}, {"Posterize", 0.4, 6}},
            {{"Rotate", 0.8, 8}, {"Color", 0.4, 0}},        {{"Rotate", 0.4, 9}, {"Equalize", 0.6, -1}},
            {{"Equalize", 0.0, -1}, {"Equalize", 0.8, -1}}, {{"Invert", 0.6, -1}, {"Equalize", 1.0, -1}},
            {{"Color", 0.6, 4}, {"Contrast", 1.0, 8}},      {{"Rotate", 0.8, 8}, {"Color", 1.0, 2}},
            {{"Color", 0.8, 8}, {"Solarize", 0.8, 7}},      {{"Sharpness", 0.4, 7}, {"Invert", 0.6, -1}},
            {{"ShearX", 0.6, 5}, {"Equalize", 1.0, -1}},    {{"Color", 0.4, 0}, {"Equalize", 0.6, -1}},
            {{"Equalize", 0.4, -1}, {"Solarize", 0.2, 4}},  {{"Solarize", 0.6, 5}, {"AutoContrast", 0.6, -1}},
            {{"Invert", 0.6, -1}, {"Equalize", 1.0, -1}},   {{"Color", 0.6, 4}, {"Contrast", 1.0, 8}},
            {{"Equalize", 0.8, -1}, {"Equalize", 0.6, -1}}};
  } else if (policy == AutoAugmentPolicy::kCifar10) {
    return {{{"Invert", 0.1, -1}, {"Contrast", 0.2, 6}},        {{"Rotate", 0.7, 2}, {"TranslateX", 0.3, 9}},
            {{"Sharpness", 0.8, 1}, {"Sharpness", 0.9, 3}},     {{"ShearY", 0.5, 8}, {"TranslateY", 0.7, 9}},
            {{"AutoContrast", 0.5, -1}, {"Equalize", 0.9, -1}}, {{"ShearY", 0.2, 7}, {"Posterize", 0.3, 7}},
            {{"Color", 0.4, 3}, {"Brightness", 0.6, 7}},        {{"Sharpness", 0.3, 9}, {"Brightness", 0.7, 9}},
            {{"Equalize", 0.6, -1}, {"Equalize", 0.5, -1}},     {{"Contrast", 0.6, 7}, {"Sharpness", 0.6, 5}},
            {{"Color", 0.7, 7}, {"TranslateX", 0.5, 8}},        {{"Equalize", 0.3, -1}, {"AutoContrast", 0.4, -1}},
            {{"TranslateY", 0.4, 3}, {"Sharpness", 0.2, 6}},    {{"Brightness", 0.9, 6}, {"Color", 0.2, 8}},
            {{"Solarize", 0.5, 2}, {"Invert", 0.0, -1}},        {{"Equalize", 0.2, -1}, {"AutoContrast", 0.6, -1}},
            {{"Equalize", 0.2, -1}, {"Equalize", 0.6, -1}},     {{"Color", 0.9, 9}, {"Equalize", 0.6, -1}},
            {{"AutoContrast", 0.8, -1}, {"Solarize", 0.2, 8}},  {{"Brightness", 0.1, 3}, {"Color", 0.7, 0}},
            {{"Solarize", 0.4, 5}, {"AutoContrast", 0.9, -1}},  {{"TranslateY", 0.9, 9}, {"TranslateY", 0.7, 9}},
            {{"AutoContrast", 0.9, -1}, {"Solarize", 0.8, 3}},  {{"Equalize", 0.8, -1}, {"Invert", 0.1, -1}},
            {{"TranslateY", 0.7, 9}, {"AutoContrast", 0.9, -1}}};
  } else {
    return {{{"ShearX", 0.9, 4}, {"Invert", 0.2, -1}},        {{"ShearY", 0.9, 8}, {"Invert", 0.7, -1}},
            {{"Equalize", 0.6, -1}, {"Solarize", 0.6, 6}},    {{"Invert", 0.9, -1}, {"Equalize", 0.6, -1}},
            {{"Equalize", 0.6, -1}, {"Rotate", 0.9, 3}},      {{"ShearX", 0.9, 4}, {"AutoContrast", 0.8, -1}},
            {{"ShearY", 0.9, 8}, {"Invert", 0.4, -1}},        {{"ShearY", 0.9, 5}, {"Solarize", 0.2, 6}},
            {{"Invert", 0.9, -1}, {"AutoContrast", 0.8, -1}}, {{"Equalize", 0.6, -1}, {"Rotate", 0.9, 3}},
            {{"ShearX", 0.9, 4}, {"Solarize", 0.3, 3}},       {{"ShearY", 0.8, 8}, {"Invert", 0.7, -1}},
            {{"Equalize", 0.9, -1}, {"TranslateY", 0.6, 6}},  {{"Invert", 0.9, -1}, {"Equalize", 0.6, -1}},
            {{"Contrast", 0.3, 3}, {"Rotate", 0.8, 4}},       {{"Invert", 0.8, -1}, {"TranslateY", 0.0, 2}},
            {{"ShearY", 0.7, 6}, {"Solarize", 0.4, 8}},       {{"Invert", 0.6, -1}, {"Rotate", 0.8, 4}},
            {{"ShearY", 0.3, 7}, {"TranslateX", 0.9, 3}},     {{"ShearX", 0.1, 6}, {"Invert", 0.6, -1}},
            {{"Solarize", 0.7, 2}, {"TranslateY", 0.6, 7}},   {{"ShearY", 0.8, 4}, {"Invert", 0.8, -1}},
            {{"ShearX", 0.7, 9}, {"TranslateY", 0.8, 3}},     {{"ShearY", 0.8, 5}, {"AutoContrast", 0.7, -1}},
            {{"ShearX", 0.7, 2}, {"Invert", 0.1, -1}}};
  }
}

Status AutoAugmentOp::Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) {
  IO_CHECK(input, output);
  if (input->Rank() != kDefaultImageRank) {
    RETURN_STATUS_UNEXPECTED("AutoAugment: input tensor is not in shape of <H,W,C>, but got rank: " +
                             std::to_string(input->Rank()));
  }
  int num_channels = input->shape()[2];
  if (num_channels != kDefaultImageChannel) {
    RETURN_STATUS_UNEXPECTED("AutoAugment: channel of input image should be 3, but got: " +
                             std::to_string(num_channels));
  }

  int transform_id;
  std::vector<float> *probs = new std::vector<float>{0, 0};
  std::vector<int32_t> *signs = new std::vector<int32_t>{0, 0};
  GetParams(transforms_.size(), &transform_id, probs, signs);

  std::vector<dsize_t> image_size = {input->shape()[0], input->shape()[1]};
  std::shared_ptr<Tensor> img = input;

  const int num_augments = 2;
  for (auto i = 0; i < num_augments; i++) {
    std::string op_name = std::get<0>(transforms_[transform_id][i]);
    float p = std::get<1>(transforms_[transform_id][i]);
    int32_t magnitude_id = std::get<2>(transforms_[transform_id][i]);
    if ((*probs)[i] <= p) {
      Space space = GetSpace(10, image_size);
      std::vector<float> magnitudes = std::get<0>(space[op_name]);
      bool sign = std::get<1>(space[op_name]);
      float magnitude = 0.0;
      if (magnitudes.size() != 1 && magnitude_id != -1) {
        magnitude = magnitudes[magnitude_id];
      }
      if (sign && (*signs)[i] == 0) {
        magnitude *= -1.0;
      }
      RETURN_IF_NOT_OK(ApplyAugment(img, &img, op_name, magnitude, interpolation_, fill_value_));
    }
  }
  *output = img;
  delete probs;
  delete signs;
  return Status::OK();
}

void AutoAugmentOp::GetParams(int transform_num, int *transform_id, std::vector<float> *probs,
                              std::vector<int32_t> *signs) {
  std::uniform_int_distribution<int32_t> id_dist(0, transform_num - 1);
  *transform_id = id_dist(rnd_);
  std::uniform_real_distribution<float> prob_dist(0, 1);

  (*probs)[0] = prob_dist(rnd_);
  (*probs)[1] = prob_dist(rnd_);

  std::uniform_int_distribution<int32_t> sign_dist(0, 1);

  (*signs)[0] = sign_dist(rnd_);
  (*signs)[1] = sign_dist(rnd_);
}

Space AutoAugmentOp::GetSpace(int32_t num_bins, const std::vector<dsize_t> &image_size) {
  Space space = {{"ShearX", {Linspace(0.0, 0.3, num_bins), true}},
                 {"ShearY", {Linspace(0.0, 0.3, num_bins), true}},
                 {"TranslateX", {Linspace(0.0, 150.0f / 331 * image_size[1], num_bins), true}},
                 {"TranslateY", {Linspace(0.0, 150.0f / 331 * image_size[0], num_bins), true}},
                 {"Rotate", {Linspace(0.0, 30, num_bins), true}},
                 {"Brightness", {Linspace(0.0, 0.9, num_bins), true}},
                 {"Color", {Linspace(0.0, 0.9, num_bins), true}},
                 {"Contrast", {Linspace(0.0, 0.9, num_bins), true}},
                 {"Sharpness", {Linspace(0.0, 0.9, num_bins), true}},
                 {"Posterize", {Linspace(0.0, num_bins - 1.f, num_bins, -4.0f / (num_bins - 1.f), 8, true), false}},
                 {"Solarize", {Linspace(256.0, 0.0, num_bins), false}},
                 {"AutoContrast", {{0}, false}},
                 {"Equalize", {{0}, false}},
                 {"Invert", {{0}, false}}};
  return space;
}
}  // namespace dataset
}  // namespace mindspore
