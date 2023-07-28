/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include "minddata/dataset/kernels/image/trivial_augment_wide_op.h"

#include "minddata/dataset/kernels/image/affine_op.h"
#include "minddata/dataset/kernels/image/image_utils.h"
#include "minddata/dataset/util/random.h"

namespace mindspore {
namespace dataset {
TrivialAugmentWideOp::TrivialAugmentWideOp(int32_t num_magnitude_bins, InterpolationMode interpolation,
                                           const std::vector<uint8_t> &fill_value)
    : num_magnitude_bins_(num_magnitude_bins), interpolation_(interpolation), fill_value_(fill_value) {
  rnd_.seed(GetSeed());
}

Status TrivialAugmentWideOp::Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) {
  IO_CHECK(input, output);
  RETURN_IF_NOT_OK(ValidateImage(input, "TrivialAugmentWide", {3}, {3}, {3}));

  std::vector<dsize_t> image_size;
  RETURN_IF_NOT_OK(ImageSize(input, &image_size));
  Space space = GetSpace(num_magnitude_bins_);
  size_t space_size = space.size();
  std::vector<std::string> op_name_list;
  (void)std::for_each(
    space.begin(), space.end(),
    [&op_name_list](const std::map<std::string, std::tuple<std::vector<float>, bool>>::value_type &p) {
      op_name_list.push_back(p.first);
    });

  int32_t op_index = RandInt(0, static_cast<int32_t>(space_size));
  const std::string op_name = op_name_list[static_cast<size_t>(op_index)];
  std::vector<float> magnitudes = std::get<0>(space[op_name]);
  bool sign = std::get<1>(space[op_name]);
  float magnitude = 0.0;
  if (magnitudes.size() != 1) {
    int32_t magnitude_index = RandInt(0, static_cast<int32_t>(magnitudes.size()));
    magnitude = magnitudes[static_cast<size_t>(magnitude_index)];
  }
  const int kRandUpperBound = 2;
  int32_t random_number = RandInt(0, kRandUpperBound);
  if (static_cast<int32_t>(sign) && random_number) {
    magnitude *= -1.0;
  }
  std::shared_ptr<Tensor> img = input;
  RETURN_IF_NOT_OK(ApplyAugment(img, &img, op_name, magnitude, interpolation_, fill_value_));
  *output = img;
  return Status::OK();
}

Space TrivialAugmentWideOp::GetSpace(int32_t num_bins) {
  Space space = {{"Identity", {{0.0}, false}},
                 {"ShearX", {Linspace(0.0, 0.99, num_bins), true}},
                 {"ShearY", {Linspace(0.0, 0.99, num_bins), true}},
                 {"TranslateX", {Linspace(0.0, 32.0, num_bins), true}},
                 {"TranslateY", {Linspace(0.0, 32.0, num_bins), true}},
                 {"Rotate", {Linspace(0.0, 135.0, num_bins), true}},
                 {"Brightness", {Linspace(0.0, 0.99, num_bins), true}},
                 {"Color", {Linspace(0.0, 0.99, num_bins), true}},
                 {"Contrast", {Linspace(0.0, 0.99, num_bins), true}},
                 {"Sharpness", {Linspace(0.0, 0.99, num_bins), true}},
                 {"Posterize",
                  {Linspace(0.0, static_cast<float>(num_bins) - 1.F, num_bins,
                            -6.0F / (static_cast<float>(num_bins) - 1.F), 8, true),
                   false}},
                 {"Solarize", {Linspace(255.0, 0.0, num_bins), false}},
                 {"AutoContrast", {{0.0}, false}},
                 {"Equalize", {{0.0}, false}}};
  return space;
}

int32_t TrivialAugmentWideOp::RandInt(int32_t low, int32_t high) {
  std::uniform_int_distribution<int32_t> dis(low, high);
  return dis(rnd_) % (high - low) + low;
}
}  // namespace dataset
}  // namespace mindspore
