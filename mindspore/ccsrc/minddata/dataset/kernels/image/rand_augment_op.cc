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
#include "minddata/dataset/kernels/image/rand_augment_op.h"

#include "minddata/dataset/kernels/image/image_utils.h"
#include "minddata/dataset/util/random.h"

namespace mindspore {
namespace dataset {
RandAugmentOp::RandAugmentOp(int32_t num_ops, int32_t magnitude, int32_t num_magnitude_bins,
                             InterpolationMode interpolation, const std::vector<uint8_t> &fill_value)
    : num_ops_(num_ops),
      magnitude_(magnitude),
      num_magnitude_bins_(num_magnitude_bins),
      interpolation_(interpolation),
      fill_value_(fill_value) {
  rnd_.seed(GetSeed());
}

Status RandAugmentOp::Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) {
  // Input correctness judgment
  IO_CHECK(input, output);
  RETURN_IF_NOT_OK(ValidateImage(input, "RandAugment", {3}, {3}, {3}));

  std::vector<dsize_t> image_size;
  RETURN_IF_NOT_OK(ImageSize(input, &image_size));
  std::shared_ptr<Tensor> img = input;
  Space space = GetSpace(num_magnitude_bins_, image_size);
  int32_t space_size = space.size();
  std::vector<std::string> op_name_list;
  std::for_each(space.begin(), space.end(),
                [&op_name_list](const std::map<std::string, std::tuple<std::vector<float>, bool>>::value_type &p) {
                  op_name_list.push_back(p.first);
                });

  for (int i = 0; i < num_ops_; ++i) {
    int32_t op_index = RandInt(0, space_size);
    std::string op_name = op_name_list[op_index];
    std::vector<float> magnitudes = std::get<0>(space[op_name]);
    bool sign = std::get<1>(space[op_name]);
    float magnitude = 0.0;
    if (magnitudes.size() != 1) {
      magnitude = magnitudes[magnitude_];
    }
    const int kRandUpperBound = 2;
    int32_t random_number = RandInt(0, kRandUpperBound);
    if (sign && random_number) {
      magnitude *= -1.0;
    }
    RETURN_IF_NOT_OK(ApplyAugment(img, &img, op_name, magnitude, interpolation_, fill_value_));
  }
  *output = img;
  return Status::OK();
}

Space RandAugmentOp::GetSpace(int32_t num_bins, const std::vector<dsize_t> &image_size) {
  Space space = {{"Identity", {{0}, false}},
                 {"ShearX", {Linspace(0.0, 0.3, num_bins), true}},
                 {"ShearY", {Linspace(0.0, 0.3, num_bins), true}},
                 {"TranslateX", {Linspace(0.0, 150.0f / 331 * image_size[1], num_bins), true}},
                 {"TranslateY", {Linspace(0.0, 150.0f / 331 * image_size[0], num_bins), true}},
                 {"Rotate", {Linspace(0.0, 30, num_bins), true}},
                 {"Brightness", {Linspace(0.0, 0.9, num_bins), true}},
                 {"Color", {Linspace(0.0, 0.9, num_bins), true}},
                 {"Contrast", {Linspace(0.0, 0.9, num_bins), true}},
                 {"Sharpness", {Linspace(0.0, 0.9, num_bins), true}},
                 {"Posterize", {Linspace(0.0, num_bins - 1.f, num_bins, -4.0f / (num_bins - 1.f), 8, true), false}},
                 {"Solarize", {Linspace(255.0, 0.0, num_bins), false}},
                 {"AutoContrast", {{0}, false}},
                 {"Equalize", {{0}, false}}};
  return space;
}

int32_t RandAugmentOp::RandInt(int32_t low, int32_t high) {
  std::uniform_int_distribution<int32_t> dis(low, high);
  return dis(rnd_) % (high - low) + low;
}
}  // namespace dataset
}  // namespace mindspore
