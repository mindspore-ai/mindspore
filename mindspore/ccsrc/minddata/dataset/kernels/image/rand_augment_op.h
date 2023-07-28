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

#ifndef MINDSPORE_MINDSPORE_CCSRC_MINDDATA_DATASET_KERNELS_IMAGE_RAND_AUGMENT_OP_H_
#define MINDSPORE_MINDSPORE_CCSRC_MINDDATA_DATASET_KERNELS_IMAGE_RAND_AUGMENT_OP_H_

#include <cstdlib>
#include <map>
#include <memory>
#include <random>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "minddata/dataset/core/tensor.h"
#include "minddata/dataset/kernels/tensor_op.h"
#ifndef ENABLE_ANDROID
#include "minddata/dataset/kernels/image/image_utils.h"
#else
#include "minddata/dataset/kernels/image/lite_image_utils.h"
#endif
#include "minddata/dataset/util/status.h"

typedef std::map<std::string, std::tuple<std::vector<float>, bool>> Space;

namespace mindspore {
namespace dataset {
class RandAugmentOp : public TensorOp {
 public:
  RandAugmentOp(int32_t num_ops, int32_t magnitude, int32_t num_magnitude_bins, InterpolationMode interpolation,
                std::vector<uint8_t> fill_value);

  ~RandAugmentOp() override = default;

  Status Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) override;

  std::string Name() const override { return kRandAugmentOp; }

 private:
  static Space GetSpace(int32_t num_bins, const std::vector<dsize_t> &image_size);

  int32_t RandInt(int32_t low, int32_t high);

  int num_ops_;
  int magnitude_;
  int num_magnitude_bins_;
  InterpolationMode interpolation_;
  std::vector<uint8_t> fill_value_;
  std::mt19937 rnd_;
};
}  // namespace dataset
}  // namespace mindspore
#endif  // MINDSPORE_MINDSPORE_CCSRC_MINDDATA_DATASET_KERNELS_IMAGE_RAND_AUGMENT_OP_H_
