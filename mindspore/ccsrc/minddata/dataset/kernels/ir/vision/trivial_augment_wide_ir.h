/**
 * Copyright 2022-2023 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_KERNELS_IR_VISION_TRIVIAL_AUGMENT_WIDE_IR_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_KERNELS_IR_VISION_TRIVIAL_AUGMENT_WIDE_IR_H_

#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "include/api/status.h"
#include "minddata/dataset/include/dataset/constants.h"
#include "minddata/dataset/include/dataset/transforms.h"
#include "minddata/dataset/kernels/ir/tensor_operation.h"

namespace mindspore {
namespace dataset {
namespace vision {
constexpr char kTrivialAugmentWideOperation[] = "TrivialAugmentWide";

class TrivialAugmentWideOperation : public TensorOperation {
 public:
  TrivialAugmentWideOperation(int32_t num_magnitude_bins, InterpolationMode interpolation,
                              const std::vector<uint8_t> &fill_value);

  ~TrivialAugmentWideOperation() override;

  std::shared_ptr<TensorOp> Build() override;

  Status ValidateParams() override;

  std::string Name() const override;

  Status to_json(nlohmann::json *out_json) override;

  static Status from_json(nlohmann::json op_params, std::shared_ptr<TensorOperation> *operation);

 private:
  int32_t num_magnitude_bins_;
  InterpolationMode interpolation_;
  std::vector<uint8_t> fill_value_;
};
}  // namespace vision
}  // namespace dataset
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_KERNELS_IR_VISION_TRIVIAL_AUGMENT_WIDE_IR_H_
