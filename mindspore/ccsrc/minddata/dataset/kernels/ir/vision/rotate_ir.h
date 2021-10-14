/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_KERNELS_IR_VISION_ROTATE_IR_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_KERNELS_IR_VISION_ROTATE_IR_H_

#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "include/api/status.h"
#include "minddata/dataset/include/dataset/constants.h"
#include "minddata/dataset/include/dataset/transforms.h"
#include "minddata/dataset/kernels/ir/tensor_operation.h"

namespace mindspore {
namespace dataset {

namespace vision {

constexpr char kRotateOperation[] = "Rotate";

class RotateOperation : public TensorOperation {
 public:
  explicit RotateOperation(FixRotationAngle angle);

  RotateOperation(float degrees, InterpolationMode resample, bool expand, const std::vector<float> &center,
                  const std::vector<uint8_t> &fill_value);

  ~RotateOperation();

  std::shared_ptr<TensorOp> Build() override;

  Status ValidateParams() override;

  std::string Name() const override;

  Status to_json(nlohmann::json *out_json) override;

  static Status from_json(nlohmann::json op_params, std::shared_ptr<TensorOperation> *operation);

  void setAngle(uint64_t angle_id);

 private:
  uint64_t angle_id_;
  float degrees_;
  InterpolationMode interpolation_mode_;
  bool expand_;
  std::vector<float> center_;
  std::vector<uint8_t> fill_value_;
  std::shared_ptr<TensorOp> rotate_op_;
};

}  // namespace vision
}  // namespace dataset
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_KERNELS_IR_VISION_ROTATE_IR_H_
