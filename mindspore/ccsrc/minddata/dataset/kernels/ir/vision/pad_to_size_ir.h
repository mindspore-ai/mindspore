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

#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_KERNELS_IR_VISION_PAD_TO_SIZE_IR_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_KERNELS_IR_VISION_PAD_TO_SIZE_IR_H_

#include <memory>
#include <string>
#include <vector>

#include "minddata/dataset/include/dataset/constants.h"
#include "minddata/dataset/kernels/ir/tensor_operation.h"

namespace mindspore {
namespace dataset {
namespace vision {
constexpr char kPadToSizeOperation[] = "PadToSize";

class PadToSizeOperation : public TensorOperation {
 public:
  PadToSizeOperation(const std::vector<int32_t> &size, const std::vector<int32_t> &offset,
                     const std::vector<uint8_t> &fill_value, BorderType padding_mode);

  ~PadToSizeOperation() override;

  std::shared_ptr<TensorOp> Build() override;

  Status ValidateParams() override;

  std::string Name() const override;

  Status to_json(nlohmann::json *out_json) override;

  static Status from_json(nlohmann::json op_params, std::shared_ptr<TensorOperation> *operation);

 private:
  std::vector<int32_t> size_;
  std::vector<int32_t> offset_;
  std::vector<uint8_t> fill_value_;
  BorderType padding_mode_;
};
}  // namespace vision
}  // namespace dataset
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_KERNELS_IR_VISION_PAD_TO_SIZE_IR_H_
