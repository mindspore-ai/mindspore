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
#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_KERNELS_IMAGE_PAD_TO_SIZE_OP_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_KERNELS_IMAGE_PAD_TO_SIZE_OP_H_

#include <memory>
#include <string>
#include <vector>

#include "minddata/dataset/kernels/tensor_op.h"

namespace mindspore {
namespace dataset {
class PadToSizeOp : public TensorOp {
 public:
  PadToSizeOp(std::vector<int32_t> size, std::vector<int32_t> offset, std::vector<uint8_t> fill_value,
              BorderType padding_mode);

  ~PadToSizeOp() override = default;

  Status Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) override;

  Status OutputShape(const std::vector<TensorShape> &inputs, std::vector<TensorShape> &outputs) override;

  std::string Name() const override { return kPadToSizeOp; }

 private:
  std::vector<int32_t> size_;
  std::vector<int32_t> offset_;
  std::vector<uint8_t> fill_value_;
  BorderType boarder_type_;
};
}  // namespace dataset
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_KERNELS_IMAGE_PAD_TO_SIZE_OP_H_
