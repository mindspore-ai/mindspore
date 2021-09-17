/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
#include "minddata/dataset/audio/kernels/frequency_masking_op.h"

#include "minddata/dataset/audio/kernels/audio_utils.h"
#include "minddata/dataset/kernels/data/data_utils.h"
#include "minddata/dataset/util/random.h"
#include "minddata/dataset/util/status.h"

namespace mindspore {
namespace dataset {
// constructor
FrequencyMaskingOp::FrequencyMaskingOp(bool iid_masks, int32_t frequency_mask_param, int32_t mask_start,
                                       float mask_value)
    : frequency_mask_param_(frequency_mask_param),
      mask_start_(mask_start),
      iid_masks_(iid_masks),
      mask_value_(mask_value) {
  rnd_.seed(GetSeed());
}

// main function
Status FrequencyMaskingOp::Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) {
  IO_CHECK(input, output);
  // input <..., freq, time>
  CHECK_FAIL_RETURN_UNEXPECTED(input->Rank() >= 2,
                               "FrequencyMasking: input tensor is not in shape of <..., freq, time>.");
  TensorShape input_shape = input->shape();
  CHECK_FAIL_RETURN_UNEXPECTED(
    input_shape[-2] >= frequency_mask_param_,
    "FrequencyMasking: frequency_mask_param should be less than or equal to the length of frequency dimension.");

  std::shared_ptr<Tensor> input_tensor;
  // typecast
  CHECK_FAIL_RETURN_UNEXPECTED(input->type() != DataType::DE_STRING,
                               "FrequencyMasking: input tensor type should be float, but got: string.");
  if (input->type() != DataType::DE_FLOAT64) {
    RETURN_IF_NOT_OK(TypeCast(input, &input_tensor, DataType(DataType::DE_FLOAT32)));
  } else {
    input_tensor = input;
  }
  // iid_masks - whether to apply different masks to each example/channel.
  if (iid_masks_ == false) {
    return MaskAlongAxis(input_tensor, output, frequency_mask_param_, mask_start_, mask_value_, 1);
  } else {
    return RandomMaskAlongAxis(input_tensor, output, frequency_mask_param_, mask_value_, 1, rnd_);
  }
}
}  // namespace dataset
}  // namespace mindspore
