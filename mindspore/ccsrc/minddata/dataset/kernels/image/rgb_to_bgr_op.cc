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
#include "minddata/dataset/kernels/image/rgb_to_bgr_op.h"
#ifndef ENABLE_ANDROID
#include "minddata/dataset/kernels/image/image_utils.h"
#else
#include "minddata/dataset/kernels/image/lite_image_utils.h"
#endif

namespace mindspore {
namespace dataset {
Status RgbToBgrOp::Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) {
  IO_CHECK(input, output);
  auto input_type = input->type();
  CHECK_FAIL_RETURN_UNEXPECTED(input_type != DataType::DE_UINT32 && input_type != DataType::DE_UINT64 &&
                                 input_type != DataType::DE_INT64 && input_type != DataType::DE_STRING,
                               "RgbToBgr: Input includes unsupported data type in [uint32, int64, uint64, string].");

  return RgbToBgr(input, output);
}
}  // namespace dataset
}  // namespace mindspore
