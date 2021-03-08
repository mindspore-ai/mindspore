/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
#include "src/runtime/kernel/arm/base/crop_base.h"
#include <vector>
#include "src/runtime/kernel/arm/fp32/crop_fp32.h"
#include "schema/model_generated.h"
#include "src/kernel_registry.h"
#include "include/errorcode.h"
#include "include/context.h"

using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_Crop;

namespace mindspore::kernel {
int CropBaseCPUKernel::Init() { return RET_OK; }

int CropBaseCPUKernel::ReSize() {
  auto *input_tensor = in_tensors_.at(kInputIndex);
  auto *out_tensor = out_tensors_.at(kOutputIndex);
  input_shape_ = input_tensor->shape();
  output_shape_ = out_tensor->shape();
  size_t input_dim = input_shape_.size();
  crop_para_->in_shape_ = input_shape_.data();
  crop_para_->out_shape_ = output_shape_.data();
  MS_ASSERT(input_dim <= CROP_OFFSET_MAX_SIZE);
  crop_para_->input_dim_ = input_dim;
  PadOffset(input_dim, crop_para_);
  return RET_OK;
}

void CropBaseCPUKernel::PadOffset(int input_dim, CropParameter *crop_para) {
  auto axis = crop_para->axis_;
  auto offsets_size = crop_para->offset_size_;
  MS_ASSERT(axis <= input_dim);
  if (offsets_size > 1) {
    MS_ASSERT(axis + offsets_size == input_dim);
  }
  for (int i = 0; i < input_dim; i++) {
    int crop_offset = 0;
    if (i >= axis) {
      if (offsets_size == 1) {
        crop_offset = crop_para->offset_[0];
      } else if (offsets_size > 1) {
        crop_offset = crop_para->offset_[i - axis];
      }
    }
    crop_para->in_offset_[i] = crop_offset;
  }
}
}  // namespace mindspore::kernel
