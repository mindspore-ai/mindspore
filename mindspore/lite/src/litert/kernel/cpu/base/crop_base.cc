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
#include "src/litert/kernel/cpu/base/crop_base.h"
#include <vector>
#include "src/litert/kernel/cpu/fp32/crop_fp32.h"
#include "schema/model_generated.h"
#include "src/litert/kernel_registry.h"
#include "include/errorcode.h"

using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_Crop;

namespace mindspore::kernel {
int CropBaseCPUKernel::Prepare() { return RET_OK; }

int CropBaseCPUKernel::ReSize() {
  // check in_tensors size
  CHECK_LESS_RETURN(in_tensors_.size(), kInputSize1);
  // check out_tensors size
  CHECK_LESS_RETURN(out_tensors_.size(), 1);
  auto *input_tensor = in_tensors_.at(kInputIndex);
  CHECK_NULL_RETURN(input_tensor);
  auto *crop_tensor = in_tensors_.at(SECOND_INPUT);
  CHECK_NULL_RETURN(crop_tensor);
  auto *out_tensor = out_tensors_.at(kOutputIndex);
  CHECK_NULL_RETURN(out_tensor);

  input_shape_ = input_tensor->shape();
  CHECK_NULL_RETURN(input_shape_.data());
  crop_shape_ = crop_tensor->shape();
  CHECK_NULL_RETURN(crop_shape_.data());
  output_shape_ = out_tensor->shape();
  CHECK_NULL_RETURN(output_shape_.data());
  size_t input_dim = input_shape_.size();
  CHECK_NULL_RETURN(crop_para_);
  crop_para_->in_shape_ = input_shape_.data();
  crop_para_->out_shape_ = output_shape_.data();
  MS_CHECK_LE(input_dim, CROP_OFFSET_MAX_SIZE, RET_ERROR);
  crop_para_->input_dim_ = static_cast<int>(input_dim);
  if (PadOffset(input_dim, crop_para_) != RET_OK) {
    MS_LOG(ERROR) << "Pad offset failed.";
    return RET_ERROR;
  }
  return RET_OK;
}

int CropBaseCPUKernel::PadOffset(int input_dim, CropParameter *crop_para) const {
  auto axis = crop_para->axis_;
  auto offsets_size = crop_para->offset_size_;
  MS_CHECK_TRUE_MSG(axis < input_dim, RET_ERROR, "The axis is invalid.");
  if (offsets_size > 1) {
    MS_CHECK_TRUE_MSG(axis + offsets_size == input_dim, RET_ERROR, "The axis and offsets is invalid");
  }
  for (int i = 0; i < input_dim; i++) {
    int crop_offset = 0;
    if (i >= axis) {
      if (offsets_size == 1) {
        crop_offset = crop_para->offset_[0];
      } else if (offsets_size > 1) {
        if (i - axis < CROP_OFFSET_MAX_SIZE) {
          crop_offset = crop_para->offset_[i - axis];
        }
      }
      MS_CHECK_GE(input_shape_[i] - crop_offset, crop_shape_[i], RET_ERROR);
    }
    crop_para->in_offset_[i] = crop_offset;
  }
  return RET_OK;
}
}  // namespace mindspore::kernel
