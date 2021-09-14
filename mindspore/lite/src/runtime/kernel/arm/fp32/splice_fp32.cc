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

#include "src/runtime/kernel/arm/fp32/splice_fp32.h"
#include "schema/model_generated.h"
#include "src/kernel_registry.h"
#include "include/errorcode.h"
#include "src/common/log_adapter.h"
#include "nnacl/fp32/splice_fp32.h"

using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::lite::RET_PARAM_INVALID;
using mindspore::schema::PrimitiveType_Splice;
namespace mindspore::kernel {
int SpliceCPUKernel::Init() {
  CHECK_LESS_RETURN(in_tensors_.size(), 1);
  CHECK_LESS_RETURN(out_tensors_.size(), 1);
  CHECK_NULL_RETURN(parameter_);
  return RET_OK;
}

int SpliceCPUKernel::ReSize() { return RET_OK; }

int SpliceCPUKernel::Run() {
  lite::Tensor *input_tensor = in_tensors_.front();
  CHECK_NULL_RETURN(input_tensor);
  lite::Tensor *output_tensor = out_tensors_.front();
  CHECK_NULL_RETURN(output_tensor);
  std::vector<int> src_shape = input_tensor->shape();
  std::vector<int> dst_shape = output_tensor->shape();
  if (src_shape.size() != dst_shape.size() || src_shape.size() != DIMENSION_3D || dst_shape.size() != DIMENSION_3D) {
    MS_LOG(ERROR) << "splice kernel src_shape size not equal to dst_shape size";
    return RET_ERROR;
  }
  int src_row = src_shape.at(DIMENSION_1D);
  int dst_row = dst_shape.at(DIMENSION_1D);
  int src_col = src_shape.at(DIMENSION_2D);
  int dst_col = dst_shape.at(DIMENSION_2D);
  if (src_col * parameter_->context_dim_ != dst_col) {
    MS_LOG(ERROR) << "splice kernel src_col not match dst_col";
    return RET_ERROR;
  }
  if (parameter_->context_dim_ * dst_row != parameter_->forward_indexes_dim_) {
    MS_LOG(ERROR) << "splice kernel param not match dst_row";
    return RET_PARAM_INVALID;
  }
  for (int i = 0; i < parameter_->forward_indexes_dim_; ++i) {
    if (parameter_->forward_indexes_[i] >= src_row) {
      MS_LOG(ERROR) << "splice kernel param not match dst_row";
      return RET_PARAM_INVALID;
    }
  }
  auto input_data = reinterpret_cast<float *>(input_tensor->data());
  CHECK_NULL_RETURN(input_data);
  auto output_data = reinterpret_cast<float *>(output_tensor->data());
  CHECK_NULL_RETURN(output_data);
  SpliceFp32(input_data, src_row, src_col, parameter_, output_data, dst_row, dst_col);
  return RET_OK;
}
REG_KERNEL(kCPU, kNumberTypeFloat, PrimitiveType_Splice, LiteKernelCreator<SpliceCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_Splice, LiteKernelCreator<SpliceCPUKernel>)
}  // namespace mindspore::kernel
