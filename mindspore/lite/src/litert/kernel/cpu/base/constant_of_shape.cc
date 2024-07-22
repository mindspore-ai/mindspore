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

#include "src/litert/kernel/cpu/base/constant_of_shape.h"
#include "schema/model_generated.h"
#include "src/litert/kernel_registry.h"

using mindspore::kernel::KERNEL_ARCH;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_NULL_PTR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_ConstantOfShape;

namespace mindspore::kernel {
int ConstantOfShapeRun(void *cdata, int task_id, float lhs_scale, float rhs_scale) {
  auto g_kernel = reinterpret_cast<ConstantOfShapeCPUKernel *>(cdata);
  CHECK_NULL_RETURN(g_kernel);
  auto ret = g_kernel->DoExecute(task_id);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "ConstantOfShapeRun error task_id[" << task_id << "] error_code[" << ret << "]";
    return ret;
  }
  return RET_OK;
}

int ConstantOfShapeCPUKernel::DoExecute(int task_id) {
  int start = task_id * thread_stride_;
  int current_stride = MSMIN(thread_stride_, param_->element_size_ - start);
  if (current_stride < 0) {
    return RET_OK;
  }

  switch (param_->data_type_) {
    case kNumberTypeFloat32:
      ConstantOfShapeFp32(reinterpret_cast<float *>(output_ptr_), start, start + current_stride,
                          param_->value_.f32_value_);
      break;
    case kNumberTypeInt32:
      ConstantOfShapeInt32(reinterpret_cast<int32_t *>(output_ptr_), start, start + current_stride,
                           param_->value_.int32_value_);
      break;
    case kNumberTypeBool:
      ConstantOfShapeBool(reinterpret_cast<bool *>(output_ptr_), start, start + current_stride,
                          param_->value_.bool_value_);
      break;
#ifdef ENABLE_FP16
    case kNumberTypeFloat16:
      ConstantOfShapeFp16(reinterpret_cast<float16_t *>(output_ptr_), start, start + current_stride,
                          param_->value_.f32_value_);
      break;
#endif
    default:
      MS_LOG(ERROR) << "Invalid datatype in ConstantOfShapeRun";
      return RET_ERROR;
  }
  return RET_OK;
}

int ConstantOfShapeCPUKernel::Run() {
  CHECK_LESS_RETURN(out_tensors_.size(), 1);
  auto output = out_tensors_.front();
  CHECK_NULL_RETURN(output);
  CHECK_NULL_RETURN(param_);
  param_->data_type_ = output->data_type();
  param_->element_size_ = output->ElementsNum();
  if (param_->element_size_ == 0) {
    MS_LOG(WARNING) << "ConstantOfShape's output element number is 0, it will output a tensor without any data.";
    return RET_OK;
  }
  MS_CHECK_GT(param_->element_size_, 0, RET_ERROR);
  output_ptr_ = output->data();
  CHECK_NULL_RETURN(output_ptr_);

  int thread_count = MSMIN(op_parameter_->thread_num_, param_->element_size_);
  if (thread_count == 0) {
    MS_LOG(ERROR) << "div zero";
    return RET_ERROR;
  }
  thread_stride_ = UP_DIV(param_->element_size_, thread_count);

  auto ret = ParallelLaunch(this->ms_context_, ConstantOfShapeRun, this, thread_count);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "ConstantOfShapeRun error error_code[" << ret << "]";
    return ret;
  }
  return ret;
}

REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_ConstantOfShape, LiteKernelCreator<ConstantOfShapeCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeFloat16, PrimitiveType_ConstantOfShape, LiteKernelCreator<ConstantOfShapeCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeInt32, PrimitiveType_ConstantOfShape, LiteKernelCreator<ConstantOfShapeCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeInt64, PrimitiveType_ConstantOfShape, LiteKernelCreator<ConstantOfShapeCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeBool, PrimitiveType_ConstantOfShape, LiteKernelCreator<ConstantOfShapeCPUKernel>)
}  // namespace mindspore::kernel
