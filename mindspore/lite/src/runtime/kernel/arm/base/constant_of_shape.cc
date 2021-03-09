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

#include "src/runtime/kernel/arm/base/constant_of_shape.h"
#include "schema/model_generated.h"
#include "src/kernel_registry.h"
#include "src/runtime/runtime_api.h"

using mindspore::kernel::KERNEL_ARCH::kCPU;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_ConstantOfShape;

namespace mindspore::kernel {
int ConstantOfShapeRun(void *cdata, int task_id) {
  auto g_kernel = reinterpret_cast<ConstantOfShapeCPUKernel *>(cdata);
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
#ifdef ENABLE_NEON
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
  auto output = out_tensors_.front();
  param_->data_type_ = output->data_type();
  param_->element_size_ = output->ElementsNum();
  output_ptr_ = output->data_c();
  int thread_count = MSMIN(op_parameter_->thread_num_, param_->element_size_);
  thread_stride_ = UP_DIV(param_->element_size_, thread_count);

  auto ret = ParallelLaunch(this->context_->thread_pool_, ConstantOfShapeRun, this, thread_count);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "ConstantOfShapeRun error error_code[" << ret << "]";
    return ret;
  }
  return ret;
}

REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_ConstantOfShape, LiteKernelCreator<ConstantOfShapeCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeInt32, PrimitiveType_ConstantOfShape, LiteKernelCreator<ConstantOfShapeCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeInt64, PrimitiveType_ConstantOfShape, LiteKernelCreator<ConstantOfShapeCPUKernel>)
}  // namespace mindspore::kernel
