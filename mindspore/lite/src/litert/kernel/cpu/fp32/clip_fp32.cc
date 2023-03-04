/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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
#include "src/litert/kernel/cpu/fp32/clip_fp32.h"
#include "schema/model_generated.h"
#include "src/litert/kernel_registry.h"
#include "nnacl/fp32/activation_fp32.h"

using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_NULL_PTR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_Clip;

namespace mindspore::kernel {
namespace {
int GetMinMaxValue(const lite::Tensor *tensor, float *data) {
  MS_ASSERT(tensor != nullptr && data != nullptr);
  MS_CHECK_TRUE_RET(tensor->data() != nullptr, lite::RET_ERROR);
  switch (tensor->data_type()) {
    case kNumberTypeFloat:
    case kNumberTypeFloat32:
      *data = *(reinterpret_cast<float *>(tensor->data()));
      break;
    case kNumberTypeInt:
    case kNumberTypeInt32:
      *data = *(reinterpret_cast<int *>(tensor->data()));
      break;
    default:
      MS_LOG(ERROR) << "Unsupported data type: " << tensor->data();
      return lite::RET_ERROR;
  }
  return lite::RET_OK;
}
}  // namespace

int ClipCPUKernel::Prepare() {
  CHECK_LESS_RETURN(in_tensors_.size(), 1);
  CHECK_NOT_EQUAL_RETURN(out_tensors_.size(), 1);
  if (!InferShapeDone()) {
    return RET_OK;
  }
  return ReSize();
}

int ClipCPUKernel::ReSize() {
  if (UpdateThreadNumPass(TC_PTYPE(PrimitiveType_Clip), 1, 1, out_tensors_.at(0)->ElementsNum()) != RET_OK) {
    return RET_ERROR;
  }
  return RET_OK;
}

int ClipCPUKernel::DoClip(int task_id) {
  auto input_addr = in_tensors_.at(0)->MutableData();
  auto output_addr = out_tensors_.at(0)->MutableData();
  MS_ASSERT(input_addr != nullptr && output_addr != nullptr);

  auto length = in_tensors_.at(0)->ElementsNum();
  int stride = UP_DIV(length, thread_num_);
  int count = MSMIN(stride, length - stride * task_id);
  if (count <= 0) {
    return RET_OK;
  }
  if (INT_MUL_OVERFLOW(stride, task_id)) {
    return RET_ERROR;
  }

  auto ret = RET_OK;
  switch (in_tensors_.at(0)->data_type()) {
    case kNumberTypeFloat:
    case kNumberTypeFloat32: {
      ret = Fp32Clip(reinterpret_cast<float *>(input_addr) + stride * task_id, count,
                     reinterpret_cast<float *>(output_addr) + stride * task_id, min_val_, max_val_);
    } break;
    case kNumberTypeInt:
    case kNumberTypeInt32: {
      ret = Int32Clip(reinterpret_cast<int *>(input_addr) + stride * task_id, count,
                      reinterpret_cast<int *>(output_addr) + stride * task_id, static_cast<int>(min_val_),
                      static_cast<int>(max_val_));
    } break;
    default:
      MS_LOG(ERROR) << "Unsupported data type: " << in_tensors_.at(0)->data_type();
      return RET_ERROR;
  }

  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Fp32 Activation error, ret: " << ret;
  }
  return ret;
}

int ClipRun(void *cdata, int task_id, float lhs_scale, float rhs_scale) {
  CHECK_NULL_RETURN(cdata);
  auto clip_kernel = reinterpret_cast<ClipCPUKernel *>(cdata);
  MS_ASSERT(clip_kernel != nullptr);
  auto error_code = clip_kernel->DoClip(task_id);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "ActivationRun error task_id[" << task_id << "] error_code[" << error_code << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int ClipCPUKernel::Run() {
  auto param = reinterpret_cast<ClipParameter *>(op_parameter_);
  MS_CHECK_TRUE_RET(param != nullptr, RET_ERROR);
  auto ret = RET_OK;
  if (in_tensors_.size() > 1) {
    auto min_tensor = in_tensors_.at(1);
    MS_CHECK_TRUE_RET(min_tensor != nullptr && min_tensor->data() != nullptr, RET_ERROR);
    ret = GetMinMaxValue(min_tensor, &(param->min_val_));
  }
  if (in_tensors_.size() > kInputSize1) {
    auto max_tensor = in_tensors_.at(kInputSize1);
    MS_CHECK_TRUE_RET(max_tensor != nullptr && max_tensor->data() != nullptr, RET_ERROR);
    ret = GetMinMaxValue(max_tensor, &(param->max_val_));
  }
  if (ret != RET_OK || param->min_val_ >= param->max_val_) {
    MS_LOG(ERROR) << "Update min value or max value failed.";
    return RET_ERROR;
  }

  ret = ParallelLaunch(this->ms_context_, ClipRun, this, thread_num_);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Crop launch fail!ret: " << ret;
    return RET_ERROR;
  }
  return RET_OK;
}

REG_KERNEL(kCPU, kNumberTypeFloat, PrimitiveType_Clip, LiteKernelCreator<ClipCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_Clip, LiteKernelCreator<ClipCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeInt, PrimitiveType_Clip, LiteKernelCreator<ClipCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeInt32, PrimitiveType_Clip, LiteKernelCreator<ClipCPUKernel>)
}  // namespace mindspore::kernel
