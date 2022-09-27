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

#include "src/litert/kernel/cpu/fp32_grad/arithmetic_self_grad.h"
#include "schema/model_generated.h"
#include "src/litert/kernel_registry.h"
#include "include/errorcode.h"
#include "nnacl/fp32/arithmetic_fp32.h"
#include "nnacl/fp32_grad/arithmetic_grad.h"
#include "plugin/device/cpu/kernel/nnacl/op_base.h"

using mindspore::kernel::KERNEL_ARCH;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_AbsGrad;
using mindspore::schema::PrimitiveType_LogGrad;
using mindspore::schema::PrimitiveType_RsqrtGrad;
using mindspore::schema::PrimitiveType_SqrtGrad;

namespace mindspore::kernel {
namespace {
constexpr static int kDyIdx = 0;
constexpr static int kXIdx = 1;
constexpr static int kDxIdx = 0;

int ArithmeticSelfGradRun(void *cdata, int thread_id, float lhs_scale, float rhs_scale) {
  auto kernel = reinterpret_cast<ArithmeticSelfGradCPUKernel *>(cdata);
  CHECK_NULL_RETURN(kernel);
  return kernel->DoArithmeticSelfGrad(thread_id);
}
}  // namespace

int ArithmeticSelfGradCPUKernel::Prepare() {
  auto type = this->type();
  switch (type) {
    case PrimitiveType_LogGrad:
      self_grad_operation_ = ElementDiv;
      break;
    case PrimitiveType_AbsGrad:
      self_grad_operation_ = ElementAbsGrad;
      break;
    case PrimitiveType_SqrtGrad:
      self_grad_operation_ = ElementSqrtGrad;
      break;
    case PrimitiveType_RsqrtGrad:
      self_grad_operation_ = ElementRsqrtGrad;
      break;
    default:
      MS_LOG(ERROR) << "Unsupported type: " << type;
      return RET_ERROR;
  }
  return RET_OK;
}

int ArithmeticSelfGradCPUKernel::DoArithmeticSelfGrad(int task_id) {
  CHECK_LESS_RETURN(in_tensors_.size(), DIMENSION_2D);
  CHECK_LESS_RETURN(out_tensors_.size(), DIMENSION_1D);
  CHECK_NULL_RETURN(in_tensors_.at(kDyIdx));
  CHECK_NULL_RETURN(in_tensors_.at(kXIdx));
  CHECK_NULL_RETURN(out_tensors_.at(kDxIdx));
  auto dy = reinterpret_cast<float *>(in_tensors_.at(kDyIdx)->MutableData());
  auto in_x = reinterpret_cast<float *>(in_tensors_.at(kXIdx)->MutableData());
  auto dx = reinterpret_cast<float *>(out_tensors_.at(kDxIdx)->MutableData());
  int length = in_tensors_.at(kDyIdx)->ElementsNum();
  CHECK_NULL_RETURN(dy);
  CHECK_NULL_RETURN(in_x);
  CHECK_NULL_RETURN(dx);

  int stride = UP_DIV(length, thread_count_);
  int count = MSMIN(stride, length - stride * task_id);
  int start = stride * task_id;

  if (count > 0) {
    (*self_grad_operation_)(dy + start, in_x + start, dx + start, count);
  }
  return RET_OK;
}

int ArithmeticSelfGradCPUKernel::ReSize() { return RET_OK; }

int ArithmeticSelfGradCPUKernel::Run() {
  auto ret = ParallelLaunch(this->ms_context_, ArithmeticSelfGradRun, this, thread_count_);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "parallel launch fail!ret: " << ret;
    return ret;
  }

  return RET_OK;
}

kernel::LiteKernel *CpuArithmeticSelfGradFp32KernelCreator(const std::vector<lite::Tensor *> &inputs,
                                                           const std::vector<lite::Tensor *> &outputs,
                                                           OpParameter *param, const lite::InnerContext *ctx,
                                                           const kernel::KernelKey &desc) {
  if (param == nullptr) {
    MS_LOG(ERROR) << "input parameter is nullptr!";
    return nullptr;
  }
  auto *kernel = new (std::nothrow) ArithmeticSelfGradCPUKernel(param, inputs, outputs, ctx);
  if (kernel == nullptr) {
    MS_LOG(ERROR) << "new ArithmeticSelfGradCPUKernel fail!";
    free(param);
    return nullptr;
  }
  return kernel;
}

REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_LogGrad, CpuArithmeticSelfGradFp32KernelCreator)
REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_AbsGrad, CpuArithmeticSelfGradFp32KernelCreator)
REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_SqrtGrad, CpuArithmeticSelfGradFp32KernelCreator)
REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_RsqrtGrad, CpuArithmeticSelfGradFp32KernelCreator)
}  // namespace mindspore::kernel
