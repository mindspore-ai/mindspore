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

#include "src/litert/kernel/cpu/fp32_grad/arithmetic_grad.h"
#include "schema/model_generated.h"
#include "src/litert/kernel_registry.h"
#include "nnacl/fp32_grad/reduce_grad.h"
#include "nnacl/fp32_grad/arithmetic_grad.h"
#include "include/errorcode.h"
#include "plugin/device/cpu/kernel/nnacl/op_base.h"

using mindspore::kernel::KERNEL_ARCH;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;

namespace mindspore::kernel {
constexpr static int kDyIdx = 0;
constexpr static int kX1Idx = 1;
constexpr static int kX2Idx = 2;
constexpr static int kDx1Idx = 0;
constexpr static int kDx2Idx = 1;

int ArithmeticGradCPUKernel::Prepare() {
  CHECK_LESS_RETURN(in_tensors_.size(), DIMENSION_3D);
  CHECK_NULL_RETURN(in_tensors_.at(kDyIdx));
  CHECK_NULL_RETURN(in_tensors_.at(kX1Idx));
  CHECK_NULL_RETURN(in_tensors_.at(kX2Idx));
  CHECK_LESS_RETURN(out_tensors_.size(), DIMENSION_2D);
  auto dx1 = out_tensors_[kDx1Idx];
  auto dx2 = out_tensors_[kDx2Idx];
  CHECK_NULL_RETURN(dx1);
  CHECK_NULL_RETURN(dx2);
  CHECK_NULL_RETURN(arithmeticParameter_);

  if ((type() == PrimitiveType_MulGrad) || (type() == PrimitiveType_DivGrad)) {
    if (dx1->ElementsNum() < dx2->ElementsNum()) {
      if (type() == PrimitiveType_MulGrad)
        arithmetic_grad_ = &ArithmeticGradCPUKernel::ArithmeticGradMul2L;
      else if (type() == PrimitiveType_DivGrad)
        arithmetic_grad_ = &ArithmeticGradCPUKernel::ArithmeticGradDiv2L;
    } else if (dx2->ElementsNum() < dx1->ElementsNum()) {
      if (type() == PrimitiveType_MulGrad)
        arithmetic_grad_ = &ArithmeticGradCPUKernel::ArithmeticGradMul1L;
      else if (type() == PrimitiveType_DivGrad)
        arithmetic_grad_ = &ArithmeticGradCPUKernel::ArithmeticGradDiv1L;
    }

    tile_data0 = new (std::nothrow) float[in_tensors_.at(0)->ElementsNum()];
    if (tile_data0 == nullptr) {
      MS_LOG(ERROR) << "new data0 fail!";
      return RET_ERROR;
    }
    tile_data1 = new (std::nothrow) float[in_tensors_.at(0)->ElementsNum()];
    if (tile_data1 == nullptr) {
      MS_LOG(ERROR) << "new data1 fail!";
      return RET_ERROR;
    }

    if (type() == PrimitiveType_DivGrad) {
      tile_data2 = new (std::nothrow) float[in_tensors_.at(0)->ElementsNum()];
      if (tile_data2 == nullptr) {
        MS_LOG(ERROR) << "new data2 fail!";
        return RET_ERROR;
      }
    }
  }

  return RET_OK;
}

int ArithmeticGradCPUKernel::ArithmeticGradAdd(float *dy, int dy_size, float *dx1, int dx1_size, float *dx2,
                                               int dx2_size) {
  if (dx1_size == dy_size) {
    memcpy(dx1, dy, static_cast<size_t>(dy_size) * sizeof(float));
  } else {
    ReduceSumByAxes(dy, arithmeticParameter_->out_shape_, dx1, arithmeticParameter_->in_shape0_,
                    arithmeticParameter_->ndim_);
  }
  if (dx2_size == dy_size) {
    memcpy(dx2, dy, static_cast<size_t>(dy_size) * sizeof(float));
  } else {
    ReduceSumByAxes(dy, arithmeticParameter_->out_shape_, dx2, arithmeticParameter_->in_shape1_,
                    arithmeticParameter_->ndim_);
  }
  return RET_OK;
}

int ArithmeticGradCPUKernel::ArithmeticGradSub(float *dy, int dy_size, float *dx1, int dx1_size, float *dx2,
                                               int dx2_size) {
  if (dx1_size == dy_size) {
    memcpy(dx1, dy, static_cast<size_t>(dy_size) * sizeof(float));
  } else {
    ReduceSumByAxes(dy, arithmeticParameter_->out_shape_, dx1, arithmeticParameter_->in_shape0_,
                    arithmeticParameter_->ndim_);
  }
  if (dx2_size == dy_size) {
    for (int i = 0; i < dx2_size; i++) {
      dx2[i] = -dy[i];
    }
  } else {
    ReduceSumByAxes(dy, arithmeticParameter_->out_shape_, dx2, arithmeticParameter_->in_shape1_,
                    arithmeticParameter_->ndim_);
    for (int i = 0; i < dx2_size; i++) {
      dx2[i] = -dx2[i];
    }
  }
  return RET_OK;
}

int ArithmeticGradCPUKernel::ArithmeticGradMul(float *dy, int dy_size, float *dx1, int dx1_size, float *dx2,
                                               int dx2_size) {
  auto x1_data = reinterpret_cast<float *>(in_tensors_[kX1Idx]->MutableData());
  auto x2_data = reinterpret_cast<float *>(in_tensors_[kX2Idx]->MutableData());
  CHECK_NULL_RETURN(x1_data);
  CHECK_NULL_RETURN(x2_data);
  ElementMul(dy, x1_data, dx2, dy_size);
  ElementMul(dy, x2_data, dx1, dy_size);
  return RET_OK;
}

int ArithmeticGradCPUKernel::ArithmeticGradMul1L(float *dy, int dy_size, float *dx1, int dx1_size, float *dx2,
                                                 int dx2_size) {
  auto x1_data = reinterpret_cast<float *>(in_tensors_[kX1Idx]->MutableData());
  auto x2_data = reinterpret_cast<float *>(in_tensors_[kX2Idx]->MutableData());
  CHECK_NULL_RETURN(x1_data);
  CHECK_NULL_RETURN(x2_data);
  ElementMul(dy, x1_data, tile_data0, dy_size);
  ReduceSumByAxes(tile_data0, arithmeticParameter_->in_shape0_, dx2, arithmeticParameter_->in_shape1_,
                  arithmeticParameter_->ndim_);

  BroadcastMul(dy, x2_data, tile_data0, tile_data1, dx1, dy_size, arithmeticParameter_);  // broadcast directly to dx1
  return RET_OK;
}

int ArithmeticGradCPUKernel::ArithmeticGradMul2L(float *dy, int dy_size, float *dx1, int dx1_size, float *dx2,
                                                 int dx2_size) {
  auto x1_data = reinterpret_cast<float *>(in_tensors_[kX1Idx]->MutableData());
  auto x2_data = reinterpret_cast<float *>(in_tensors_[kX2Idx]->MutableData());
  CHECK_NULL_RETURN(x1_data);
  CHECK_NULL_RETURN(x2_data);
  ElementMul(dy, x2_data, tile_data0, dy_size);
  ReduceSumByAxes(tile_data0, arithmeticParameter_->in_shape0_, dx1, arithmeticParameter_->in_shape1_,
                  arithmeticParameter_->ndim_);

  BroadcastMul(dy, x1_data, tile_data0, tile_data1, dx2, dy_size, arithmeticParameter_);  // broadcast directly to dx2
  return RET_OK;
}

int ArithmeticGradCPUKernel::ArithmeticGradDiv(float *dy, int dy_size, float *dx1, int dx1_size, float *dx2,
                                               int dx2_size) {
  auto x1 = reinterpret_cast<float *>(in_tensors_[kX1Idx]->MutableData());
  auto x2 = reinterpret_cast<float *>(in_tensors_[kX2Idx]->MutableData());
  CHECK_NULL_RETURN(x1);
  CHECK_NULL_RETURN(x2);
  ElementDiv(dy, x2, dx1, dy_size);
  ElementMulAndDivNegSquare(dy, x1, x2, dx2, dy_size);
  return RET_OK;
}

int ArithmeticGradCPUKernel::ArithmeticGradDiv1L(float *dy, int dy_size, float *dx1, int dx1_size, float *dx2,
                                                 int dx2_size) {
  auto x1_data = reinterpret_cast<float *>(in_tensors_[kX1Idx]->MutableData());
  auto x2_data = reinterpret_cast<float *>(in_tensors_[kX2Idx]->MutableData());
  CHECK_NULL_RETURN(x1_data);
  CHECK_NULL_RETURN(x2_data);

  ElementMul(x2_data, x2_data, dx2, dx2_size);
  ElementMul(x1_data, dy, dx1, dy_size);  // use dx1 buffer
  BroadcastDiv(dx1, dx2, tile_data0, tile_data1, tile_data2, dy_size,
               arithmeticParameter_);  // broadcast directly to dx1
  ReduceSumByAxes(tile_data2, arithmeticParameter_->in_shape0_, dx2, arithmeticParameter_->in_shape1_,
                  arithmeticParameter_->ndim_);
  for (int i = 0; i < dx2_size; i++) {
    dx2[i] = -dx2[i];
  }

  // broadcasting x2
  BroadcastDiv(dy, x2_data, tile_data0, tile_data1, dx1, dy_size, arithmeticParameter_);  // broadcast directly to dx1
  return RET_OK;
}

int ArithmeticGradCPUKernel::ArithmeticGradDiv2L(float *dy, int dy_size, float *dx1, int dx1_size, float *dx2,
                                                 int dx2_size) {
  auto x1_data = reinterpret_cast<float *>(in_tensors_[kX1Idx]->MutableData());
  auto x2_data = reinterpret_cast<float *>(in_tensors_[kX2Idx]->MutableData());
  CHECK_NULL_RETURN(x1_data);
  CHECK_NULL_RETURN(x2_data);

  // dx1 = dy/x2
  ElementDiv(dy, x2_data, tile_data0, dy_size);  // first multiply into temp
  ReduceSumByAxes(tile_data0, arithmeticParameter_->in_shape0_, dx1, arithmeticParameter_->in_shape1_,
                  arithmeticParameter_->ndim_);

  // dx2 = -dy*x1/(x2*x2)
  BroadcastMul(dy, x1_data, tile_data0, tile_data1, tile_data2, dy_size, arithmeticParameter_);  // broadcast numerator
  ElementDivNegSquare(tile_data2, x2_data, dx2, dy_size);
  return RET_OK;
}

int ArithmeticGradCPUKernel::ArithmeticGradMaximum(float *dy, int dy_size, float *dx1, int dx1_size, float *dx2,
                                                   int dx2_size) {
  const int kMaxX1Idx = 0;
  const int kMaxX2Idx = 1;
  const int kMaxDyIdx = 2;
  auto x1 = reinterpret_cast<float *>(in_tensors_[kMaxX1Idx]->MutableData());
  auto x2 = reinterpret_cast<float *>(in_tensors_[kMaxX2Idx]->MutableData());
  dy = reinterpret_cast<float *>(in_tensors_[kMaxDyIdx]->MutableData());
  CHECK_NULL_RETURN(x1);
  CHECK_NULL_RETURN(x2);
  CHECK_NULL_RETURN(dy);

  MaximumByAxes(x1, x2, dy, arithmeticParameter_->in_shape0_, arithmeticParameter_->in_shape1_,
                arithmeticParameter_->out_shape_, dx1, dx2, arithmeticParameter_->ndim_);
  return RET_OK;
}

int ArithmeticGradCPUKernel::ArithmeticGradMinimum(float *dy, int dy_size, float *dx1, int dx1_size, float *dx2,
                                                   int dx2_size) {
  const int kMinX1Idx = 0;
  const int kMinX2Idx = 1;
  const int kMinDyIdx = 2;
  auto x1 = reinterpret_cast<float *>(in_tensors_[kMinX1Idx]->MutableData());
  auto x2 = reinterpret_cast<float *>(in_tensors_[kMinX2Idx]->MutableData());
  dy = reinterpret_cast<float *>(in_tensors_[kMinDyIdx]->MutableData());
  CHECK_NULL_RETURN(x1);
  CHECK_NULL_RETURN(x2);
  CHECK_NULL_RETURN(dy);

  MinimumByAxes(x1, x2, dy, arithmeticParameter_->in_shape0_, arithmeticParameter_->in_shape1_,
                arithmeticParameter_->out_shape_, dx1, dx2, arithmeticParameter_->ndim_);
  return RET_OK;
}

int ArithmeticGradCPUKernel::ReSize() { return RET_OK; }

int ArithmeticGradCPUKernel::DoExecute(int task_id) {
  auto dy = reinterpret_cast<float *>(in_tensors_[kDyIdx]->MutableData());
  auto dx1 = reinterpret_cast<float *>(out_tensors_[kDx1Idx]->MutableData());
  auto dx2 = reinterpret_cast<float *>(out_tensors_[kDx2Idx]->MutableData());
  CHECK_NULL_RETURN(dy);
  CHECK_NULL_RETURN(dx1);
  CHECK_NULL_RETURN(dx2);

  size_t dy_size = in_tensors_.at(kDyIdx)->ElementsNum();
  size_t dx1_size = out_tensors_.at(kDx1Idx)->ElementsNum();
  size_t dx2_size = out_tensors_.at(kDx2Idx)->ElementsNum();
  (this->*arithmetic_grad_)(dy, dy_size, dx1, dx1_size, dx2, dx2_size);
  return RET_OK;
}

int ArithmeticGradRun(void *cdata, int task_id, float lhs_scale, float rhs_scale) {
  auto Arithmetic_kernel = reinterpret_cast<ArithmeticGradCPUKernel *>(cdata);
  CHECK_NULL_RETURN(Arithmetic_kernel);
  auto error_code = Arithmetic_kernel->DoExecute(task_id);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "ArithmeticGradRun error task_id[" << task_id << "] error_code[" << error_code << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int ArithmeticGradCPUKernel::Run() {
  int error_code = ParallelLaunch(this->ms_context_, ArithmeticGradRun, this, 1);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "Arithmetic Grad function error error_code[" << error_code << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

kernel::LiteKernel *CpuArithmeticGradFp32KernelCreator(const std::vector<lite::Tensor *> &inputs,
                                                       const std::vector<lite::Tensor *> &outputs,
                                                       OpParameter *opParameter, const lite::InnerContext *ctx,
                                                       const kernel::KernelKey &desc) {
  MS_CHECK_TRUE_MSG(opParameter != nullptr, nullptr, "Op parameter is nullptr.");
  auto *kernel = new (std::nothrow) ArithmeticGradCPUKernel(opParameter, inputs, outputs, ctx);
  if (kernel == nullptr) {
    MS_LOG(ERROR) << "new ArithmeticGradCPUKernel fail!";
    free(opParameter);
    return nullptr;
  }
  return kernel;
}

REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_MulGrad, CpuArithmeticGradFp32KernelCreator)
REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_AddGrad, CpuArithmeticGradFp32KernelCreator)
REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_SubGrad, CpuArithmeticGradFp32KernelCreator)
REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_DivGrad, CpuArithmeticGradFp32KernelCreator)
REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_MaximumGrad, CpuArithmeticGradFp32KernelCreator)
REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_MinimumGrad, CpuArithmeticGradFp32KernelCreator)
}  // namespace mindspore::kernel
