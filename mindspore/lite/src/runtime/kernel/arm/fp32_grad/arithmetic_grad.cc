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

#include "schema/model_generated.h"
#include "src/kernel_registry.h"
#include "src/runtime/kernel/arm/nnacl/fp32_grad/reduce_grad.h"
#include "src/runtime/kernel/arm/nnacl/fp32_grad/arithmetic_grad.h"
#include "src/runtime/kernel/arm/fp32_grad/arithmetic_grad.h"
#include "include/errorcode.h"

using mindspore::kernel::KERNEL_ARCH::kCPU;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;

namespace mindspore::kernel {
namespace {
constexpr int kArithGradOpInputNum = 3;
constexpr int kArithGradOpOutputNum = 2;
}  // namespace

int ArithmeticGradCPUKernel::Init() {
  auto ret = InferShape();
  return ret;
}

int ArithmeticGradCPUKernel::InferShape() {
  if (inputs_.size() != kArithGradOpInputNum) {
    MS_LOG(ERROR) << "The number of input must be " << kArithGradOpInputNum;
    return RET_ERROR;
  }
  if (outputs_.size() != kArithGradOpOutputNum) {
    MS_LOG(ERROR) << "The number of output must be " << kArithGradOpOutputNum;
    return RET_ERROR;
  }
  auto dy = inputs_[0];
  auto x1 = inputs_[1];
  auto x2 = inputs_[2];
  auto dx1 = outputs_[0];
  auto dx2 = outputs_[1];

  MS_ASSERT(dy != nullptr);
  MS_ASSERT(x1 != nullptr);
  MS_ASSERT(x2 != nullptr);
  MS_ASSERT(dx1 != nullptr);
  MS_ASSERT(dx2 != nullptr);

  auto inShape0 = x1->shape();
  auto inShape1 = x2->shape();
  auto outShape = dy->shape();

  if ((type() == PrimitiveType_AddGrad) || (type() == PrimitiveType_SubGrad)) {
    arithmeticParameter_->ndim_ = outShape.size();
    auto fillDimNum0 = outShape.size() - inShape0.size();
    auto fillDimNum1 = outShape.size() - inShape1.size();
    int j0 = 0;
    int j1 = 0;
    for (unsigned int i = 0; i < outShape.size(); i++) {
      arithmeticParameter_->in_shape0_[i] = (i < fillDimNum0) ? 1 : inShape0[j0++];
      arithmeticParameter_->in_shape1_[i] = (i < fillDimNum1) ? 1 : inShape1[j1++];
      arithmeticParameter_->out_shape_[i] = outShape[i];
    }
  } else {
    // if (inShape0.size() < inShape1.size())
    if (dx1->ElementsNum() < dx2->ElementsNum()) {
      arithmeticParameter_->ndim_ = inShape1.size();
      if (type() == PrimitiveType_MulGrad)
        arithmetic_grad_ = &ArithmeticGradCPUKernel::ArithmeticGradMul2L;
      else if (type() == PrimitiveType_DivGrad)
        arithmetic_grad_ = &ArithmeticGradCPUKernel::ArithmeticGradDiv2L;

      auto fillDimNum = inShape1.size() - inShape0.size();  // This will not work for batch!
      int j = 0;
      for (unsigned int i = 0; i < inShape1.size(); i++) {
        if (i < fillDimNum) {
          arithmeticParameter_->in_shape1_[i] = 1;
        } else {
          arithmeticParameter_->in_shape1_[i] = inShape0[j++];
        }
        arithmeticParameter_->in_shape0_[i] = inShape1[i];
        arithmeticParameter_->out_shape_[i] = outShape[i];
      }
    } else if (dx2->ElementsNum() < dx1->ElementsNum()) {  // if (inShape0.size() > inShape1.size())
      arithmeticParameter_->ndim_ = inShape0.size();
      if (type() == PrimitiveType_MulGrad)
        arithmetic_grad_ = &ArithmeticGradCPUKernel::ArithmeticGradMul1L;
      else if (type() == PrimitiveType_DivGrad)
        arithmetic_grad_ = &ArithmeticGradCPUKernel::ArithmeticGradDiv1L;
      arithmeticParameter_->broadcasting_ = true;
      arithmeticParameter_->ndim_ = inShape0.size();
      int j = 0;
      auto fillDimNum = inShape0.size() - inShape1.size();
      for (unsigned int i = 0; i < inShape0.size(); i++) {
        if (i < fillDimNum) {
          arithmeticParameter_->in_shape1_[i] = 1;
        } else {
          arithmeticParameter_->in_shape1_[i] = inShape1[j++];
        }
        arithmeticParameter_->in_shape0_[i] = inShape0[i];
        arithmeticParameter_->out_shape_[i] = outShape[i];
      }
    } else {
      arithmeticParameter_->broadcasting_ = false;
      for (unsigned int i = 0; i < inShape0.size(); i++) {
        arithmeticParameter_->in_shape1_[i] = inShape1[i];
        arithmeticParameter_->in_shape0_[i] = inShape0[i];
        arithmeticParameter_->out_shape_[i] = outShape[i];
      }
    }
    tile_data0 = new (std::nothrow) float[inputs_.at(0)->ElementsNum()];
    MS_ASSERT(tile_data0 != nullptr);
    tile_data1 = new (std::nothrow) float[inputs_.at(0)->ElementsNum()];
    MS_ASSERT(tile_data1 != nullptr);
    if (type() == PrimitiveType_DivGrad) {
      tile_data2 = new (std::nothrow) float[inputs_.at(0)->ElementsNum()];
      MS_ASSERT(tile_data2 != nullptr);
    }
  }

  dx1->set_shape(x1->shape());
  dx2->set_shape(x2->shape());
  // outTensor->set_shape(out_shape);
  dx1->set_data_type(dy->data_type());
  dx2->set_data_type(dy->data_type());
  return RET_OK;
}

void ArithmeticGradCPUKernel::ArithmeticGradAdd(float *dy, int dy_size, float *dx1, int dx1_size, float *dx2,
                                                int dx2_size) {
  if (dx1_size == dy_size)
    memcpy(dx1, dy, dy_size * sizeof(float));
  else
    ReduceSumByAxes(dy, arithmeticParameter_->out_shape_, dx1, arithmeticParameter_->in_shape0_,
                    arithmeticParameter_->ndim_);
  if (dx2_size == dy_size)
    memcpy(dx2, dy, dy_size * sizeof(float));
  else
    ReduceSumByAxes(dy, arithmeticParameter_->out_shape_, dx2, arithmeticParameter_->in_shape1_,
                    arithmeticParameter_->ndim_);
}

void ArithmeticGradCPUKernel::ArithmeticGradSub(float *dy, int dy_size, float *dx1, int dx1_size, float *dx2,
                                                int dx2_size) {
  if (dx1_size == dy_size)
    memcpy(dx1, dy, dy_size * sizeof(float));
  else
    ReduceSumByAxes(dy, arithmeticParameter_->out_shape_, dx1, arithmeticParameter_->in_shape0_,
                    arithmeticParameter_->ndim_);
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
}

void ArithmeticGradCPUKernel::ArithmeticGradMul(float *dy, int dy_size, float *dx1, int dx1_size, float *dx2,
                                                int dx2_size) {
  auto x1_data = reinterpret_cast<float *>(inputs_[1]->Data());
  auto x2_data = reinterpret_cast<float *>(inputs_[2]->Data());
  ElementMul(dy, x1_data, dx2, dy_size);
  ElementMul(dy, x2_data, dx1, dy_size);
}

void ArithmeticGradCPUKernel::ArithmeticGradMul1L(float *dy, int dy_size, float *dx1, int dx1_size, float *dx2,
                                                  int dx2_size) {
  auto x1_data = reinterpret_cast<float *>(inputs_[1]->Data());
  auto x2_data = reinterpret_cast<float *>(inputs_[2]->Data());
  ElementMul(dy, x1_data, tile_data0, dy_size);
  ReduceSumByAxes(tile_data0, arithmeticParameter_->in_shape0_, dx2, arithmeticParameter_->in_shape1_,
                  arithmeticParameter_->ndim_);

  BroadcastMul(dy, x2_data, tile_data0, tile_data1, dx1, dy_size, arithmeticParameter_);  // broadcast directly to dx1
}

void ArithmeticGradCPUKernel::ArithmeticGradMul2L(float *dy, int dy_size, float *dx1, int dx1_size, float *dx2,
                                                  int dx2_size) {
  auto x1_data = reinterpret_cast<float *>(inputs_[1]->Data());
  auto x2_data = reinterpret_cast<float *>(inputs_[2]->Data());
  ElementMul(dy, x2_data, tile_data0, dy_size);
  ReduceSumByAxes(tile_data0, arithmeticParameter_->in_shape0_, dx1, arithmeticParameter_->in_shape1_,
                  arithmeticParameter_->ndim_);

  BroadcastMul(dy, x1_data, tile_data0, tile_data1, dx2, dy_size, arithmeticParameter_);  // broadcast directly to dx2
}

void ArithmeticGradCPUKernel::ArithmeticGradDiv(float *dy, int dy_size, float *dx1, int dx1_size, float *dx2,
                                                int dx2_size) {
  auto x1 = reinterpret_cast<float *>(inputs_[1]->Data());
  auto x2 = reinterpret_cast<float *>(inputs_[2]->Data());
  ElementDiv(dy, x2, dx1, dy_size);
  ElementMulAndDivNegSquare(dy, x1, x2, dx2, dy_size);
}

void ArithmeticGradCPUKernel::ArithmeticGradDiv1L(float *dy, int dy_size, float *dx1, int dx1_size, float *dx2,
                                                  int dx2_size) {
  auto x1_data = reinterpret_cast<float *>(inputs_[1]->Data());
  auto x2_data = reinterpret_cast<float *>(inputs_[2]->Data());

  ElementMul(x2_data, x2_data, dx2, dx2_size);
  ElementMul(x1_data, dy, dx1, dy_size);  // use dx1 buffer
  BroadcastDiv(dx1, dx2, tile_data0, tile_data1, tile_data2, dy_size,
               arithmeticParameter_);  // broadcast directly to dx1
  ReduceSumByAxes(tile_data2, arithmeticParameter_->in_shape0_, dx2, arithmeticParameter_->in_shape1_,
                  arithmeticParameter_->ndim_);
  for (int i = 0; i < dx2_size; i++) dx2[i] = -dx2[i];
  // ReduceNegSumPrefix(tile_data2, dy_size, dx2, dx2_size); //then reduce into dx2

  // broadcasting x2
  BroadcastDiv(dy, x2_data, tile_data0, tile_data1, dx1, dy_size, arithmeticParameter_);  // broadcast directly to dx1
}

void ArithmeticGradCPUKernel::ArithmeticGradDiv2L(float *dy, int dy_size, float *dx1, int dx1_size, float *dx2,
                                                  int dx2_size) {
  auto x1_data = reinterpret_cast<float *>(inputs_[1]->Data());
  auto x2_data = reinterpret_cast<float *>(inputs_[2]->Data());

  // dx1 = dy/x2
  ElementDiv(dy, x2_data, tile_data0, dy_size);  // first multiply into temp
  ReduceSumByAxes(tile_data0, arithmeticParameter_->in_shape0_, dx1, arithmeticParameter_->in_shape1_,
                  arithmeticParameter_->ndim_);

  // dx2 = -dy*x1/(x2*x2)
  BroadcastMul(dy, x1_data, tile_data0, tile_data1, tile_data2, dy_size, arithmeticParameter_);  // broadcast numerator
  ElementDivNegSquare(tile_data2, x2_data, dx2, dy_size);
}

int ArithmeticGradCPUKernel::ReSize() { return RET_OK; }

int ArithmeticGradCPUKernel::Run() {
  auto dy = reinterpret_cast<float *>(inputs_[0]->Data());
  // auto input1_data1 = reinterpret_cast<float *>(inputs_[1]->Data());
  auto dx1 = reinterpret_cast<float *>(outputs_[0]->Data());
  auto dx2 = reinterpret_cast<float *>(outputs_[1]->Data());

  size_t dy_size = inputs_.at(0)->ElementsNum();
  size_t dx1_size = outputs_.at(0)->ElementsNum();
  size_t dx2_size = outputs_[1]->ElementsNum();
  (this->*arithmetic_grad_)(dy, dy_size, dx1, dx1_size, dx2, dx2_size);
  return RET_OK;
}

kernel::LiteKernel *CpuArithmeticGradFp32KernelCreator(const std::vector<lite::tensor::Tensor *> &inputs,
                                                       const std::vector<lite::tensor::Tensor *> &outputs,
                                                       OpParameter *opParameter, const lite::Context *ctx,
                                                       const kernel::KernelKey &desc,
                                                       const lite::Primitive *primitive) {
  MS_EXCEPTION_IF_NULL(opParameter);
  if (opParameter == nullptr) {
    return nullptr;
  }
  auto *kernel = new (std::nothrow) ArithmeticGradCPUKernel(opParameter, inputs, outputs, ctx, primitive);
  MS_ASSERT(kernel != nullptr);
  auto ret = kernel->Init();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Init kernel failed, name: " << opParameter->name_ << ", type: "
                  << schema::EnumNamePrimitiveType(static_cast<schema::PrimitiveType>(opParameter->type_));
    delete kernel;
    return nullptr;
  }
  return kernel;
}

REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_MulGrad, CpuArithmeticGradFp32KernelCreator)
REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_AddGrad, CpuArithmeticGradFp32KernelCreator)
REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_SubGrad, CpuArithmeticGradFp32KernelCreator)
REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_DivGrad, CpuArithmeticGradFp32KernelCreator)
}  // namespace mindspore::kernel
