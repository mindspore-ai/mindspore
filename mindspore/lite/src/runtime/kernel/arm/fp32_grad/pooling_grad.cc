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

#include "src/runtime/kernel/arm/fp32_grad/pooling_grad.h"
#include "schema/model_generated.h"
#include "src/kernel_registry.h"
#include "src/runtime/kernel/arm/nnacl/fp32/pooling.h"
#include "src/runtime/kernel/arm/nnacl/fp32_grad/pooling_grad.h"
#include "include/errorcode.h"

using mindspore::kernel::KERNEL_ARCH::kCPU;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_PoolingGrad;

namespace mindspore::kernel {
#if 0
int PoolingGradCPUKernel::TfPadding(int input_w, int input_h, int &output_w, int &output_h) {
  PoolingParameter *pool_param = reinterpret_cast<PoolingParameter *> (opParameter);

  auto stride_w = pool_param->stride_w_;
  auto stride_h = pool_param->stride_h_;
  auto window_w = pool_param->window_w_;
  auto window_h = pool_param->window_h_;
  auto pad_up = pool_param->pad_u_;
  auto pad_down = pool_param->pad_d_;
  auto pad_left = pool_param->pad_l_;
  auto pad_right = pool_param->pad_r_;
  if (pool_param->pad_mode_ == PADMODE_SAME) {
    output_w = ceil(input_w / stride_w);
    output_h = ceil(input_h / stride_h);
  } else {
    output_w = ceil((input_w + pad_left + pad_right - window_w + 1) / stride_w);
    output_h = ceil((input_h + pad_up + pad_down - window_h + 1) / stride_h);
  }
  return RET_OK;
}

int PoolingGradCPUKernel::CaffePadding(int input_w, int input_h, int &output_w, int &output_h) {
  PoolingParameter *pool_param = reinterpret_cast<PoolingParameter *> (opParameter);

  auto round_mode = pool_param->round_mode_;
  auto stride_w = pool_param->stride_w_;
  auto stride_h = pool_param->stride_h_;
  auto window_w = pool_param->window_w_;
  auto window_h = pool_param->window_h_;
  auto pad_up = pool_param->pad_u_;
  auto pad_down = pool_param->pad_d_;
  auto pad_left = pool_param->pad_l_;
  auto pad_right = pool_param->pad_r_;
  if (round_mode == ROUNDMODE_FLOOR && false) {
    output_w = floor((input_w + pad_left + pad_right - window_w) / stride_w + 1);
    output_h = floor((input_h + pad_up + pad_down - window_h) / stride_h + 1);
  } else if (round_mode == ROUNDMODE_CEIL || true) {
    output_w = ceil((input_w + pad_left + pad_right - window_w) / stride_w + 1);
    output_h = ceil((input_h + pad_up + pad_down - window_h) / stride_h + 1);
  } else {
    MS_LOG(ERROR) << "round mode not support.";
  }

  if (pad_left > 0 || pad_up > 0) {
    if ((output_w - 1) * stride_w >= input_w + pad_left) {
      --output_w;
    }
    if ((output_h - 1) * stride_h >= input_h + pad_up) {
      --output_h;
    }
  }
  return RET_OK;
}

int PoolingGradCPUKernel::OnnxPadding(int input_w, int input_h, int &output_w, int &output_h) {
  PoolingParameter *pool_param = reinterpret_cast<PoolingParameter *> (opParameter);

  auto round_mode = pool_param->round_mode_;
  auto stride_w = pool_param->stride_w_;
  auto stride_h = pool_param->stride_h_;
  auto window_w = pool_param->window_w_;
  auto window_h = pool_param->window_h_;
  auto pad_up = pool_param->pad_u_;
  auto pad_down = pool_param->pad_d_;
  auto pad_left = pool_param->pad_l_;
  auto pad_right = pool_param->pad_r_;
  if (round_mode == ROUNDMODE_FLOOR) {
    output_w = floor((input_w + pad_left + pad_right - window_w) / stride_w + 1);
    output_h = floor((input_h + pad_up + pad_down - window_h) / stride_h + 1);
  } else if (round_mode == ROUNDMODE_CEIL) {
    MS_LOG(ERROR) << "RoundMode_CEIL mode not support.";
  } else {
    MS_LOG(ERROR) << "OnnxPadding round mode not support.";
  }
  return RET_OK;
}
#endif

int PoolingGradCPUKernel::Init() {
  // InferShape():
  // auto *in_tensor = reinterpret_cast<float *>(inputs_.at(0)->Data());
  // auto *x_tensor = reinterpret_cast<float *>(inputs_.at(1)->Data());

  PoolingParameter *pool_param = reinterpret_cast<PoolingParameter *>(opParameter);

  auto in_shape = inputs_.at(0)->shape();
  int input_h = in_shape.at(1);
  int input_w = in_shape.at(2);

  if (pool_param->global_) {
    pool_param->window_w_ = input_w;
    pool_param->window_h_ = input_h;
  }

  // Emir -- here I assume we get the outputshape in the output tensor
  auto *out_tensor = outputs_.front();
  auto out_shape = out_tensor->shape();

#if 0
  int output_w = 0, output_h = 0;
  auto fmk_type = pool_param->fmk_type_;
  switch (fmk_type) {
    case lite::FmkType_TF:
      break;
    case lite::FmkType_CAFFE:
      CaffePadding(input_w, input_h, output_w, output_h);
      break;
    case lite::FmkType_ONNX:
      OnnxPadding(input_w, input_h, output_w, output_h);
      break;
    case lite::FmkType_MS:
      break;
    case lite::FmkType_TFLITE:
      TfPadding(input_w, input_h, output_w, output_h);
      break;
    default:
      MS_LOG(ERROR) << "Not support this framework.";
  }
  std::vector<int> out_shape{in_tensor->shape()};
  out_shape.at(1) = output_h;
  out_shape.at(2) = output_w;
#endif
  out_tensor->set_shape(out_shape);
  out_tensor->set_data_type(inputs_.at(0)->data_type());
  return RET_OK;
}

int PoolingGradCPUKernel::ReSize() { return RET_OK; }

int PoolingGradCPUKernel::Run() {
  PoolingParameter *pool_param = reinterpret_cast<PoolingParameter *>(opParameter);
  auto input_ptr = reinterpret_cast<float *>(inputs_.at(0)->Data());
  auto output_ptr = reinterpret_cast<float *>(outputs_.at(0)->Data());

  if (pool_param->max_pooling_) {
    auto ind = reinterpret_cast<int *>(inputs_.at(1)->Data());
    MaxPoolingGrad(input_ptr, ind, output_ptr, pool_param);
  } else {
    AvgPoolingGrad(input_ptr, output_ptr, pool_param);
  }
  return RET_OK;
}

kernel::LiteKernel *CpuPoolingGradFp32KernelCreator(const std::vector<lite::tensor::Tensor *> &inputs,
                                                    const std::vector<lite::tensor::Tensor *> &outputs,
                                                    OpParameter *opParameter, const lite::Context *ctx,
                                                    const kernel::KernelKey &desc, const lite::Primitive *primitive) {
  MS_ASSERT(opParameter != nullptr);
  MS_ASSERT(desc.type == schema::PrimitiveType_PoolingGrad);

  auto *kernel = new (std::nothrow) PoolingGradCPUKernel(opParameter, inputs, outputs, ctx, primitive);
  MS_ASSERT(kernel != nullptr);
  auto ret = kernel->Init();
  if (RET_OK != ret) {
    MS_LOG(ERROR) << "Init kernel failed, name: " << opParameter->name_ << ", type: "
                  << schema::EnumNamePrimitiveType(static_cast<schema::PrimitiveType>(opParameter->type_));
    delete kernel;
    return nullptr;
  }
  return kernel;
}

REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_PoolingGrad, CpuPoolingGradFp32KernelCreator)
}  // namespace mindspore::kernel
