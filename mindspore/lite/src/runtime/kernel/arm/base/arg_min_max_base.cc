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
#include "src/runtime/kernel/arm/base/arg_min_max_base.h"
#include "src/runtime/kernel/arm/nnacl/arg_min_max.h"
#include "src/runtime/kernel/arm/fp32/argminmax.h"
#include "src/runtime/kernel/arm/int8/argminmax_int8.h"
#include "src/runtime/kernel/arm/nnacl/arithmetic_common.h"
#include "schema/model_generated.h"
#include "src/kernel_factory.h"
#include "include/errorcode.h"
#include "include/context.h"

using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_FORMAT_ERR;
using mindspore::lite::RET_OK;
using mindspore::lite::RET_PARAM_INVALID;
using mindspore::schema::PrimitiveType_ArgMax;
using mindspore::schema::PrimitiveType_ArgMin;

namespace mindspore::kernel {
int ArgMinMaxBaseCPUKernel::Init() {
  auto param = reinterpret_cast<ArgMinMaxParameter *>(op_parameter_);
  switch (op_parameter_->type_) {
    case PrimitiveType_ArgMax:
      param->get_max_ = true;
      break;
    case PrimitiveType_ArgMin:
      param->get_max_ = false;
      break;
    default:
      MS_LOG(ERROR) << "Unexpected type " << op_parameter_->type_;
      return RET_ERROR;
  }

  return RET_OK;
}

int ArgMinMaxBaseCPUKernel::ReSize() {
  auto in_shape = in_tensors_.at(0)->shape();
  auto dims_size = in_shape.size();
  auto param = reinterpret_cast<ArgMinMaxParameter *>(op_parameter_);
  int axis = param->axis_ < 0 ? param->axis_ + dims_size : param->axis_;
  param->axis_ = axis;
  param->dims_size_ = dims_size;
  if (param->topk_ <= 0) {
    MS_LOG(ERROR) << "Invalid topk " << param->topk_;
    return RET_PARAM_INVALID;
  }
  param->topk_ = MSMIN(param->topk_, in_shape[axis]);
  if (param->topk_ > 1 || param->keep_dims_) {
    if (context_ != nullptr && context_->allocator != nullptr) {
      param->arg_elements_ =
        reinterpret_cast<ArgElement *>(context_->allocator->Malloc(sizeof(ArgElement) * in_shape[axis]));
      data_from_allocator_ = true;
    } else {
      param->arg_elements_ = reinterpret_cast<ArgElement *>(malloc(sizeof(ArgElement) * in_shape[axis]));
    }
    if (param->arg_elements_ == nullptr) {
      MS_LOG(ERROR) << "malloc memroy fail!";
      return RET_ERROR;
    }
  }
  ComputeStrides(in_shape.data(), param->in_strides_, in_shape.size());
  auto out_shape = out_tensors_.at(0)->shape();
  ComputeStrides(out_shape.data(), param->out_strides_, out_shape.size());
  return RET_OK;
}

int ArgMinMaxBaseCPUKernel::Run() {
  auto input = in_tensors_.at(0);

  auto input_data = reinterpret_cast<const void *>(in_tensors_.at(0)->Data());
  auto output_data = out_tensors_.at(0)->Data();

  auto shape = input->shape().data();
  auto param = reinterpret_cast<ArgMinMaxParameter *>(op_parameter_);
  ArgMinMax(input_data, output_data, reinterpret_cast<const int *>(shape), param);
  return RET_OK;
}

void ArgMinMaxBaseCPUKernel::FreeTmpMemory() {
  auto param = reinterpret_cast<ArgMinMaxParameter *>(op_parameter_);
  if (param->arg_elements_ == nullptr) {
    return;
  }
  if (data_from_allocator_) {
    context_->allocator->Free(param->arg_elements_);
  } else {
    free(param->arg_elements_);
  }
  param->arg_elements_ = nullptr;
}

kernel::LiteKernel *CpuArgMinMaxInt8KernelCreator(const std::vector<lite::tensor::Tensor *> &inputs,
                                                  const std::vector<lite::tensor::Tensor *> &outputs,
                                                  OpParameter *op_parameter, const lite::Context *ctx,
                                                  const kernel::KernelKey &desc, const lite::Primitive *primitive) {
  if (op_parameter == nullptr) {
    MS_LOG(ERROR) << "Input op_parameter is nullptr!";
    return nullptr;
  }
  auto kernel = new (std::nothrow) ArgMinMaxInt8CPUKernel(op_parameter, inputs, outputs, ctx, primitive);
  if (kernel == nullptr) {
    MS_LOG(ERROR) << "new ArgMinMaxInt8CPUKernel fail!";
    return nullptr;
  }

  auto ret = kernel->Init();
  if (ret != RET_OK) {
    delete kernel;
    MS_LOG(ERROR) << "Init kernel failed, name: " << op_parameter->name_ << ", type: "
                  << schema::EnumNamePrimitiveType(static_cast<schema::PrimitiveType>(op_parameter->type_));
    return nullptr;
  }
  return kernel;
}

kernel::LiteKernel *CpuArgMinMaxFp32KernelCreator(const std::vector<lite::tensor::Tensor *> &inputs,
                                                  const std::vector<lite::tensor::Tensor *> &outputs,
                                                  OpParameter *op_parameter, const lite::Context *ctx,
                                                  const kernel::KernelKey &desc, const lite::Primitive *primitive) {
  if (op_parameter == nullptr) {
    MS_LOG(ERROR) << "Input op_parameter is nullptr!";
    return nullptr;
  }
  auto kernel = new (std::nothrow) ArgMinMaxCPUKernel(op_parameter, inputs, outputs, ctx, primitive);
  if (kernel == nullptr) {
    MS_LOG(ERROR) << "new ArgMinMaxCPUKernel fail!";
    return nullptr;
  }
  auto ret = kernel->Init();
  if (ret != RET_OK) {
    delete kernel;
    MS_LOG(ERROR) << "Init kernel failed, name: " << op_parameter->name_ << ", type: "
                  << schema::EnumNamePrimitiveType(static_cast<schema::PrimitiveType>(op_parameter->type_));
    return nullptr;
  }
  return kernel;
}

REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_ArgMax, CpuArgMinMaxFp32KernelCreator)
REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_ArgMin, CpuArgMinMaxFp32KernelCreator)
REG_KERNEL(kCPU, kNumberTypeInt8, PrimitiveType_ArgMax, CpuArgMinMaxInt8KernelCreator)
REG_KERNEL(kCPU, kNumberTypeInt8, PrimitiveType_ArgMin, CpuArgMinMaxInt8KernelCreator)
}  // namespace mindspore::kernel
