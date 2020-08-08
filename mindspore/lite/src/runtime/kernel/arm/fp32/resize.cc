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

#include "src/runtime/kernel/arm/fp32/resize.h"
#include <vector>
#include "schema/model_generated.h"
#include "src/kernel_registry.h"
#include "src/runtime/kernel/arm/nnacl/resize.h"
#include "src/runtime/kernel/arm/nnacl/pack.h"
#include "include/errorcode.h"
#include "src/runtime/runtime_api.h"

using mindspore::kernel::KERNEL_ARCH::kCPU;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_INVALID_OP_ATTR;
using mindspore::lite::RET_NULL_PTR;
using mindspore::lite::RET_OK;

namespace mindspore::kernel {
namespace {
constexpr int kInputNum = 1;
constexpr int kOutputNum = 1;
constexpr int kRank = 4;
}  // namespace

int ResizeCPUKernel::CheckParameters() {
  auto parameter = reinterpret_cast<ResizeParameter *>(opParameter);
  if (parameter == nullptr) {
    MS_LOG(ERROR) << "cast ResizeParameter failed.";
    return RET_NULL_PTR;
  }
  method_ = parameter->method_;
  if (method_ != schema::ResizeMethod_BILINEAR && method_ != schema::ResizeMethod_NEAREST_NEIGHBOR) {
    MS_LOG(ERROR) << "Resize method should be bilinear or nearest_neighbor, but got " << method_;
    return RET_INVALID_OP_ATTR;
  }
  new_height_ = parameter->new_height_;
  if (new_height_ < 1) {
    MS_LOG(ERROR) << "Resize new_height should >= 1, but got " << new_height_;
    return RET_INVALID_OP_ATTR;
  }
  new_width_ = parameter->new_width_;
  if (new_width_ < 1) {
    MS_LOG(ERROR) << "Resize new_width should >= 1, but got " << new_width_;
    return RET_INVALID_OP_ATTR;
  }
  align_corners_ = parameter->align_corners_;
  preserve_aspect_ratio = parameter->preserve_aspect_ratio_;
  if (preserve_aspect_ratio) {
    MS_LOG(ERROR) << "Resize currently not support preserve_aspect_ratio true";
    return RET_ERROR;
  }
  return RET_OK;
}

int ResizeCPUKernel::CheckInputsOuputs() {
  if (inputs_.size() != kInputNum) {
    MS_LOG(ERROR) << "Resize input num should be " << kInputNum << ", but got " << inputs_.size();
    return RET_ERROR;
  }
  auto input = inputs_.at(0);
  if (input == nullptr) {
    return RET_NULL_PTR;
  }
  if (outputs_.size() != kOutputNum) {
    MS_LOG(ERROR) << "Resize output num should be " << kOutputNum << ", but got " << outputs_.size();
    return RET_ERROR;
  }
  auto output = outputs_.at(0);
  if (output == nullptr) {
    return RET_NULL_PTR;
  }
  return RET_OK;
}

int ResizeCPUKernel::Init() {
  if (context_->infer_shape_interrupt_ && !context_->running_) {
    SetNeedReInit();
    return RET_OK;
  }
  auto ret = CheckParameters();
  if (ret != RET_OK) {
    return ret;
  }
  ret = CheckInputsOuputs();
  if (ret != RET_OK) {
    return ret;
  }

  auto output = outputs_.at(0);
  auto input = inputs_.at(0);
  auto input_shape = input->shape();
  if (input_shape.size() != kRank) {
    return RET_ERROR;
  }
  schema::Format execute_format;
  size_t exec_input_size;
  switch (method_) {
    case schema::ResizeMethod_BILINEAR: {
      execute_format = schema::Format_NC4HW4;
      output->SetFormat(schema::Format_NC4HW4);
      exec_input_size = input->ElementsC4Num();
      break;
    }
    case schema::ResizeMethod_NEAREST_NEIGHBOR: {
      execute_format = schema::Format_NHWC;
      output->SetFormat(schema::Format_NHWC);
      exec_input_size = input->ElementsNum();
      break;
    }
    default: {
      MS_LOG(ERROR) << "Resize unknown method " << method_;
      return RET_ERROR;
    }
  }

  auto input_format = input->GetFormat();
  if (input_format != execute_format) {
    auto input_type = input->data_type();
    layout_convertor_ = LayoutTransform(input_type, input_format, execute_format);
    exec_input_data_ = reinterpret_cast<float *>(malloc(exec_input_size * sizeof(float)));
    if (exec_input_data_ == nullptr) {
      return RET_NULL_PTR;
    }
  }

  return RET_OK;
}

int ResizeImpl(int task_id, LiteParallelGroupEnv *penv, void *cdata) {
  auto resize = reinterpret_cast<ResizeCPUKernel *>(cdata);
  auto error_code = resize->RunImpl(task_id);
  if (error_code != NNACL_OK) {
    MS_LOG(ERROR) << "Resize Run error task_id[" << task_id << "] error_code[" << error_code << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int ResizeCPUKernel::RunImpl(int task_id) {
  auto input = inputs_.at(0);
  auto input_data = reinterpret_cast<float *>(input->Data());
  if (input_data == nullptr) {
    return RET_NULL_PTR;
  }
  auto output_data = reinterpret_cast<float *>(outputs_.at(0)->Data());
  if (output_data == nullptr) {
    return RET_NULL_PTR;
  }
  auto input_shape = input->shape();
  if (input_shape.size() != kRank) {
    return RET_ERROR;
  }
  if (context_ == nullptr) {
    return RET_NULL_PTR;
  }

  int ret = 0;
  switch (method_) {
    case schema::ResizeMethod_BILINEAR: {
      if (layout_convertor_ != nullptr) {
        layout_convertor_(input_data, exec_input_data_, input->Batch(), input->Height() * input->Width(),
                          input->Channel());
        ret = ResizeBilinear(exec_input_data_, output_data, inputs_[0]->shape().data(), outputs_[0]->shape().data(),
                             align_corners_, task_id, context_->thread_num_);
      } else {
        ret = ResizeBilinear(input_data, output_data, inputs_[0]->shape().data(), outputs_[0]->shape().data(),
                             align_corners_, task_id, context_->thread_num_);
      }
      break;
    }
    case schema::ResizeMethod_NEAREST_NEIGHBOR: {
      if (align_corners_) {
        MS_LOG(ERROR) << "ResizeNearestNeighbor not support align_corners.";
        return RET_ERROR;
      }
      if (layout_convertor_ != nullptr) {
        layout_convertor_(input_data, exec_input_data_, input->Batch(), input->Height() * input->Width(),
                          input->Channel());
        ret = ResizeNearestNeighbor(exec_input_data_, output_data, input_shape.data(), outputs_[0]->shape().data(),
                                    task_id, context_->thread_num_);
      } else {
        ret = ResizeNearestNeighbor(input_data, output_data, input_shape.data(), outputs_[0]->shape().data(), task_id,
                                    context_->thread_num_);
      }
      break;
    }
    case schema::ResizeMethod_UNKNOW:
    default: {
      MS_LOG(ERROR) << "Resize unknown method " << method_;
      ret = NNACL_ERR;
    }
  }
  return ret;
}

int ResizeCPUKernel::Run() {
  auto ret = Prepare();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Prepare failed.";
    return RET_ERROR;
  }
  int error_code = LiteBackendParallelLaunch(ResizeImpl, this, context_->thread_num_);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "Resize run error, error_code[" << error_code << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

kernel::LiteKernel *CpuResizeFp32KernelCreator(const std::vector<lite::tensor::Tensor *> &inputs,
                                               const std::vector<lite::tensor::Tensor *> &outputs,
                                               OpParameter *opParameter, const lite::Context *ctx,
                                               const kernel::KernelKey &desc, const lite::Primitive *primitive) {
  if (opParameter == nullptr) {
    MS_LOG(ERROR) << "Input opParameter is nullptr!";
    return nullptr;
  }
  MS_ASSERT(desc.type == schema::PrimitiveType_Resize);
  auto *kernel = new (std::nothrow) ResizeCPUKernel(opParameter, inputs, outputs, ctx, primitive);
  if (kernel == nullptr) {
    MS_LOG(ERROR) << "new ResizeCPUKernel fail!";
    return nullptr;
  }
  auto ret = kernel->Init();
  if (ret != RET_OK) {
    delete kernel;
    MS_LOG(ERROR) << "Init kernel failed, name: " << opParameter->name_ << ", type: "
                  << schema::EnumNamePrimitiveType(static_cast<schema::PrimitiveType>(opParameter->type_));
    return nullptr;
  }

  return kernel;
}

REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_Resize, CpuResizeFp32KernelCreator)
}  // namespace mindspore::kernel
