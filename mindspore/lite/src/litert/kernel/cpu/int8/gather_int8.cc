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
#include "src/litert/kernel/cpu/int8/gather_int8.h"
#include <vector>
#include "src/litert/kernel/cpu/int8/dynamic_gather_int8.h"
#include "nnacl/gather_parameter.h"
#include "nnacl/int8/gather_int8.h"
#include "nnacl/int8/quantize.h"
#include "schema/model_generated.h"
#include "src/litert/kernel_registry.h"
#include "include/errorcode.h"

using mindspore::kernel::KERNEL_ARCH;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_Gather;

namespace mindspore::kernel {
int GatherInt8CPUKernel::Prepare() {
  CHECK_LESS_RETURN(in_tensors_.size(), C2NUM);
  CHECK_LESS_RETURN(out_tensors_.size(), 1);
  CHECK_NULL_RETURN(in_tensors_[0]);
  CHECK_NULL_RETURN(in_tensors_[1]);
  CHECK_NULL_RETURN(out_tensors_[0]);
  if (in_tensors_[0]->data_type() != mindspore::kNumberTypeInt8 ||
      out_tensors_[0]->data_type() != mindspore::kNumberTypeInt8) {
    MS_LOG(ERROR) << "Datatype error, input0 data_type is " << in_tensors_[0]->data_type() << ", output data_type is "
                  << out_tensors_[0]->data_type();
    return RET_ERROR;
  }
  if (in_tensors_.size() == kInputSize2) {
    CHECK_NULL_RETURN(in_tensors_.at(C2NUM));
    auto axis_data = reinterpret_cast<int *>(in_tensors_.at(C2NUM)->data());
    if (axis_data == nullptr) {
      MS_LOG(ERROR) << "GatherInt8CPUkernel input[2] data nullptr.";
      return RET_ERROR;
    }
    axis_ = *axis_data;
  } else {
    axis_ = (reinterpret_cast<GatherParameter *>(op_parameter_))->axis_;
  }
  auto in_quant_args = in_tensors_.at(0)->quant_params();
  CHECK_LESS_RETURN(in_quant_args.size(), 1);
  auto out_quant_args = out_tensors_.at(0)->quant_params();
  CHECK_LESS_RETURN(out_quant_args.size(), 1);
  param_.alpha_ = in_quant_args.front().scale / out_quant_args.front().scale;
  param_.zp_in_ = in_quant_args.front().zeroPoint;
  param_.zp_out_ = out_quant_args.front().zeroPoint;

  if (!InferShapeDone()) {
    return RET_OK;
  }
  return ReSize();
}

int GatherInt8CPUKernel::ReSize() { return RET_OK; }

int GatherInt8CPUKernel::DoGather(int task_id) {
  auto input_tensor = in_tensors_.at(0);
  auto indices_tensor = in_tensors_.at(1);
  auto out_tensor = out_tensors_.at(0);

  auto input_ptr = reinterpret_cast<int8_t *>(input_tensor->data());
  CHECK_NULL_RETURN(input_ptr);
  auto output_ptr = reinterpret_cast<int8_t *>(out_tensor->data());
  CHECK_NULL_RETURN(output_ptr);
  auto in_shape = input_tensor->shape();
  int in_rank = in_shape.size();
  const int limit = in_shape.at(axis_);
  MS_CHECK_LT(axis_, in_rank, RET_ERROR);
  int indices_element_size = indices_tensor->ElementsNum();
  MS_CHECK_GT(indices_element_size, 0, RET_ERROR);

  if (indices_tensor->data_type() == kNumberTypeInt32) {
    auto indices_ptr = reinterpret_cast<int32_t *>(indices_tensor->data());
    CHECK_NULL_RETURN(indices_ptr);
    for (int i = 0; i < indices_element_size; ++i) {
      if (indices_ptr[i] >= limit) {
        MS_LOG(ERROR) << " indice data: " << indices_ptr[i] << " is not in [ 0, " << limit - 1 << " ]";
        return RET_ERROR;
      }
    }
  } else if (indices_tensor->data_type() == kNumberTypeInt64) {
    auto indices_ptr = reinterpret_cast<int64_t *>(indices_tensor->data());
    CHECK_NULL_RETURN(indices_ptr);
    for (int i = 0; i < indices_element_size; ++i) {
      if (indices_ptr[i] >= limit) {
        MS_LOG(ERROR) << " indice data: " << indices_ptr[i] << " is not in [ 0, " << limit - 1 << " ]";
        return RET_ERROR;
      }
    }
  } else {
    MS_LOG(ERROR) << "Unsupported data type:" << indices_tensor->data_type();
    return RET_ERROR;
  }

  int outer_size = 1;
  for (int i = 0; i < axis_; ++i) {
    outer_size *= in_shape.at(i);
  }

  int inner_size = 1;
  for (int i = axis_ + 1; i < in_rank; ++i) {
    inner_size *= in_shape.at(i);
  }

  int stride = UP_DIV(outer_size, thread_count_);
  int count = MSMIN(stride, outer_size - stride * task_id);
  auto thread_stride = stride * task_id;

  input_ptr += thread_stride * inner_size * limit;
  output_ptr += thread_stride * inner_size * indices_element_size;
  if (indices_tensor->data_type() == kNumberTypeInt32) {
    return GatherInt8Int32Index(input_ptr, output_ptr, count, inner_size, limit,
                                reinterpret_cast<int32_t *>(indices_tensor->data()), indices_element_size, param_);
  } else {
    return GatherInt8Int64Index(input_ptr, output_ptr, count, inner_size, limit,
                                reinterpret_cast<int64_t *>(indices_tensor->data()), indices_element_size, param_);
  }
}

int GatherInt8Run(void *cdata, int task_id, float, float) {
  auto gather_kernel = reinterpret_cast<GatherInt8CPUKernel *>(cdata);
  auto error_code = gather_kernel->DoGather(task_id);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "GatherRun error task_id[" << task_id << "] error_code[" << error_code << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int GatherInt8CPUKernel::Run() {
  int error_code = ParallelLaunch(this->ms_context_, GatherInt8Run, this, thread_count_);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "Gather function error error_code[" << error_code << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

kernel::LiteKernel *GatherInt8CPUKernelCreator(const std::vector<lite::Tensor *> &inputs,
                                               const std::vector<lite::Tensor *> &outputs, OpParameter *parameter,
                                               const lite::InnerContext *ctx, const kernel::KernelKey &desc) {
  if (parameter == nullptr) {
    MS_LOG(ERROR) << "parameter is nullptr.";
    return nullptr;
  }

  LiteKernel *kernel = nullptr;
  if (parameter->quant_type_ == schema::QuantType_QUANT_ALL) {
    kernel = new (std::nothrow) GatherInt8CPUKernel(parameter, inputs, outputs, ctx);
  } else if (parameter->quant_type_ == schema::QuantType_QUANT_DYNAMIC) {
    const int axis_index = 2;
    if (inputs.size() > axis_index + 1 && inputs.at(axis_index)) {
      MS_LOG(ERROR) << "kernel: " << parameter->name_ << " is unsupported Axis is not const.";
      return nullptr;
    }
    kernel = new (std::nothrow) DynamicGatherInt8CPUKernel(parameter, inputs, outputs, ctx);
  } else {
    MS_LOG(ERROR) << "kernel: " << parameter->name_ << " is unsupported quant type:" << parameter->quant_type_;
    free(parameter);
    return nullptr;
  }
  if (kernel == nullptr) {
    MS_LOG(ERROR) << "kernel: " << parameter->name_ << "is nullptr.";
    free(parameter);
    return nullptr;
  }
  return kernel;
}

REG_KERNEL(kCPU, kNumberTypeInt8, PrimitiveType_Gather, GatherInt8CPUKernelCreator)
}  // namespace mindspore::kernel
