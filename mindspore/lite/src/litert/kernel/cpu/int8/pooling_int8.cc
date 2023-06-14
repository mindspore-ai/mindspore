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

#include "src/litert/kernel/cpu/int8/pooling_int8.h"
#include <cfloat>
#include "nnacl/int8/pooling_int8.h"
#include "include/errorcode.h"
#include "src/litert/kernel_registry.h"

using mindspore::kernel::KERNEL_ARCH;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;

using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_MEMORY_FAILED;
using mindspore::schema::PrimitiveType_AvgPoolFusion;
using mindspore::schema::PrimitiveType_MaxPoolFusion;

namespace mindspore::kernel {
int PoolingInt8CPUKernel::SetQuantParam() {
  // per tensor init
  pooling_quant_arg_ = reinterpret_cast<QuantArg **>(malloc(TWO_TENSOR * sizeof(QuantArg *)));
  if (pooling_quant_arg_ == nullptr) {
    MS_LOG(ERROR) << "malloc pooling_quant_arg failed.";
    return RET_MEMORY_FAILED;
  }
  pooling_quant_arg_[0] = reinterpret_cast<QuantArg *>(malloc(sizeof(QuantArg)));
  if (pooling_quant_arg_[0] == nullptr) {
    pooling_quant_arg_[1] = nullptr;
    MS_LOG(ERROR) << "malloc pooling_quant_arg[0] failed.";
    free(pooling_quant_arg_);
    pooling_quant_arg_ = nullptr;
    return RET_MEMORY_FAILED;
  }
  pooling_quant_arg_[1] = reinterpret_cast<QuantArg *>(malloc(sizeof(QuantArg)));
  if (pooling_quant_arg_[1] == nullptr) {
    MS_LOG(ERROR) << "malloc pooling_quant_arg[1] failed.";
    free(pooling_quant_arg_[0]);
    pooling_quant_arg_[0] = nullptr;
    free(pooling_quant_arg_);
    pooling_quant_arg_ = nullptr;
    return RET_MEMORY_FAILED;
  }
  auto *input_tensor = in_tensors_.at(kInputIndex);
  auto in_quant_arg = input_tensor->quant_params();
  MS_CHECK_TRUE_MSG(input_tensor != nullptr, RET_ERROR, "input_tensor is nullptr.");
  if (in_quant_arg.empty()) {
    MS_LOG(ERROR) << "input tensor quant_params() return empty vector.";
    FreeQuantParam();
    return RET_ERROR;
  }
  auto *out_tensor = out_tensors_.at(kOutputIndex);
  auto out_quant_arg = out_tensor->quant_params();
  MS_CHECK_TRUE_MSG(out_tensor != nullptr, RET_ERROR, "out_tensor is nullptr.");
  if (out_quant_arg.empty()) {
    MS_LOG(ERROR) << "output tensor quant_params() return empty vector.";
    FreeQuantParam();
    return RET_ERROR;
  }
  pooling_quant_arg_[0][0].scale_ = in_quant_arg.front().scale;
  pooling_quant_arg_[0][0].zp_ = in_quant_arg.front().zeroPoint;
  pooling_quant_arg_[1][0].scale_ = out_quant_arg.front().scale;
  pooling_quant_arg_[1][0].zp_ = out_quant_arg.front().zeroPoint;

  if (std::abs(pooling_quant_arg_[0][0].scale_ - pooling_quant_arg_[1][0].scale_) < FLT_EPSILON &&
      pooling_quant_arg_[0][0].zp_ == pooling_quant_arg_[1][0].zp_) {
    quantize_ = false;
  } else {
    quantize_ = true;
  }
  return RET_OK;
}

void PoolingInt8CPUKernel::FreeQuantParam() {
  if (pooling_quant_arg_ != nullptr) {
    for (int i = 0; i < TWO_TENSOR; ++i) {
      if (*(pooling_quant_arg_ + i) != nullptr) {
        free(*(pooling_quant_arg_ + i));
      }
    }
    free(pooling_quant_arg_);
    pooling_quant_arg_ = nullptr;
  }
}

int PoolingInt8CPUKernel::Prepare() {
  MS_CHECK_TRUE_MSG(in_tensors_.size() == 1, RET_ERROR, "input tensor size error.");
  MS_CHECK_TRUE_MSG(out_tensors_.size() == 1, RET_ERROR, "output tensor size error.");
  CHECK_NULL_RETURN(in_tensors_[0]);
  CHECK_NULL_RETURN(out_tensors_[0]);
  CHECK_NULL_RETURN(pooling_param_);
  CHECK_NULL_RETURN(op_parameter_);
  if (op_parameter_->quant_type_ != schema::QuantType_QUANT_NONE &&
      op_parameter_->quant_type_ != schema::QuantType_QUANT_ALL) {
    MS_LOG(ERROR) << "Invalid quant type: " << op_parameter_->quant_type_;
    return RET_ERROR;
  }

  if (in_tensors_[0]->data_type() != mindspore::kNumberTypeInt8 ||
      out_tensors_[0]->data_type() != mindspore::kNumberTypeInt8) {
    MS_LOG(ERROR) << "Datatype error, input0 data_type is " << in_tensors_[0]->data_type() << ", output data_type is "
                  << out_tensors_[0]->data_type();
    return RET_ERROR;
  }

  int ret = SetQuantParam();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Set pooling quant param failed.";
    return RET_ERROR;
  }
  if (!InferShapeDone()) {
    return RET_OK;
  }
  return ReSize();
}

int PoolingInt8CPUKernel::ReSize() {
  auto in_tensor = this->in_tensors_.front();
  auto out_tensor = this->out_tensors_.front();
  compute_.input_batch_ = in_tensor->Batch();
  compute_.input_channel_ = in_tensor->Channel();
  compute_.input_h_ = in_tensor->Height();
  compute_.input_w_ = in_tensor->Width();
  compute_.output_batch_ = out_tensor->Batch();
  compute_.output_channel_ = out_tensor->Channel();
  compute_.output_h_ = out_tensor->Height();
  compute_.output_w_ = out_tensor->Width();
  compute_.window_h_ = pooling_param_->window_h_;
  compute_.window_w_ = pooling_param_->window_w_;
  if (pooling_param_->global_) {
    pooling_param_->window_h_ = compute_.input_h_;
    pooling_param_->window_w_ = compute_.input_w_;
  }
  compute_.minf = INT8_MIN;
  compute_.maxf = INT8_MAX;

  FreeQuantParam();
  int ret = SetQuantParam();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Set pooling quant param failed.";
    return ret;
  }
  return RET_OK;
}

int PoolingInt8CPUKernel::RunImpl(int task_id) {
  auto input_data = reinterpret_cast<int8_t *>(in_tensors_.at(kInputIndex)->MutableData());
  CHECK_NULL_RETURN(input_data);
  auto output_data = reinterpret_cast<int8_t *>(out_tensors_.at(kOutputIndex)->MutableData());
  CHECK_NULL_RETURN(output_data);
  CHECK_NULL_RETURN(pooling_param_);
  if (pooling_param_->pool_mode_ == PoolMode_MaxPool) {
    if (quantize_) {
      MaxPoolingWithQuantInt8(input_data, output_data, pooling_param_, &compute_, pooling_quant_arg_, task_id,
                              op_parameter_->thread_num_);
    } else {
      MaxPoolingOptInt8(input_data, output_data, pooling_param_, &compute_, task_id, op_parameter_->thread_num_);
    }
  } else {
    auto ret = AvgPoolingOptInt8(input_data, output_data, pooling_param_, &compute_, pooling_quant_arg_, task_id,
                                 op_parameter_->thread_num_);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "AvgPooling run failed.";
      return ret;
    }
  }
  return RET_OK;
}

int PoolingInt8Impl(void *cdata, int task_id, float, float) {
  auto pooling = reinterpret_cast<PoolingInt8CPUKernel *>(cdata);
  auto error_code = pooling->RunImpl(task_id);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "PoolingInt8 Run error task_id[" << task_id << "] error_code[" << error_code << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int PoolingInt8CPUKernel::Run() {
  int error_code = ParallelLaunch(this->ms_context_, PoolingInt8Impl, this, op_parameter_->thread_num_);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "poolingInt8 error error_code[" << error_code << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

REG_KERNEL(kCPU, kNumberTypeInt8, PrimitiveType_AvgPoolFusion, LiteKernelCreator<PoolingInt8CPUKernel>)
REG_KERNEL(kCPU, kNumberTypeInt8, PrimitiveType_MaxPoolFusion, LiteKernelCreator<PoolingInt8CPUKernel>)
}  // namespace mindspore::kernel
