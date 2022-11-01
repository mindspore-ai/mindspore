/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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
#include "src/litert/kernel/cpu/fp16/dynamic_quant_fp16.h"
#include <vector>
#include <algorithm>
#include "src/litert/kernel_registry.h"
#include "schema/model_generated.h"
#include "include/errorcode.h"
#include "nnacl/dynamic_quant_parameter.h"
#include "nnacl/fp16/dynamic_quant_fp16.h"
#include "nnacl/fp16/quant_dtype_cast_fp16.h"

using mindspore::kernel::KERNEL_ARCH;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_NULL_PTR;
using mindspore::lite::RET_OK;
using mindspore::lite::RET_PARAM_INVALID;
using mindspore::schema::PrimitiveType_DynamicQuant;
namespace mindspore::kernel {
namespace {
constexpr int kBucketNums = 8;
constexpr int k8Bit = 8;
constexpr int kMinNums = 512;
constexpr float kDefaultRange = 0.01;
}  // namespace
int DynamicQuantFp16CPUKernel::Prepare() {
  CHECK_LESS_RETURN(in_tensors_.size(), C1NUM);
  CHECK_LESS_RETURN(out_tensors_.size(), C1NUM);
  auto in_tensor = in_tensors_.front();
  CHECK_NULL_RETURN(in_tensor);
  auto out_tensor = out_tensors_.front();
  CHECK_NULL_RETURN(out_tensor);
  auto param = reinterpret_cast<DynamicQuantParameter *>(op_parameter_);
  CHECK_NULL_RETURN(param);
  src_dtype_ = in_tensor->data_type();
  dst_dtype_ = param->dst_type_;
  symmetric_ = param->symmetric_;
  if (out_tensor->data_type() != dst_dtype_) {
    MS_LOG(ERROR) << "param data type and tensor data type do not match.";
    return RET_ERROR;
  }
  if (!InferShapeDone()) {
    return RET_OK;
  }
  return ReSize();
}

int DynamicQuantFp16CPUKernel::ReSize() {
  auto in_tensor = in_tensors_.front();
  num_unit_ = static_cast<int>(in_tensor->ElementsNum());
  if (num_unit_ < kMinNums) {
    thread_n_num_ = 1;
  } else {
    thread_n_num_ = MSMIN(thread_num_, num_unit_);
    // Limit for 8 thread
    thread_n_num_ = MSMIN(thread_n_num_, kBucketNums);
  }
  for (int i = 0; i < kBucketNums; ++i) {
    real_min_array_[i] = FLT16_MAX;
    real_max_array_[i] = -FLT16_MAX;
  }
  MS_CHECK_GT(thread_n_num_, 0, RET_ERROR);
  thread_n_stride_ = UP_DIV(num_unit_, thread_n_num_);
  return RET_OK;
}

int DynamicQuantFp16CPUKernel::CalculateMinMax(int task_id) {
  int num_unit_thread = MSMIN(thread_n_stride_, num_unit_ - task_id * thread_n_stride_);
  if (num_unit_thread <= 0) {
    return RET_OK;
  }
  int thread_offset = task_id * thread_n_stride_;
  float16_t *data = float16_ptr_ + thread_offset;
  CalculateMinMaxFp16(data, num_unit_thread, &real_min_array_[task_id], &real_max_array_[task_id]);
  return RET_OK;
}

int CalculateFp16MinMaxRun(void *cdata, int task_id, float, float) {
  CHECK_NULL_RETURN(cdata);
  auto g_kernel = reinterpret_cast<DynamicQuantFp16CPUKernel *>(cdata);
  auto ret = g_kernel->CalculateMinMax(task_id);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "CalculateFp16MinMaxRun error task_id[" << task_id << "] error_code[" << ret << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

void DynamicQuantFp16CPUKernel::ReduceMinMaxFp32() {
  for (int i = 0; i < kBucketNums; i++) {
    if (real_min_array_[i] < real_min_) {
      real_min_ = real_min_array_[i];
    }
    if (real_max_array_[i] > real_max_) {
      real_max_ = real_max_array_[i];
    }
  }
  return;
}

void DynamicQuantFp16CPUKernel::CalculateScaleZp() {
  lite::LiteQuantParam quant_parm;
  double scale;
  int zp = 0;
  constexpr int kQSymmetricRange = 255;
  constexpr int kQAsymmetricRange = 254;
  if (!symmetric_) {
    auto range = real_max_ - real_min_;
    if (range <= 0) {
      range = kDefaultRange;
      MS_LOG(WARNING) << name_ << " range is 0 and set the range to 0.01.";
    }
    scale = range / kQSymmetricRange;  // -128 ~ 127
    zp = static_cast<int>(std::round(INT8_MIN - real_min_ / scale));
  } else {
    auto max = std::max(abs(real_max_), abs(real_min_));
    scale = 2 * max / kQAsymmetricRange;  // -127 ~ 127
  }
  quant_parm.scale = scale;
  quant_parm.zeroPoint = zp;
  quant_parm.bitNum = k8Bit;
  quant_parm.inited = true;
  this->out_tensors_.front()->set_quant_params({quant_parm});
  return;
}

int DynamicQuantFp16CPUKernel::QuantData(int task_id) {
  int num_unit_thread = MSMIN(thread_n_stride_, num_unit_ - task_id * thread_n_stride_);
  if (num_unit_thread <= 0) {
    return RET_OK;
  }
  int thread_offset = task_id * thread_n_stride_;
  auto quant_arg = out_tensors_.front()->quant_params().front();
  int ret;
  TypeId data_type = out_tensors_.front()->data_type();
  if (data_type == TypeId::kNumberTypeInt8) {
    ret = DoQuantizeFp16ToInt8(float16_ptr_ + thread_offset, int8_ptr_ + thread_offset, quant_arg.scale,
                               quant_arg.zeroPoint, num_unit_thread);
  } else {
    MS_LOG(ERROR) << "Data type not supported:" << data_type;
    return RET_PARAM_INVALID;
  }
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "QuantDTypeCast error task_id[" << task_id << "] error_code[" << ret << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int QuantFp16DataRun(void *cdata, int task_id, float, float) {
  CHECK_NULL_RETURN(cdata);
  auto g_kernel = reinterpret_cast<DynamicQuantFp16CPUKernel *>(cdata);
  auto ret = g_kernel->QuantData(task_id);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "CalculateFp16MinMaxRun error task_id[" << task_id << "] error_code[" << ret << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int DynamicQuantFp16CPUKernel::Run() {
  bool support_dtype = in_tensors_[0]->data_type() == TypeId::kNumberTypeFloat16 &&
                       out_tensors_[0]->data_type() == TypeId::kNumberTypeInt8;
  if (!support_dtype) {
    MS_LOG(ERROR) << "Unsupported data type input:" << in_tensors_.front()->data_type()
                  << ", output:" << out_tensors_.front()->data_type();
    return RET_ERROR;
  }
  int8_ptr_ = reinterpret_cast<int8_t *>(out_tensors_[0]->data());
  CHECK_NULL_RETURN(int8_ptr_);
  float16_ptr_ = reinterpret_cast<float16_t *>(in_tensors_[0]->data());
  CHECK_NULL_RETURN(float16_ptr_);
  auto ret = ParallelLaunch(this->ms_context_, CalculateFp16MinMaxRun, this, thread_n_num_);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Run error error_code[" << ret << "]";
    return RET_ERROR;
  }
  ReduceMinMaxFp32();
  CalculateScaleZp();
  ret = ParallelLaunch(this->ms_context_, QuantFp16DataRun, this, thread_n_num_);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Run error error_code[" << ret << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

kernel::LiteKernel *DynamicQuantFp16CPUCreator(const std::vector<lite::Tensor *> &inputs,
                                               const std::vector<lite::Tensor *> &outputs, OpParameter *parameter,
                                               const lite::InnerContext *ctx, const kernel::KernelKey &desc) {
  if (parameter == nullptr) {
    MS_LOG(ERROR) << "parameter is nullptr.";
    return nullptr;
  }
  auto *kernel = new (std::nothrow) DynamicQuantFp16CPUKernel(parameter, inputs, outputs, ctx);
  if (kernel == nullptr) {
    MS_LOG(ERROR) << "kernel: " << parameter->name_ << "is nullptr.";
    free(parameter);
    return nullptr;
  }
  if (inputs.size() != 1) {
    MS_LOG(ERROR) << "inputs number should be 1, but " << inputs.size() << " is given.";
    return nullptr;
  }
  if (outputs.size() != 1) {
    MS_LOG(ERROR) << "outputs number should be 1, but " << outputs.size() << " is given.";
    return nullptr;
  }

  return kernel;
}

REG_KERNEL(kCPU, kNumberTypeFloat16, PrimitiveType_DynamicQuant, DynamicQuantFp16CPUCreator)
}  // namespace mindspore::kernel
