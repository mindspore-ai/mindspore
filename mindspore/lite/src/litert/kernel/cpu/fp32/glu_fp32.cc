/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "src/litert/kernel/cpu/fp32/glu_fp32.h"
#include <vector>
#include "src/litert/kernel_registry.h"
#include "src/litert/kernel/cpu/base/split_base.h"
#include "nnacl/fp32/activation_fp32.h"
#include "nnacl/fp32/arithmetic_fp32.h"

using mindspore::kernel::KERNEL_ARCH;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_GLU;

namespace mindspore::kernel {
const int kGluBranchNum = 2;
int GluCPUKernel::MallocTmpBuffer() {
  FreeTmpBuffer();
  auto in_tensor = in_tensors_.front();
  for (size_t i = 0; i < kSplitNum; i++) {
    split_ptr_[i] = ms_context_->allocator->Malloc(in_tensor->Size() / kSplitNum);
    if (split_ptr_[i] == nullptr) {
      MS_LOG(ERROR) << "GluCPUKernel malloc split ptr failed.";
      return RET_ERROR;
    }
  }
  sigmoid_ptr_ = reinterpret_cast<int8_t *>(ms_context_->allocator->Malloc(in_tensor->Size() / kSplitNum));
  if (sigmoid_ptr_ == nullptr) {
    MS_LOG(ERROR) << "GluCPUKernel malloc sigmoid ptr failed.";
    return RET_ERROR;
  }
  return RET_OK;
}

void GluCPUKernel::FreeTmpBuffer() {
  for (size_t i = 0; i < kSplitNum; i++) {
    if (split_ptr_.at(i) != nullptr) {
      ms_context_->allocator->Free(split_ptr_.at(i));
      split_ptr_.at(i) = nullptr;
    }
  }
  if (sigmoid_ptr_ != nullptr) {
    ms_context_->allocator->Free(sigmoid_ptr_);
    sigmoid_ptr_ = nullptr;
  }
}

int GluCPUKernel::Prepare() {
  CHECK_LESS_RETURN(in_tensors_.size(), 1);
  CHECK_LESS_RETURN(out_tensors_.size(), 1);
  CHECK_NULL_RETURN(in_tensors_[0]);
  CHECK_NULL_RETURN(out_tensors_[0]);
  if (!InferShapeDone()) {
    return RET_OK;
  }
  return ReSize();
}

int GluCPUKernel::ReSize() {
  split_param_.num_split_ = kSplitNum;
  split_param_.split_dim_ = glu_param_->axis_;
  if (split_param_.split_sizes_ != nullptr) {
    delete[] split_param_.split_sizes_;
  }
  split_param_.split_sizes_ = this->split_sizes_;
  memset(split_param_.split_sizes_, 0, kSplitNum * sizeof(int));

  auto in_tensor = in_tensors_.front();
  auto status = SplitBaseCPUKernel::CheckAndInitSplitParam(*in_tensor, &split_param_);
  if (RET_OK != status) {
    MS_LOG(ERROR) << "CheckAndInitSplitParam failed";
    return status;
  }
  FreeTmpBuffer();

  // split_count means the previous dims num before split dim
  // e.g. input dims is [1, 3, 4, 8], split axis is 2, num_split is 2, so split_count_ is 1*3, num_unit_ is 1*3*2
  num_unit_ = split_param_.split_count_ * split_param_.num_split_;
  usable_thread_num_ = MSMIN(op_parameter_->thread_num_, num_unit_);
  if (usable_thread_num_ != 0) {
    thread_n_stride_ = UP_DIV(num_unit_, usable_thread_num_);
  }
  return RET_OK;
}

int GluCPUKernel::Split(int task_id) const {
  MS_CHECK_INT_MUL_NOT_OVERFLOW(task_id, thread_n_stride_, RET_ERROR);
  int num_unit_thread = MSMIN(thread_n_stride_, num_unit_ - task_id * thread_n_stride_);
  if (num_unit_thread <= 0) {
    return RET_OK;
  }
  int thread_offset = task_id * thread_n_stride_;
  auto ret =
    DoSplit(input_ptr_, const_cast<void **>(split_ptr_.data()), in_tensors_.front()->shape().data(), thread_offset,
            num_unit_thread, &split_param_, lite::DataTypeSize(in_tensors_.front()->data_type()));
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Split error task_id[" << task_id << "] error_code[" << ret << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int GluCPUKernel::Sigmoid(int task_id) const {
  auto input_addr = reinterpret_cast<float *>(split_ptr_.at(1));
  auto output_addr = reinterpret_cast<float *>(sigmoid_ptr_);
  CHECK_NULL_RETURN(input_addr);
  CHECK_NULL_RETURN(output_addr);
  auto length = in_tensors_.at(0)->ElementsNum() / kGluBranchNum;
  MS_CHECK_TRUE_RET(op_parameter_->thread_num_ != 0, RET_ERROR);
  int stride = UP_DIV(length, op_parameter_->thread_num_);
  MS_CHECK_INT_MUL_NOT_OVERFLOW(stride, task_id, RET_ERROR);
  int count = MSMIN(stride, length - stride * task_id);
  if (count <= 0) {
    return RET_OK;
  }
  return ::Sigmoid(input_addr + stride * task_id, count, output_addr + stride * task_id);
}

int GluCPUKernel::Mul(int task_id) const {
  auto input_addr0 = reinterpret_cast<float *>(split_ptr_.at(0));
  auto input_addr1 = reinterpret_cast<float *>(sigmoid_ptr_);
  auto output_addr = reinterpret_cast<float *>(out_tensors_.at(0)->data());
  CHECK_NULL_RETURN(input_addr0);
  CHECK_NULL_RETURN(input_addr1);
  CHECK_NULL_RETURN(output_addr);
  auto length = in_tensors_.at(0)->ElementsNum() / kGluBranchNum;
  MS_CHECK_TRUE_RET(op_parameter_->thread_num_ != 0, RET_ERROR);
  int stride = UP_DIV(length, op_parameter_->thread_num_);
  MS_CHECK_INT_MUL_NOT_OVERFLOW(stride, task_id, RET_ERROR);
  int count = MSMIN(stride, length - stride * task_id);
  if (count <= 0) {
    return RET_OK;
  }
  int offset = stride * task_id;
  return ElementMul(input_addr0 + offset, input_addr1 + offset, output_addr + offset, count);
}

static int SplitRun(void *cdata, int task_id, float, float) {
  auto g_kernel = reinterpret_cast<const GluCPUKernel *>(cdata);
  return g_kernel->Split(task_id);
}

static int SigmoidRun(void *cdata, int task_id, float, float) {
  auto activation_kernel = reinterpret_cast<const GluCPUKernel *>(cdata);
  return activation_kernel->Sigmoid(task_id);
}

static int MulRun(void *cdata, int task_id, float, float) {
  auto g_kernel = reinterpret_cast<const GluCPUKernel *>(cdata);
  return g_kernel->Mul(task_id);
}

int GluCPUKernel::Run() {
  input_ptr_ = in_tensors_.front()->data();

  auto ret = MallocTmpBuffer();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Malloc tmp buffer failed";
    return ret;
  }

  ret = ParallelLaunch(this->ms_context_, SplitRun, this, usable_thread_num_);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "split error error_code[" << ret << "]";
    FreeTmpBuffer();
    return ret;
  }

  ret = ParallelLaunch(this->ms_context_, SigmoidRun, this, op_parameter_->thread_num_);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "sigmoid error error_code[" << ret << "]";
    FreeTmpBuffer();
    return ret;
  }

  ret = ParallelLaunch(this->ms_context_, MulRun, this, op_parameter_->thread_num_);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "mul error error_code[" << ret << "]";
    FreeTmpBuffer();
    return ret;
  }
  FreeTmpBuffer();
  return RET_OK;
}

REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_GLU, LiteKernelCreator<GluCPUKernel>)
}  // namespace mindspore::kernel
