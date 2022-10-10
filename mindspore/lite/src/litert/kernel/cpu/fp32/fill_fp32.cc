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

#include "src/litert/kernel/cpu/fp32/fill_fp32.h"
#include <vector>
#include "schema/model_generated.h"
#include "src/litert/kernel_registry.h"
#include "include/errorcode.h"

using mindspore::kernel::KERNEL_ARCH;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_NULL_PTR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_Fill;

namespace mindspore::kernel {
int FillCPUKernel::Prepare() {
  CHECK_LESS_RETURN(in_tensors_.size(), kInputSize1);
  CHECK_LESS_RETURN(out_tensors_.size(), 1);
  if (!InferShapeDone()) {
    return RET_OK;
  }
  return ReSize();
}

int FillCPUKernel::ReSize() {
  if (UpdateThreadNumPass(TC_PTYPE(PrimitiveType_Fill), 0, 1, out_tensors_.front()->Size()) != RET_OK) {
    return RET_ERROR;
  }
  auto output = out_tensors_.front();
  CHECK_NULL_RETURN(output);
  data_size_ = output->ElementsNum();
  thread_sz_count_ = MSMIN(thread_num_, data_size_);
  if (thread_sz_count_ != 0) {
    thread_sz_stride_ = UP_DIV(data_size_, thread_sz_count_);
  }
  return RET_OK;
}

int FillCPUKernel::DoFill(int task_id) {
  MS_CHECK_INT_MUL_NOT_OVERFLOW(task_id, thread_sz_stride_, RET_ERROR);
  int size = MSMIN(thread_sz_stride_, data_size_ - task_id * thread_sz_stride_);
  if (size <= 0) {
    return RET_OK;
  }
  int offset = task_id * thread_sz_stride_;
  auto input_tensor = in_tensors_.at(0);
  int ret = RET_OK;
  if (input_tensor->data_type() == kNumberTypeFloat32 || input_tensor->data_type() == kNumberTypeFloat) {
    ret = FillFp32(out_ptr_ + offset, size, src_data_);
  } else if (input_tensor->data_type() == kNumberTypeInt32 || input_tensor->data_type() == kNumberTypeInt) {
    ret = FillInt32(int32_out_ptr_ + offset, size, int32_src_data_);
  } else if (input_tensor->data_type() == kNumberTypeBool) {
    ret = FillBool(bool_out_ptr_ + offset, size, bool_src_data_);
  } else {
    return RET_ERROR;
  }
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "FillRun error task_id[" << task_id << "] error_code[" << ret << "]";
    return ret;
  }
  return RET_OK;
}

int FillRun(void *cdata, int task_id, float lhs_scale, float rhs_scale) {
  auto g_kernel = reinterpret_cast<FillCPUKernel *>(cdata);
  CHECK_NULL_RETURN(g_kernel);
  auto ret = g_kernel->DoFill(task_id);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "FillRun error task_id[" << task_id << "] error_code[" << ret << "]";
    return ret;
  }
  return RET_OK;
}

int FillCPUKernel::Run() {
  auto fill_input = in_tensors_.front();
  CHECK_NULL_RETURN(fill_input);
  auto output = out_tensors_.front();
  CHECK_NULL_RETURN(output);
  if (fill_input->data_type() == kNumberTypeFloat32 || fill_input->data_type() == kNumberTypeFloat) {
    auto fill_data = reinterpret_cast<float *>(fill_input->data());
    CHECK_NULL_RETURN(fill_data);
    src_data_ = fill_data[0];
    out_ptr_ = reinterpret_cast<float *>(output->MutableData());
    CHECK_NULL_RETURN(out_ptr_);
  } else if (fill_input->data_type() == kNumberTypeInt32 || fill_input->data_type() == kNumberTypeInt) {
    auto fill_data = reinterpret_cast<int *>(fill_input->data());
    CHECK_NULL_RETURN(fill_data);
    MS_CHECK_TRUE_RET(fill_input->shape().empty(), RET_ERROR);
    int32_src_data_ = fill_data[0];
    int32_out_ptr_ = reinterpret_cast<int *>(output->MutableData());
    CHECK_NULL_RETURN(int32_out_ptr_);
  } else if (fill_input->data_type() == kNumberTypeBool) {
    auto fill_data = reinterpret_cast<bool *>(fill_input->data());
    CHECK_NULL_RETURN(fill_data);
    bool_src_data_ = fill_data[0];
    bool_out_ptr_ = reinterpret_cast<bool *>(output->MutableData());
    CHECK_NULL_RETURN(bool_out_ptr_);
  } else {
    MS_LOG(ERROR) << "unsupported fill data type " << fill_input->data_type();
    return RET_ERROR;
  }
  auto ret = ParallelLaunch(this->ms_context_, FillRun, this, thread_sz_count_);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "FillRun error error_code[" << ret << "]";
    return ret;
  }
  return RET_OK;
}

REG_KERNEL(kCPU, kNumberTypeBool, PrimitiveType_Fill, LiteKernelCreator<FillCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeInt32, PrimitiveType_Fill, LiteKernelCreator<FillCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_Fill, LiteKernelCreator<FillCPUKernel>)
}  // namespace mindspore::kernel
