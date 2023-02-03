/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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
#include "src/runtime/kernel/cpu/fp32/sparse_reshape_fp32.h"
#include <vector>
#include "schema/model_generated.h"
#include "nnacl/fp32/sparse_reshape_fp32.h"
#include "src/runtime/kernel_registry.h"
#include "include/errorcode.h"
#include "nnacl/common_func.h"

using mindspore::kernel::KERNEL_ARCH;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_SparseReshape;

namespace mindspore::kernel {
namespace {
const uint32_t kInput_indices = 0;
const uint32_t kInput_inshape = 1;
const uint32_t kInput_outshape = 2;
const uint32_t kOutput_indices = 0;
const uint32_t kOutput_outshape = 1;
}  // namespace

int SparseReshapeCPUKernel::Prepare() { return RET_OK; }

int SparseReshapeCPUKernel::SoftCopyInputToOutput(lite::Tensor *src_tensor, lite::Tensor *dst_tensor) {
  CHECK_NULL_RETURN(src_tensor);
  CHECK_NULL_RETURN(dst_tensor);

  if (src_tensor->allocator() == nullptr || src_tensor->allocator() != dst_tensor->allocator() ||
      src_tensor->allocator() != ms_context_->allocator || /* runtime allocator */
      op_parameter_->is_train_session_) {
    CHECK_NULL_RETURN(dst_tensor->data());
    CHECK_NULL_RETURN(src_tensor->data());
    MS_CHECK_FALSE(src_tensor->Size() == 0, RET_ERROR);
    auto size = src_tensor->Size();
    thread_num_ = MSMIN(static_cast<size_t>(op_parameter_->thread_num_), UP_DIV(size, 16384));  // PerThreadMin : 16384
    if (thread_num_ < 1) {
      thread_num_ = 1;
    }
    auto block_size = UP_DIV(size, thread_num_);
    thread_num_ = UP_DIV(size, block_size);
    auto input_data = static_cast<const uint8_t *>(src_tensor->data());
    auto output_data = static_cast<uint8_t *>(dst_tensor->data());
    auto Copy = [input_data, output_data, size, block_size, this](void *, int task_id, float, float) {
      auto in_start = input_data + task_id * block_size;
      auto out_start = output_data + task_id * block_size;
      auto copy_size = block_size;
      if (task_id == (thread_num_ - 1)) {
        copy_size = size - task_id * block_size;
      }
      memcpy(out_start, in_start, copy_size);
      return RET_OK;
    };
    if (input_data != output_data) {
      if (thread_num_ == 1) {
        memcpy(output_data, input_data, size);
        return RET_OK;
      }
      return lite::ParallelLaunch(this->ms_context_, Copy, nullptr, thread_num_);
    }
    return RET_OK;
  }

  dst_tensor->FreeData();
  dst_tensor->ResetRefCount();
  dst_tensor->set_data(src_tensor->data());
  if (src_tensor->IsConst()) {
    dst_tensor->set_own_data(false);
  } else {
    dst_tensor->set_own_data(src_tensor->own_data());
  }
  return RET_OK;
}

int SparseReshapeCPUKernel::Run() {
  auto in_indices_ptr = reinterpret_cast<int32_t *>(in_tensors_[kInput_indices]->data());
  auto in_inshape_ptr = reinterpret_cast<int32_t *>(in_tensors_[kInput_inshape]->data());
  auto in_outshape_ptr = reinterpret_cast<int32_t *>(in_tensors_[kInput_outshape]->data());

  auto out_indices_ptr = reinterpret_cast<int32_t *>(out_tensors_[kOutput_indices]->data());
  auto out_outshape_ptr = reinterpret_cast<int32_t *>(out_tensors_[kOutput_outshape]->data());

  auto input_rank = in_tensors_[kInput_inshape]->ElementsNum();
  auto output_rank = in_tensors_[kInput_outshape]->ElementsNum();

  SparseReshapeInferOutputShapeFp32(in_inshape_ptr, in_outshape_ptr, out_outshape_ptr, input_rank, output_rank);

  bool inshape_same_to_outshape = (input_rank == output_rank);
  if (inshape_same_to_outshape) {
    for (int i = 0; i < input_rank; i++) {
      if (in_inshape_ptr[i] != in_outshape_ptr[i]) {
        inshape_same_to_outshape = false;
        break;
      }
    }
  }

  if (inshape_same_to_outshape) {
    return SoftCopyInputToOutput(in_tensors_[kInput_indices], out_tensors_[kOutput_indices]);
  }

  std::vector<int32_t> in_stride(input_rank);
  std::vector<int32_t> out_stride(output_rank);
  auto in_indices_shape = in_tensors_[kInput_indices]->shape()[0];
  SparseReshapeInOutCoordTrans(in_indices_ptr, in_inshape_ptr, out_outshape_ptr, in_indices_shape, out_indices_ptr,
                               in_stride.data(), out_stride.data(), input_rank, output_rank);

  return RET_OK;
}

REG_KERNEL(kCPU, kNumberTypeInt32, PrimitiveType_SparseReshape, LiteKernelCreator<SparseReshapeCPUKernel>)
// REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_SparseReshape, LiteKernelCreator<SparseReshapeCPUKernel>)
}  // namespace mindspore::kernel
