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
#include "src/litert/kernel/cpu/fp32/sparse_reshape_fp32.h"
#include <vector>
#include "schema/model_generated.h"
// #include "nnacl/fp32/sparse_reshape_fp32.h"
#include "src/litert/kernel_registry.h"
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
  //   const int64_t nnz = SizeToLong(indices_shape_[0]);

  int64_t dense_size = 1;
  int64_t dividend = 1;
  int64_t out_num = 1;
  int64_t ui = -1;
  for (int64_t i = 0; i < input_rank; i++) {
    dense_size *= *(in_inshape_ptr + i);
  }

  for (int64_t d = 0; d < output_rank; d++) {
    const int32_t size = *(in_outshape_ptr + d);
    if (size == -1) {
      if (ui != -1) {
        MS_LOG(ERROR) << "For '" << this->name_
                      << "', there should be at most one '-1' dimension in 'newshape' tensor, but got two or more.";
        return RET_ERROR;
      }
      ui = d;
    } else {
      if (size < 0) {
        MS_LOG(ERROR) << "For '" << this->name_ << "', the size of newshape rank-" << d
                      << " should be a non-negative number, but got " << size << ".";
        return RET_ERROR;
      }
      dividend *= size;
      *(out_outshape_ptr + d) = size;
      out_num *= size;
    }
  }
  if (ui != -1) {
    // (void)CheckAndConvertUtils::CheckInteger("divident", dividend, kGreaterThan, 0, this->name_);
    const int64_t missing = dense_size / dividend;
    if (dividend * missing != dense_size) {
      MS_LOG(ERROR) << "For '" << this->name_ << "', the requested shape should be a multiple of " << dividend
                    << " and " << missing << ", but got a SparseTensor with " << dense_size << " dense values.";
      return RET_ERROR;
    }
    out_num *= missing;
    *(out_outshape_ptr + ui) = missing;
  }

  if (out_num != dense_size) {
    MS_LOG(ERROR) << "For '" << this->name_ << "', the requested shape has the dense shape of " << out_num
                  << ", but got the input newshape is a tensor with " << dense_size;
    return RET_ERROR;
  }

  auto in_indices_shape = in_tensors_[kInput_indices]->shape();
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

  std::vector<int64_t> in_stride(input_rank);
  std::vector<int64_t> out_stride(output_rank);
  in_stride[input_rank - 1] = 1;
  for (int64_t d = input_rank - 2; d >= 0; d--) {
    in_stride[d] = in_stride[d + 1] * in_inshape_ptr[d + 1];
  }

  out_stride[output_rank - 1] = 1;
  for (int64_t d = output_rank - 2; d >= 0; d--) {
    out_stride[d] = out_stride[d + 1] * out_outshape_ptr[d + 1];
  }

  for (int i = 0; i < in_indices_shape[0]; i++) {
    int ori_index = 0;
    for (int32_t j = 0; j < input_rank; j++) {
      ori_index += in_indices_ptr[i * input_rank + j] * in_stride[j];
    }

    for (int32_t j = 0; j < output_rank; j++) {
      out_indices_ptr[i * output_rank + j] = ori_index / out_stride[j];
      ori_index %= out_stride[j];
    }
  }

  return RET_OK;
}

REG_KERNEL(kCPU, kNumberTypeInt32, PrimitiveType_SparseReshape, LiteKernelCreator<SparseReshapeCPUKernel>)
// REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_SparseReshape, LiteKernelCreator<SparseReshapeCPUKernel>)
}  // namespace mindspore::kernel
