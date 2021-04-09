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
#include "src/runtime/kernel/arm/fp32/sparse_to_dense_fp32.h"

#include <vector>
#include <limits>

#include "include/errorcode.h"
#include "mindspore/lite/nnacl/fp32/sparse_to_dense_fp32.h"
#include "schema/model_generated.h"
#include "schema/ops_generated.h"
#include "src/kernel_registry.h"
#include "src/runtime/runtime_api.h"

using mindspore::kernel::KERNEL_ARCH::kCPU;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_SparseToDense;

namespace mindspore::kernel {
int SparseToDenseCPUKernel::Init() {
  auto input2 = in_tensors_.at(2);
  MS_ASSERT(input2);
  auto input3 = in_tensors_.at(3);
  MS_ASSERT(input3);
  sparse_values = reinterpret_cast<float *>(input2->MutableData());
  default_value = reinterpret_cast<float *>(input3->MutableData())[0];
  if (input2->ElementsNum() == 1) {
    isScalar = true;
  }
  if (!InferShapeDone()) {
    return RET_OK;
  }
  return ReSize();
}

int SparseToDenseCPUKernel::ReSize() {
  auto output0 = out_tensors_.at(0);
  std::vector<int> out_shape_tensor = output0->shape();
  auto output_shape_tmp = reinterpret_cast<int *>(out_shape_tensor.data());
  int output_dim = output0->shape().size();
  for (int i = 0; i < DIMENSION_4D - output_dim; i++) {
    output_shape[i] = 1;
  }
  for (int i = 0; i < output_dim; i++) {
    output_shape[i + DIMENSION_4D - output_dim] = output_shape_tmp[i];
  }
  output_num = output0->ElementsNum();
  return RET_OK;
}

int SparseToDenseCPUKernel::DoExcute(int task_id) {
  int real_dst_count = MSMIN(index_num - task_id * count_unit_, count_unit_);
  if (real_dst_count <= 0) {
    return RET_OK;
  }
  int index_start = task_id * count_unit_;
  int index_end = index_start + real_dst_count;
  int out_width = output_num / index_num;
  MS_ASSERT(sparse_indices_vect);
  MS_ASSERT(output_shape);
  MS_ASSERT(sparse_values);
  MS_ASSERT(output_data);
  SparseToDense(sparse_indices_vect, output_shape, sparse_values, default_value, output_data, isScalar, index_start,
                index_end, out_width);
  return RET_OK;
}

int SparseToDenseRun(void *cdata, int task_id) {
  auto s2ddata = reinterpret_cast<SparseToDenseCPUKernel *>(cdata);
  auto ret = s2ddata->DoExcute(task_id);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "SparseToDenseRun error task_id[" << task_id << "] error_code[" << ret << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int SparseToDenseCPUKernel::GenerateIndices() {
  auto input0 = in_tensors_.at(0);
  index_num = input0->shape().at(0);
  if (index_num >= std::numeric_limits<int>::max() / static_cast<int>(sizeof(int *))) {
    MS_LOG(ERROR) << "Input dim is invalid, dim: " << index_num;
    return RET_ERROR;
  }
  sparse_indices_vect = reinterpret_cast<int **>(ctx_->allocator->Malloc(sizeof(int *) * index_num));
  if (sparse_indices_vect == nullptr) {
    MS_LOG(ERROR) << "Null pointer reference: sparse_indices_vect.";
    return RET_ERROR;
  }
  index_dim = input0->shape().size();
  int *sparse_indices = reinterpret_cast<int *>(input0->MutableData());
  switch (index_dim) {
    case 0:
    case 1: {
      for (int i = 0; i < index_num; i++) {
        sparse_indices_vect[i] = new int[DIMENSION_4D];
        if (sparse_indices_vect[i] == nullptr) {
          MS_LOG(ERROR) << "Null pointer reference: sparse_indices_vect[" << i << "].";
          return RET_ERROR;
        }
        for (int j = 0; j < DIMENSION_4D - 1; j++) {
          sparse_indices_vect[i][j] = 0;
        }
        sparse_indices_vect[i][DIMENSION_4D - 1] = sparse_indices[i];
      }
      break;
    }
    case 2: {
      int true_dims = input0->shape().at(1);
      MS_ASSERT(true_dims <= DIMENSION_4D);
      for (int i = 0; i < index_num; i++) {
        sparse_indices_vect[i] = new int[DIMENSION_4D];
        if (sparse_indices_vect[i] == nullptr) {
          MS_LOG(ERROR) << "Null pointer reference: sparse_indices_vect[" << i << "].";
          return RET_ERROR;
        }
        for (int j = 0; j < DIMENSION_4D - true_dims; j++) {
          sparse_indices_vect[i][j] = 0;
        }
        for (int j = 0; j < true_dims; j++) {
          sparse_indices_vect[i][j + DIMENSION_4D - true_dims] = sparse_indices[i * true_dims + j];
        }
      }
      break;
    }
    default: {
      MS_LOG(ERROR) << "Indices dimensions is " << index_dim << ", which must be 0, 1 or 2";
      return RET_ERROR;
    }
  }
  return RET_OK;
}

int SparseToDenseCPUKernel::IndicesValidCheck() {
  int d1 = output_shape[1] * output_shape[2] * output_shape[3];
  int d2 = output_shape[2] * output_shape[3];
  int d3 = output_shape[3];
  int index_before = -1;
  for (int i = 0; i < index_num; i++) {
    int index = d1 * sparse_indices_vect[i][0] + d2 * sparse_indices_vect[i][1] + d3 * sparse_indices_vect[i][2] +
                sparse_indices_vect[i][3];
    if (index <= index_before) {
      return RET_ERROR;
    }
    index_before = index;
  }
  return RET_OK;
}

int SparseToDenseCPUKernel::Run() {
  auto ret = GenerateIndices();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Generate Indices failed.";
    return RET_ERROR;
  }
  if (s2d_param->validate_indices_ == true) {
    auto ret2 = IndicesValidCheck();
    if (ret2 != RET_OK) {
      MS_LOG(ERROR) << "The sparse indices is not valid.";
      return RET_ERROR;
    }
  }
  output_data = reinterpret_cast<float *>(out_tensors_.at(0)->MutableData());
  count_unit_ = thread_count_ > 1 ? UP_DIV(index_num, thread_count_) : index_num;
  ret = ParallelLaunch(this->context_->thread_pool_, SparseToDenseRun, this, s2d_param->thread_num_);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "SparseToDenseRun error: error_code[" << ret << "]";
    return RET_ERROR;
  }

  if (sparse_indices_vect != nullptr) {
    for (int i = 0; i < index_num; i++) {
      if (sparse_indices_vect[i] != nullptr) {
        delete sparse_indices_vect[i];
      }
    }
    ctx_->allocator->Free(sparse_indices_vect);
    sparse_indices_vect = nullptr;
  }

  return RET_OK;
}

REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_SparseToDense, LiteKernelCreator<SparseToDenseCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeInt32, PrimitiveType_SparseToDense, LiteKernelCreator<SparseToDenseCPUKernel>)
}  // namespace mindspore::kernel
