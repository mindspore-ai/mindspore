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
#include "src/runtime/kernel/cpu/fp32/sparse_segment_sum_fp32.h"
#include <vector>
#include <functional>
#include "schema/model_generated.h"
#include "src/runtime/kernel_registry.h"
#include "nnacl/fp32/sparse_segment_sum_fp32.h"
#include "include/errorcode.h"
#include "nnacl/common_func.h"

using mindspore::kernel::KERNEL_ARCH;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_SparseSegmentSum;

namespace mindspore::kernel {
namespace {
const uint32_t kInput_data = 0;
const uint32_t kInput_indices = 1;
const uint32_t kInput_segment_ids = 2;
const uint32_t kOutput_data = 0;
}  // namespace
int SparseSegmentSumCPUKernel::PreProcess() { return RET_OK; }

int SparseSegmentSumCPUKernel::Prepare() { return RET_OK; }

int SparseSegmentSumCPUKernel::RunInferOutDataShape() {
  std::vector<int> in_data_shape = in_tensors_[kInput_data]->shape();
  std::vector<int> in_segment_ids_shape = in_tensors_[kInput_segment_ids]->shape();
  std::vector<int> out_data_shape;

  auto in_segment_ids_ptr = reinterpret_cast<int32_t *>(in_tensors_[kInput_segment_ids]->data());
  if (in_segment_ids_ptr[0] != 0) {
    MS_LOG(ERROR) << "For '" << this->name_ << "', indices should start from 0.";
    return RET_ERROR;
  }
  out_data_shape.emplace_back(in_segment_ids_ptr[in_segment_ids_shape[0] - 1] + 1);
  for (size_t i = 1; i < in_data_shape.size(); i++) {
    out_data_shape.emplace_back(in_data_shape[i]);
  }

  auto origin_out_data_shape = out_tensors_.at(kOutput_data)->shape();
  out_tensors_.at(kOutput_data)->set_shape(out_data_shape);
  out_tensors_.at(kOutput_data)->FreeData();
  out_tensors_.at(kOutput_data)->set_shape_changed(out_data_shape != origin_out_data_shape);
  return RET_OK;
}

int SparseSegmentSumCPUKernel::RunSparseSegmentSumCalc() {
  std::vector<int> in_data_shape = in_tensors_[kInput_data]->shape();
  std::vector<int> in_segment_ids_shape = in_tensors_[kInput_segment_ids]->shape();

  size_t data_num =
    std::accumulate(in_data_shape.begin(), in_data_shape.end(), 1, std::multiplies<int>()) / in_data_shape[0];
  size_t data_ids_num =
    std::accumulate(in_segment_ids_shape.begin(), in_segment_ids_shape.end(), 1, std::multiplies<int>());

  auto *in_data_ptr = in_tensors_[kInput_data]->data();
  auto in_indcie_ptr = reinterpret_cast<int32_t *>(in_tensors_[kInput_indices]->data());
  auto in_segment_ids_ptr = reinterpret_cast<int32_t *>(in_tensors_[kInput_segment_ids]->data());
  auto *out_data_ptr = out_tensors_[kOutput_data]->MutableData();

  auto input_data_type = in_tensors_[kInput_data]->data_type();
  switch (input_data_type) {
    case kNumberTypeInt32:
      SparseSegmentSumCalcInt32(reinterpret_cast<int32_t *>(in_data_ptr), in_indcie_ptr, in_segment_ids_ptr,
                                reinterpret_cast<int32_t *>(out_data_ptr), data_ids_num, data_num);

      break;
    case kNumberTypeFloat32:
      SparseSegmentSumCalcFp32(reinterpret_cast<float *>(in_data_ptr), in_indcie_ptr, in_segment_ids_ptr,
                               reinterpret_cast<float *>(out_data_ptr), data_ids_num, data_num);
      break;
    default:
      MS_LOG(ERROR) << "Unsupported data type: " << input_data_type << " of SparseFillEmptyRows cpu kernel.";
      return RET_ERROR;
  }

  return RET_OK;
}

int SparseSegmentSumCPUKernel::Run() {
  auto ret = RunInferOutDataShape();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Run Infer OutDataShape.";
    return RET_ERROR;
  }

  ret = RunSparseSegmentSumCalc();  // do not use ParallelLaunch, because of writing the same memory
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Run SparseSegmentSum Calc.";
    return RET_ERROR;
  }

  for (auto *output : this->out_tensors()) {
    output->ResetRefCount();
  }

  return RET_OK;
}

REG_KERNEL(kCPU, kNumberTypeInt32, PrimitiveType_SparseSegmentSum, LiteKernelCreator<SparseSegmentSumCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_SparseSegmentSum, LiteKernelCreator<SparseSegmentSumCPUKernel>)
// REG_KERNEL(kCPU, kNumberTypeBool, PrimitiveType_SparseSegmentSum, LiteKernelCreator<SparseSegmentSumCPUKernel>)
}  // namespace mindspore::kernel
