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
#include "src/runtime/kernel/cpu/fp32/sparse_fill_empty_rows_fp32.h"
#include <vector>
#include <algorithm>
#include "schema/model_generated.h"
#include "src/runtime/kernel_registry.h"
#include "include/errorcode.h"
#include "nnacl/common_func.h"

using mindspore::kernel::KERNEL_ARCH;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_SparseFillEmptyRows;

namespace mindspore::kernel {
namespace {
const uint32_t kInput_indices = 0;
const uint32_t kInput_values = 1;
const uint32_t kInput_dense_shape = 2;
const uint32_t kInput_default_value = 3;
const uint32_t kOutput_y_indices = 0;
const uint32_t kOutput_y_values = 1;
const uint32_t kOutput_empty_row_indicator = 2;
const uint32_t kOutput_reverse_index_map = 3;
}  // namespace

int SparseFillEmptyRowsCPUKernel::PreProcess() { return RET_OK; }

int SparseFillEmptyRowsCPUKernel::Prepare() { return RET_OK; }

void SparseFillEmptyRowsCPUKernel::UpdataTensorShape(lite::Tensor *tensor, std::vector<int> *new_shape) {
  auto origin_shape = tensor->shape();
  tensor->set_shape(*new_shape);
  tensor->FreeData();
  tensor->set_shape_changed(*new_shape != origin_shape);
}

int SparseFillEmptyRowsCPUKernel::RunInferOutputShape() {
  auto dense_shape_ptr = reinterpret_cast<int32_t *>(in_tensors_[kInput_dense_shape]->data());

  std::vector<int> in_indcie_shape = in_tensors_[kInput_indices]->shape();

  dense_rows_ = dense_shape_ptr[0];
  N_ = in_indcie_shape[0];
  rank_ = in_indcie_shape[1];

  std::vector<int> out_indcie_shape;
  std::vector<int> out_values_shape;
  std::vector<int> out_empty_row_indicator_shape;
  std::vector<int> out_reverse_index_shape;

  out_empty_row_indicator_shape.push_back(dense_rows_);
  out_reverse_index_shape.push_back(N_);

  UpdataTensorShape(out_tensors_.at(kOutput_empty_row_indicator), &out_empty_row_indicator_shape);
  if (out_tensors_.size() == C4NUM) {
    UpdataTensorShape(out_tensors_.at(kOutput_reverse_index_map), &out_reverse_index_shape);
  }

  if (dense_rows_ == 0) {
    if (N_ != 0) {
      MS_LOG(ERROR) << "For '" << this->name_ << "' dense_shape[0] = 0, but indices.shape[0] =  " << N_;
      return RET_ERROR;
    }
    out_indcie_shape.push_back(0);
    out_indcie_shape.push_back(rank_);
    out_values_shape.push_back(0);

    UpdataTensorShape(out_tensors_.at(kOutput_y_indices), &out_indcie_shape);
    UpdataTensorShape(out_tensors_.at(kOutput_y_values), &out_values_shape);
    return RET_OK;
  }

  auto output_empty_row_indicator_ptr =
    reinterpret_cast<bool *>(out_tensors_[kOutput_empty_row_indicator]->MutableData());

  auto indices_ptr = reinterpret_cast<int32_t *>(in_tensors_[kInput_indices]->data());
  scratch_.clear();
  scratch_.resize(dense_rows_, 0);
  for (int32_t i = 0; i < N_; ++i) {
    const int32_t row = indices_ptr[i * rank_];
    if (row < 0 || row >= dense_rows_) {
      MS_LOG(ERROR) << "For '" << this->name_ << "', indices([" << i << "], 0) is invalid: [" << row << "] >= [ "
                    << dense_rows_ << "]";
      return RET_ERROR;
    }
    ++scratch_[row];
  }
  for (int32_t row = 0; row < dense_rows_; ++row) {
    // Scratch here describes the number of elements in this dense row
    output_empty_row_indicator_ptr[row] = (scratch_[row] == 0);
    // In filled version, each row has at least one element.
    scratch_[row] = std::max(scratch_[row], int32_t{1});
    if (row > 0) {
      scratch_[row] += scratch_[row - 1];
    }
  }
  out_indcie_shape.push_back(scratch_[dense_rows_ - 1]);
  out_indcie_shape.push_back(rank_);
  out_values_shape.push_back(scratch_[dense_rows_ - 1]);

  UpdataTensorShape(out_tensors_.at(kOutput_y_indices), &out_indcie_shape);
  UpdataTensorShape(out_tensors_.at(kOutput_y_values), &out_values_shape);
  return RET_OK;
}

template <typename T>
int SparseFillEmptyRowsCPUKernel::RunOutputData() {
  auto output_y_indices_ptr = reinterpret_cast<int32_t *>(out_tensors_[kOutput_y_indices]->MutableData());
  int32_t *output_reverse_index_map_ptr = nullptr;
  if (out_tensors_.size() == C4NUM) {
    output_reverse_index_map_ptr = reinterpret_cast<int32_t *>(out_tensors_[kOutput_reverse_index_map]->MutableData());
  }

  auto *values_ptr = reinterpret_cast<T *>(in_tensors_[kInput_values]->data());
  CHECK_NULL_RETURN(values_ptr);
  auto *default_value = reinterpret_cast<T *>(in_tensors_[kInput_default_value]->data());
  CHECK_NULL_RETURN(default_value);

  (void)std::memset(output_y_indices_ptr, 0, scratch_[dense_rows_ - 1] * rank_ * sizeof(int32_t));

  auto *output_y_values_ptr = reinterpret_cast<T *>(out_tensors_[kOutput_y_values]->MutableData());
  for (int32_t i = 0; i < scratch_[dense_rows_ - 1]; ++i) {
    output_y_values_ptr[i] = (*default_value);
  }

  auto indices_ptr = reinterpret_cast<int32_t *>(in_tensors_[kInput_indices]->data());
  std::vector<int32_t> filled_count(dense_rows_, 0);
  if (output_reverse_index_map_ptr == nullptr) {
    for (int32_t i = 0; i < N_; ++i) {
      const int32_t row = indices_ptr[i * rank_];
      int32_t &offset = filled_count[row];
      const int32_t output_i = ((row == 0) ? 0 : scratch_[row - 1]) + offset;
      offset++;  // Increment the filled count for this row.
      std::copy_n(indices_ptr + i * rank_, rank_, output_y_indices_ptr + output_i * rank_);
      output_y_values_ptr[output_i] = values_ptr[i];
    }
  } else {
    for (int32_t i = 0; i < N_; ++i) {
      const int32_t row = indices_ptr[i * rank_];
      int32_t &offset = filled_count[row];
      const int32_t output_i = ((row == 0) ? 0 : scratch_[row - 1]) + offset;
      offset++;  // Increment the filled count for this row.
      std::copy_n(indices_ptr + i * rank_, rank_, output_y_indices_ptr + output_i * rank_);
      output_y_values_ptr[output_i] = values_ptr[i];
      // We'll need this reverse index map to backprop correctly.
      output_reverse_index_map_ptr[i] = output_i;
    }
  }

  for (int32_t row = 0; row < dense_rows_; ++row) {
    const int32_t row_count = filled_count[row];
    if (row_count == 0) {  // We haven't filled this row
      const int32_t starting_index = (row == 0) ? 0 : scratch_[row - 1];
      // Remaining index values were set to zero already.
      // The value at this index was set to default_value already.
      // Just need to set the row index in the right location.
      output_y_indices_ptr[starting_index * rank_] = row;
    }
  }

  return RET_OK;
}

int SparseFillEmptyRowsCPUKernel::Run() {
  RunInferOutputShape();
  auto input_data_type = in_tensors_[kInput_values]->data_type();
  switch (input_data_type) {
    case kNumberTypeInt32:
      RunOutputData<int32_t>();
      break;
    case kNumberTypeFloat32:
      RunOutputData<float>();
      break;
    default:
      MS_LOG(ERROR) << "Unsupported data type: " << input_data_type << " of SparseFillEmptyRows cpu kernel.";
      return RET_ERROR;
  }

  for (auto *output : this->out_tensors()) {
    output->ResetRefCount();
  }
  return RET_OK;
}

REG_KERNEL(kCPU, kNumberTypeInt32, PrimitiveType_SparseFillEmptyRows, LiteKernelCreator<SparseFillEmptyRowsCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_SparseFillEmptyRows, LiteKernelCreator<SparseFillEmptyRowsCPUKernel>)
}  // namespace mindspore::kernel
