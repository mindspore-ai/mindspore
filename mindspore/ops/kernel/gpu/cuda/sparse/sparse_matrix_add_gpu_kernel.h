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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_SPARSE_MATRIX_ADD_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_SPARSE_MATRIX_ADD_GPU_KERNEL_H_

#include <cuda_runtime_api.h>
#include <cusparse.h>
#include <vector>
#include <string>
#include <memory>
#include <map>
#include <functional>
#include <algorithm>
#include "kernel/gpu/gpu_kernel.h"
#include "include/utils/utils.h"

namespace mindspore {
namespace kernel {
// The cusparse <https://docs.nvidia.com/cuda/cusparse/#csrgeam2> do not provide data generalization APIs.
// Here we use template specialization instead of switch-case statements.
template <typename T>
void CuSparseGetBufferSize(cusparseHandle_t handle, int m, int n, T alpha, cusparseMatDescr_t x1_descr, int x1_nnz,
                           T *x1_value, int *x1_row, int *x1_col, T beta, cusparseMatDescr_t x2_descr, int x2_nnz,
                           T *x2_value, int *x2_row, int *x2_col, cusparseMatDescr_t y_descr, T *y_value, int *y_row,
                           int *y_col, size_t *buffer_size);

template <typename T>
void CuSparseApplyGeam(cusparseHandle_t handle, int m, int n, T alpha, cusparseMatDescr_t x1_descr, int x1_nnz,
                       T *x1_value, int *x1_row, int *x1_col, T beta, cusparseMatDescr_t x2_descr, int x2_nnz,
                       T *x2_value, int *x2_row, int *x2_col, cusparseMatDescr_t y_descr, T *y_value, int *y_row,
                       int *y_col, void *buffer);

// Invoke different cusparse APIs with macro.
#define GEAM_GET_BUFFER_SIZE(abbr, ...) \
  CHECK_CUSPARSE_RET_WITH_EXCEPT(cusparse##abbr##csrgeam2_bufferSizeExt(__VA_ARGS__), "Get buffer size failed.");

#define GEAM_APPLY(abbr, ...) \
  CHECK_CUSPARSE_RET_WITH_EXCEPT(cusparse##abbr##csrgeam2(__VA_ARGS__), "apply geam failed.");

#define CUSPARSE_GEAM_GET_BUFFER_SIZE_SPECIALIZE(T, abbr)                                                              \
  template <>                                                                                                          \
  void CuSparseGetBufferSize<T>(cusparseHandle_t handle, int m, int n, T alpha, cusparseMatDescr_t x1_descr,           \
                                int x1_nnz, T *x1_value, int *x1_row, int *x1_col, T beta,                             \
                                cusparseMatDescr_t x2_descr, int x2_nnz, T *x2_value, int *x2_row, int *x2_col,        \
                                cusparseMatDescr_t y_descr, T *y_value, int *y_row, int *y_col, size_t *buffer_size) { \
    GEAM_GET_BUFFER_SIZE(abbr, handle, m, n, &alpha, x1_descr, x1_nnz, x1_value, x1_row, x1_col, &beta, x2_descr,      \
                         x2_nnz, x2_value, x2_row, x2_col, y_descr, y_value, y_row, y_col, buffer_size);               \
  }

#define CUSPARSE_GEAM_APPLY_SPECIALIZE(T, abbr)                                                                        \
  template <>                                                                                                          \
  void CuSparseApplyGeam<T>(cusparseHandle_t handle, int m, int n, T alpha, cusparseMatDescr_t x1_descr, int x1_nnz,   \
                            T *x1_value, int *x1_row, int *x1_col, T beta, cusparseMatDescr_t x2_descr, int x2_nnz,    \
                            T *x2_value, int *x2_row, int *x2_col, cusparseMatDescr_t y_descr, T *y_value, int *y_row, \
                            int *y_col, void *buffer) {                                                                \
    GEAM_APPLY(abbr, handle, m, n, &alpha, x1_descr, x1_nnz, x1_value, x1_row, x1_col, &beta, x2_descr, x2_nnz,        \
               x2_value, x2_row, x2_col, y_descr, y_value, y_row, y_col, buffer);                                      \
  }

#define CUSPARSE_GEAM_SPECIALIZE(T, abbr)           \
  CUSPARSE_GEAM_GET_BUFFER_SIZE_SPECIALIZE(T, abbr) \
  CUSPARSE_GEAM_APPLY_SPECIALIZE(T, abbr)

// Specialization
CUSPARSE_GEAM_SPECIALIZE(float, S)
CUSPARSE_GEAM_SPECIALIZE(double, D)
CUSPARSE_GEAM_SPECIALIZE(cuComplex, C)
CUSPARSE_GEAM_SPECIALIZE(cuDoubleComplex, Z)

// SparseMatrixAdd GPU kernel.
template <typename T>
class SparseMatrixAddGpuKernel : public NativeGpuKernelMod {
 public:
  SparseMatrixAddGpuKernel() { handle_ = device::gpu::GPUDeviceManager::GetInstance().GetCuSparseHandle(); }
  ~SparseMatrixAddGpuKernel() override = default;

  bool Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) {
    type_id_ = inputs.at(InputList::X1_VALUE)->dtype_id();
    CHECK_CUSPARSE_RET_WITH_EXCEPT(cusparseCreateMatDescr(&x1_descr_), "Create descriptor failed.");
    CHECK_CUSPARSE_RET_WITH_EXCEPT(cusparseCreateMatDescr(&x2_descr_), "Create descriptor failed.");
    CHECK_CUSPARSE_RET_WITH_EXCEPT(cusparseCreateMatDescr(&y_descr_), "Create descriptor failed.");
    CHECK_CUSPARSE_RET_WITH_EXCEPT(cusparseSetMatIndexBase(x1_descr_, CUSPARSE_INDEX_BASE_ZERO),
                                   "Set descriptor base index failed.");
    CHECK_CUSPARSE_RET_WITH_EXCEPT(cusparseSetMatIndexBase(x2_descr_, CUSPARSE_INDEX_BASE_ZERO),
                                   "Set descriptor base index failed.");
    CHECK_CUSPARSE_RET_WITH_EXCEPT(cusparseSetMatIndexBase(y_descr_, CUSPARSE_INDEX_BASE_ZERO),
                                   "Set descriptor base index failed.");
    return true;
  }

  int Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) {
    if (auto ret = KernelMod::Resize(inputs, outputs); ret != KRET_OK && ret != KRET_UNKNOWN_OUT_SHAPE) {
      return ret;
    }
    output_size_list_.clear();
    output_size_list_.emplace_back(inputs[InputList::X1_DENSE_SHAPE]->size());
    output_size_list_.emplace_back(inputs[InputList::X1_BATCH_POINTER]->size());
    output_size_list_.emplace_back(inputs[InputList::X1_ROW]->size());
    output_size_list_.emplace_back(inputs[InputList::X1_VALUE]->size() + inputs[InputList::X2_VALUE]->size());
    output_size_list_.emplace_back(inputs[InputList::X1_VALUE]->size() + inputs[InputList::X2_VALUE]->size());
    return KRET_OK;
  }

  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &,
              const std::vector<KernelTensor *> &outputs, void *stream_ptr) override {
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    ParseKernelParam(inputs, stream);

    auto x1_rows = GetDeviceAddress<int>(inputs, InputList::X1_ROW);
    auto x1_cols = GetDeviceAddress<int>(inputs, InputList::X1_COLUMN);
    auto x1_values = GetDeviceAddress<T>(inputs, InputList::X1_VALUE);

    auto x2_rows = GetDeviceAddress<int>(inputs, InputList::X2_ROW);
    auto x2_cols = GetDeviceAddress<int>(inputs, InputList::X2_COLUMN);
    auto x2_values = GetDeviceAddress<T>(inputs, InputList::X2_VALUE);

    auto y_dense_shape = GetDeviceAddress<int>(outputs, OutputList::DENSE_SHAPE);
    auto y_batch_pointer = GetDeviceAddress<int>(outputs, OutputList::BATCH_POINTER);
    auto y_rows = GetDeviceAddress<int>(outputs, OutputList::ROW);
    auto y_cols = GetDeviceAddress<int>(outputs, OutputList::COLUMN);
    auto y_values = GetDeviceAddress<T>(outputs, OutputList::VALUE);

    // The kernel should manage the buffer by itself as it depends on inputs value instead of shape.
    size_t buffer_size = GetBufferSize(x1_rows, x1_cols, x1_values, x2_rows, x2_cols, x2_values, stream);
    auto &allocator = device::gpu::GPUMemoryAllocator::GetInstance();
    void *buffer = allocator.AllocTensorMem(buffer_size);
    MS_EXCEPTION_IF_NULL(buffer);

    ApplyGeam(x1_rows, x1_cols, x1_values, x2_rows, x2_cols, x2_values, y_rows, y_cols, y_values, buffer, stream);
    allocator.FreeTensorMem(buffer);

    // Fill y_dense_shape and y_batch_pointer.
    CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
      cudaMemcpyAsync(y_dense_shape, inputs[InputList::X1_DENSE_SHAPE]->device_ptr(),
                      inputs[InputList::X1_DENSE_SHAPE]->size(), cudaMemcpyDeviceToDevice, stream),
      "cudaMemcpy failed.");

    CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
      cudaMemcpyAsync(y_batch_pointer, y_batch_pointer_host_.data(), y_batch_pointer_host_.size() * sizeof(int),
                      cudaMemcpyHostToDevice, stream),
      "cudaMemcpy failed.");

    return true;
  }

 protected:
  bool IsNeedUpdateOutputShapeAndSize() override { return true; }
  void UpdateOutputShapeAndSize(const std::vector<KernelTensor *> &inputs,
                                const std::vector<KernelTensor *> &outputs) override {
    std::vector<ShapeVector> shapes = {
      {SizeToLong(x1_dense_shape_host_.size())},                              // dense shape
      {SizeToLong(y_batch_pointer_host_.size())},                             // batch pointer
      {SizeToLong((row_ + 1) * batch_size_)},                                 // row
      {SizeToLong(y_batch_pointer_host_[y_batch_pointer_host_.size() - 1])},  // col
      {SizeToLong(y_batch_pointer_host_[y_batch_pointer_host_.size() - 1])},  // values
    };
    for (size_t i = 0; i < outputs.size(); ++i) {
      outputs[i]->SetShapeVector(shapes[i]);
      outputs[i]->set_size(LongToSize(std::accumulate(
        shapes[i].begin(), shapes[i].end(), UnitSizeInBytes(outputs[i]->dtype_id()), std::multiplies<int64_t>())));
    }
  }

 private:
  void ParseKernelParam(const std::vector<KernelTensor *> &inputs, cudaStream_t stream) {
    // Due to the design of primitive, dense_shape, batch_pointer, alpha, beta are on the device memory, while cusparse
    // requires it on the host. Additional memory copy between host and device will lead to significant performance
    // degradation. However, such an interface can only be compromised.
    auto x1_dense_shape = GetDeviceAddress<int>(inputs, InputList::X1_DENSE_SHAPE);
    x1_dense_shape_host_.resize(inputs[InputList::X1_DENSE_SHAPE]->size() / sizeof(int));
    CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
      cudaMemcpyAsync(x1_dense_shape_host_.data(), x1_dense_shape, x1_dense_shape_host_.size() * sizeof(int),
                      cudaMemcpyDeviceToHost, stream),
      "cudaMemcpy failed.");

    auto x1_batch_pointer = GetDeviceAddress<int>(inputs, InputList::X1_BATCH_POINTER);
    x1_batch_pointer_host_.resize(inputs[InputList::X1_BATCH_POINTER]->size() / sizeof(int));
    CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
      cudaMemcpyAsync(x1_batch_pointer_host_.data(), x1_batch_pointer, x1_batch_pointer_host_.size() * sizeof(int),
                      cudaMemcpyDeviceToHost, stream),
      "cudaMemcpy failed.");

    auto x2_dense_shape = GetDeviceAddress<int>(inputs, InputList::X2_DENSE_SHAPE);
    x2_dense_shape_host_.resize(inputs[InputList::X2_DENSE_SHAPE]->size() / sizeof(int));
    CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
      cudaMemcpyAsync(x2_dense_shape_host_.data(), x2_dense_shape, x2_dense_shape_host_.size() * sizeof(int),
                      cudaMemcpyDeviceToHost, stream),
      "cudaMemcpy failed.");

    auto x2_batch_pointer = GetDeviceAddress<int>(inputs, InputList::X2_BATCH_POINTER);
    x2_batch_pointer_host_.resize(inputs[InputList::X2_BATCH_POINTER]->size() / sizeof(int));
    CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
      cudaMemcpyAsync(x2_batch_pointer_host_.data(), x2_batch_pointer, x2_batch_pointer_host_.size() * sizeof(int),
                      cudaMemcpyDeviceToHost, stream),
      "cudaMemcpy failed.");

    auto alpha = GetDeviceAddress<T>(inputs, InputList::ALPHA);
    auto beta = GetDeviceAddress<T>(inputs, InputList::BETA);
    CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(cudaMemcpyAsync(&alpha_host_, alpha, sizeof(T), cudaMemcpyDeviceToHost, stream),
                                       "cudaMemcpy failed.");
    CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(cudaMemcpyAsync(&beta_host_, beta, sizeof(T), cudaMemcpyDeviceToHost, stream),
                                       "cudaMemcpy failed.");
    CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(cudaStreamSynchronize(stream), "cudaStreamSynchronize failed.");

    if (x1_dense_shape_host_ != x2_dense_shape_host_) {
      MS_LOG(EXCEPTION) << "The inputs dense shape should be same";
    }

    if (x1_batch_pointer_host_.size() != x2_batch_pointer_host_.size()) {
      MS_LOG(EXCEPTION) << "The batch pointer dim should be same. x1 batch dim: " << x1_batch_pointer_host_.size()
                        << "x2 batch dim: " << x2_batch_pointer_host_.size();
    }

    if (x1_batch_pointer_host_.size() == kIndex2) {
      batch_size_ = 1;
      row_ = x1_dense_shape_host_[kIndex0];
      col_ = x1_dense_shape_host_[kIndex1];
    } else {
      batch_size_ = x1_dense_shape_host_[kIndex0];
      row_ = x1_dense_shape_host_[kIndex1];
      col_ = x1_dense_shape_host_[kIndex2];
    }
  }

  size_t GetBufferSize(int *x1_rows, int *x1_cols, T *x1_values, int *x2_rows, int *x2_cols, T *x2_values,
                       cudaStream_t stream) {
    size_t max_buffer_size = 0;
    for (int i = 0; i < batch_size_; i++) {
      int x1_offset = x1_batch_pointer_host_[i];
      int x2_offset = x2_batch_pointer_host_[i];
      int row_offset = (row_ + 1) * i;

      int x1_nnz = x1_batch_pointer_host_[i + 1] - x1_offset;
      int x2_nnz = x2_batch_pointer_host_[i + 1] - x2_offset;

      int *x1_row = x1_rows + row_offset;
      int *x2_row = x2_rows + row_offset;
      int *x1_col = x1_cols + x1_offset;
      int *x2_col = x2_cols + x2_offset;
      T *x1_value = x1_values + x1_offset;
      T *x2_value = x2_values + x2_offset;

      size_t buffer_size;
      CuSparseGetBufferSize(handle_, row_, col_, alpha_host_, x1_descr_, x1_nnz, x1_value, x1_row, x1_col, beta_host_,
                            x2_descr_, x2_nnz, x2_value, x2_row, x2_col, y_descr_, static_cast<T *>(nullptr),
                            static_cast<int *>(nullptr), static_cast<int *>(nullptr), &buffer_size);
      max_buffer_size = std::max(max_buffer_size, buffer_size);
    }

    return max_buffer_size;
  }

  void ApplyGeam(int *x1_rows, int *x1_cols, T *x1_values, int *x2_rows, int *x2_cols, T *x2_values, int *y_rows,
                 int *y_cols, T *y_values, void *buffer, cudaStream_t stream) {
    y_batch_pointer_host_.clear();
    y_batch_pointer_host_.emplace_back(0);

    int total_nnz = 0;
    for (int i = 0; i < batch_size_; i++) {
      int x1_offset = x1_batch_pointer_host_[i];
      int x2_offset = x2_batch_pointer_host_[i];
      int row_offset = (row_ + 1) * i;

      int x1_nnz = x1_batch_pointer_host_[i + 1] - x1_offset;
      int x2_nnz = x2_batch_pointer_host_[i + 1] - x2_offset;

      int *x1_row = x1_rows + row_offset;
      int *x2_row = x2_rows + row_offset;
      int *x1_col = x1_cols + x1_offset;
      int *x2_col = x2_cols + x2_offset;
      T *x1_value = x1_values + x1_offset;
      T *x2_value = x2_values + x2_offset;

      int *y_row = y_rows + row_offset;
      int y_nnz;
      CHECK_CUSPARSE_RET_WITH_EXCEPT(
        cusparseXcsrgeam2Nnz(handle_, row_, col_, x1_descr_, x1_nnz, x1_row, x1_col, x2_descr_, x2_nnz, x2_row, x2_col,
                             y_descr_, y_row, &y_nnz, buffer),
        "Get y_nnz failed.");
      total_nnz += y_nnz;
      y_batch_pointer_host_.emplace_back(total_nnz);

      int *y_col = y_cols + y_batch_pointer_host_[i];
      T *y_value = y_values + y_batch_pointer_host_[i];
      CuSparseApplyGeam(handle_, row_, col_, alpha_host_, x1_descr_, x1_nnz, x1_value, x1_row, x1_col, beta_host_,
                        x2_descr_, x2_nnz, x2_value, x2_row, x2_col, y_descr_, y_value, y_row, y_col, buffer);
    }
  }

  TypeId type_id_{kTypeUnknown};
  cusparseHandle_t handle_{nullptr};
  cusparseMatDescr_t x1_descr_{nullptr};
  cusparseMatDescr_t x2_descr_{nullptr};
  cusparseMatDescr_t y_descr_{nullptr};

  int batch_size_{1};
  int row_{0};
  int col_{0};

  std::vector<int> x1_dense_shape_host_;
  std::vector<int> x2_dense_shape_host_;
  std::vector<int> x1_batch_pointer_host_;
  std::vector<int> x2_batch_pointer_host_;
  std::vector<int> y_batch_pointer_host_;

  T alpha_host_;
  T beta_host_;

  enum InputList {
    X1_DENSE_SHAPE,
    X1_BATCH_POINTER,
    X1_ROW,
    X1_COLUMN,
    X1_VALUE,
    X2_DENSE_SHAPE,
    X2_BATCH_POINTER,
    X2_ROW,
    X2_COLUMN,
    X2_VALUE,
    ALPHA,
    BETA
  };
  enum OutputList { DENSE_SHAPE, BATCH_POINTER, ROW, COLUMN, VALUE };
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_SPARSE_MATRIX_ADD_GPU_KERNEL_H_
