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
#include "plugin/device/gpu/kernel/gpu_kernel.h"

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
class SparseMatrixAddGpuKernel : public DeprecatedNativeGpuKernelMod {
 public:
  SparseMatrixAddGpuKernel() { handle_ = device::gpu::GPUDeviceManager::GetInstance().GetCuSparseHandle(); }
  ~SparseMatrixAddGpuKernel() override = default;

  bool Init(const CNodePtr &kernel_node) override {
    kernel_node_ = kernel_node;
    auto dense_shape = common::AnfAlgo::GetNodeAttr<std::vector<int64_t>>(kernel_node, "x1_dense_shape");
    RETURN_IF_FALSE_WITH_LOG(dense_shape.size() != kMatrixDims, "The rank of dense_shape should be 2.");
    row_ = LongToInt(dense_shape[0]);
    col_ = LongToInt(dense_shape[1]);

    const auto &x1_col_shape = common::AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, InputList::X1_COLUMN);
    RETURN_IF_FALSE_WITH_LOG(x1_col_shape.size() != 1, "The rank of column should be 1.");
    x1_nnz_ = SizeToInt(x1_col_shape[0]);

    const auto &x2_col_shape = common::AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, InputList::X2_COLUMN);
    RETURN_IF_FALSE_WITH_LOG(x2_col_shape.size() != 1, "The rank of row should be 1.");
    x2_nnz_ = SizeToInt(x2_col_shape[0]);

    type_id_ = common::AnfAlgo::GetPrevNodeOutputInferDataType(kernel_node, InputList::X1_VALUE);

    CHECK_CUSPARSE_RET_WITH_EXCEPT(cusparseCreateMatDescr(&x1_descr_), "Create descriptor failed.");
    CHECK_CUSPARSE_RET_WITH_EXCEPT(cusparseCreateMatDescr(&x2_descr_), "Create descriptor failed.");
    CHECK_CUSPARSE_RET_WITH_EXCEPT(cusparseCreateMatDescr(&y_descr_), "Create descriptor failed.");
    CHECK_CUSPARSE_RET_WITH_EXCEPT(cusparseSetMatIndexBase(x1_descr_, CUSPARSE_INDEX_BASE_ZERO),
                                   "Set descriptor base index failed.");
    CHECK_CUSPARSE_RET_WITH_EXCEPT(cusparseSetMatIndexBase(x2_descr_, CUSPARSE_INDEX_BASE_ZERO),
                                   "Set descriptor base index failed.");
    CHECK_CUSPARSE_RET_WITH_EXCEPT(cusparseSetMatIndexBase(y_descr_, CUSPARSE_INDEX_BASE_ZERO),
                                   "Set descriptor base index failed.");

    InitSizeLists();
    is_need_updateop_ = true;
    return true;
  }

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    auto x1_row = GetDeviceAddress<int>(inputs, InputList::X1_ROW);
    auto x1_col = GetDeviceAddress<int>(inputs, InputList::X1_COLUMN);
    auto x1_value = GetDeviceAddress<T>(inputs, InputList::X1_VALUE);

    auto x2_row = GetDeviceAddress<int>(inputs, InputList::X2_ROW);
    auto x2_col = GetDeviceAddress<int>(inputs, InputList::X2_COLUMN);
    auto x2_value = GetDeviceAddress<T>(inputs, InputList::X2_VALUE);

    auto alpha = GetDeviceAddress<T>(inputs, InputList::ALPHA);
    auto beta = GetDeviceAddress<T>(inputs, InputList::BETA);

    auto y_row = GetDeviceAddress<int>(outputs, OutputList::ROW);
    auto y_col = GetDeviceAddress<int>(outputs, OutputList::COLUMN);
    auto y_value = GetDeviceAddress<T>(outputs, OutputList::VALUE);

    T alpha_host, beta_host;
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(cudaMemcpyAsync(&alpha_host, alpha, sizeof(T), cudaMemcpyHostToDevice, stream),
                                       "cudaMemcpy failed.");
    CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(cudaMemcpyAsync(&beta_host, beta, sizeof(T), cudaMemcpyHostToDevice, stream),
                                       "cudaMemcpy failed.");
    CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(cudaStreamSynchronize(stream), "cudaStreamSynchronize failed.");

    size_t buffer_size = 0;
    CuSparseGetBufferSize(handle_, row_, col_, alpha_host, x1_descr_, x1_nnz_, x1_value, x1_row, x1_col, beta_host,
                          x2_descr_, x2_nnz_, x2_value, x2_row, x2_col, y_descr_, y_value, y_row, y_col, &buffer_size);

    // The kernel should manage the buffer by itself as it depends on inputs value instead of shape.
    auto &allocator = device::gpu::GPUMemoryAllocator::GetInstance();
    void *buffer = allocator.AllocTensorMem(buffer_size);
    MS_EXCEPTION_IF_NULL(buffer);

    CHECK_CUSPARSE_RET_WITH_EXCEPT(
      cusparseXcsrgeam2Nnz(handle_, row_, col_, x1_descr_, x1_nnz_, x1_row, x1_col, x2_descr_, x2_nnz_, x2_row, x2_col,
                           y_descr_, y_row, &y_nnz_, buffer),
      "Get y_nnz failed.");

    CuSparseApplyGeam(handle_, row_, col_, alpha_host, x1_descr_, x1_nnz_, x1_value, x1_row, x1_col, beta_host,
                      x2_descr_, x2_nnz_, x2_value, x2_row, x2_col, y_descr_, y_value, y_row, y_col, buffer);

    allocator.FreeTensorMem(buffer);
    return true;
  }

  void UpdateOp() {
    std::vector<TypeId> types;
    types.push_back(kNumberTypeInt32);
    types.push_back(kNumberTypeInt32);
    types.push_back(type_id_);

    std::vector<std::vector<size_t>> shapes;
    shapes.push_back({
      IntToSize(row_ + 1),
    });
    shapes.push_back({
      IntToSize(y_nnz_),
    });
    shapes.push_back({
      IntToSize(y_nnz_),
    });

    common::AnfAlgo::SetOutputInferTypeAndShape(types, shapes, kernel_node_.lock().get());
  }

 protected:
  void InitSizeLists() override {
    // x1
    input_size_list_.push_back((row_ + 1) * sizeof(int));
    input_size_list_.push_back(x1_nnz_ * sizeof(int));
    input_size_list_.push_back(x1_nnz_ * sizeof(T));

    // x2
    input_size_list_.push_back((row_ + 1) * sizeof(int));
    input_size_list_.push_back(x2_nnz_ * sizeof(int));
    input_size_list_.push_back(x2_nnz_ * sizeof(T));

    // alpha, beta
    input_size_list_.push_back(sizeof(T));
    input_size_list_.push_back(sizeof(T));

    // Suppose all of the input elements are different.
    // The exact y_nnz will calculated after kernel launched.
    output_size_list_.push_back((x1_nnz_ + x2_nnz_ + 1) * sizeof(int));
    output_size_list_.push_back((x1_nnz_ + x2_nnz_) * sizeof(int));
    output_size_list_.push_back((x1_nnz_ + x2_nnz_) * sizeof(T));
  }

 private:
  std::weak_ptr<CNode> kernel_node_;
  TypeId type_id_{kTypeUnknown};
  cusparseHandle_t handle_{nullptr};
  cusparseMatDescr_t x1_descr_{nullptr};
  cusparseMatDescr_t x2_descr_{nullptr};
  cusparseMatDescr_t y_descr_{nullptr};
  int row_{0};
  int col_{0};
  int x1_nnz_{0};
  int x2_nnz_{0};
  int y_nnz_{0};

  static constexpr size_t kMatrixDims = 2;
  enum InputList { X1_ROW, X1_COLUMN, X1_VALUE, X2_ROW, X2_COLUMN, X2_VALUE, ALPHA, BETA };
  enum OutputList { ROW, COLUMN, VALUE };
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_SPARSE_MATRIX_ADD_GPU_KERNEL_H_
