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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_EINSUM_HELPER_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_EINSUM_HELPER_H_

#include <ctype.h>
#include <vector>
#include <string>
#include <unordered_map>
#include <map>
#include <tuple>
#include <set>
#include <memory>
#include <functional>
#include "plugin/device/gpu/kernel/gpu_kernel.h"
#include "plugin/device/gpu/kernel/gpu_kernel_factory.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/transpose_impl.cuh"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/einsum_impl.cuh"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/tile_impl.cuh"
#include "plugin/device/gpu/kernel/kernel_constants.h"

namespace mindspore {
namespace kernel {
constexpr size_t LABEL_NUM = 52;
constexpr size_t ELL_VAL = 52;
constexpr int LOWER_CASE_BEGIN = 26;
constexpr char ELLIPSIS = '.';
constexpr int SPLIT_DIM = 4;
constexpr int SHAPE_WORKSPACE_NUM = 3;
constexpr int DIM_TWO = 2;
constexpr int DIM_THREE = 3;
constexpr int DIM_FOUR = 4;
constexpr int HALF_TYPE_WORK_SIZE_MUL = 2;
constexpr int ELL_LEN = 3;
// tuple<>:0:操作信息{diagonal, transpose, reduce_sum}, 1:input_shape, 2:operate_param, 3: out_shape
using OpStruct = std::tuple<std::string, std::vector<int64_t>, std::vector<size_t>, std::vector<int64_t>>;
template <typename T>
class EinsumHelper {
 public:
  EinsumHelper() {
    label_perm_idx_ = std::vector<int64_t>(LABEL_NUM, -1);
    element_count_ = std::vector<size_t>(LABEL_NUM, 0);
    ell_idx_ = 0;
    ell_dim_ = 0;
    perm_idx_ = 0;
    out_size_ = 0;
    shape_ptr_idx_start_ = 0;
    workspace_ptr_.clear();
    left_elements_.clear();
    element_shape_map_.clear();
  }

  int64_t char_to_index(const char cur_char) {
    if (cur_char <= 'z' && cur_char >= 'a') {
      return static_cast<int64_t>(cur_char - 'a' + LOWER_CASE_BEGIN);
    }
    return static_cast<int64_t>(cur_char - 'A');
  }
  size_t GetShapeSize(const std::vector<int64_t> &shape) {
    size_t size = sizeof(T);
    for (auto &dim : shape) {
      size *= static_cast<size_t>(dim);
    }
    return size;
  }
  void Init(const std::vector<void *> workspace_list, const TypeId &cur_type) {
    for (auto ptr : workspace_list) {
      workspace_ptr_.emplace_back(ptr);
    }
    shape_ptr_idx_start_ = workspace_ptr_.size() - SHAPE_WORKSPACE_NUM;
    data_type_ = cur_type;
  }
  bool Preprocess(const std::string &orig_equatioin, const std::string &node_name,
                  const std::vector<std::vector<int64_t>> &input_shapes, std::vector<int64_t> *out_shape,
                  std::vector<std::vector<OpStruct>> *single_op, std::vector<OpStruct> *res_op) {
    std::string equation = orig_equatioin;
    node_name_ = node_name;
    equation.erase(std::remove(equation.begin(), equation.end(), ' '), equation.end());
    left_elements_ = std::vector<std::vector<size_t>>(input_shapes.size());
    if (equation == "") {
      MS_LOG(ERROR) << "For " << node_name_ << ", euqation is required, but got none.";
      return false;
    }
    bool flag = CalOutShape(equation, input_shapes, out_shape);
    if (!flag) {
      return false;
    }
    StatCalProcess(input_shapes, single_op, res_op);
    return true;
  }

  void SingleElementProcess(const std::string &func_name, T *input_ptr, T *output_ptr,
                            const std::vector<size_t> &inp_shape, const std::vector<size_t> &operate_info,
                            void *stream_ptr) {
    if (func_name == "Diagonal") {
      Diagonal(input_ptr, output_ptr, inp_shape, operate_info, stream_ptr);
    } else if (func_name == "ReduceSum") {
      ReduceSumCuda(input_ptr, output_ptr, inp_shape, operate_info, stream_ptr);
    } else {
      Transpose(input_ptr, output_ptr, inp_shape, operate_info, stream_ptr);
    }
  }
  void TwoElementProcess(const std::string &func_name, T *lft_input, T *rht_input, T *output,
                         const std::vector<size_t> &lft_shape, const std::vector<size_t> &rht_shape,
                         const std::vector<size_t> &out_shape, void *stream_ptr) {
    if (func_name == "Mul") {
      Mul(lft_input, rht_input, output, lft_shape, rht_shape, out_shape, stream_ptr);
    } else if (func_name == "Dot") {
      Dot(lft_input, rht_input, output, lft_shape, rht_shape, out_shape, stream_ptr);
    } else {
      transpose_x1_ = CUBLAS_OP_N;
      transpose_x2_ = CUBLAS_OP_N;
      Bmm(lft_input, rht_input, output, lft_shape, rht_shape, out_shape, stream_ptr);
    }
  }
  void SingleElementProcessGrad(const std::string &func_name, T *dout_ptr, T *dinp_ptr,
                                const std::vector<size_t> &dinp_shape, const std::vector<size_t> &operate_info,
                                void *stream_ptr) {
    if (func_name == "Diagonal") {
      DiagonalGrad(dout_ptr, dinp_ptr, dinp_shape, operate_info, stream_ptr);
    } else if (func_name == "ReduceSum") {
      ReduceSumCudaGrad(dout_ptr, dinp_ptr, dinp_shape, operate_info, stream_ptr);
    } else {
      std::vector<size_t> dout_shape(dinp_shape.size());
      std::vector<size_t> doperate_info(dinp_shape.size());
      for (size_t idx = 0; idx < dinp_shape.size(); ++idx) {
        dout_shape[idx] = dinp_shape[operate_info[idx]];
        doperate_info[operate_info[idx]] = idx;
      }
      Transpose(dout_ptr, dinp_ptr, dout_shape, doperate_info, stream_ptr);
    }
  }
  void TwoElementProcessGrad(const std::string &func_name, T *dout, T *mid_res, T *drht, T *dlft,
                             const std::vector<size_t> &dlft_shape, const std::vector<size_t> &drht_shape,
                             const std::vector<size_t> &dout_shape, void *stream_ptr) {
    if (func_name == "Mul") {
      MulGrad(dout, mid_res, drht, dlft, dlft_shape, drht_shape, dout_shape, stream_ptr);
    } else if (func_name == "Dot") {
      DotGrad(dout, mid_res, drht, dlft, dout_shape, dlft_shape, drht_shape, stream_ptr);
    } else {
      transpose_x1_ = CUBLAS_OP_N;
      transpose_x2_ = CUBLAS_OP_T;
      Bmm(dout, drht, dlft, dout_shape, drht_shape, dlft_shape, stream_ptr);
      transpose_x1_ = CUBLAS_OP_T;
      transpose_x2_ = CUBLAS_OP_N;
      Bmm(mid_res, dout, drht, dlft_shape, dout_shape, drht_shape, stream_ptr);
    }
  }

 protected:
  void Mul(T *lft_input, T *rht_input, T *output, const std::vector<size_t> &lft_shape,
           const std::vector<size_t> &rht_shape, const std::vector<size_t> &out_shape, void *stream_ptr) {
    size_t lft_num = 1;
    size_t rht_num = 1;
    size_t out_num = 1;
    bool need_broadcast = false;
    size_t *lft_shape_ptr = reinterpret_cast<size_t *>(workspace_ptr_[shape_ptr_idx_start_]);
    size_t *rht_shape_ptr = reinterpret_cast<size_t *>(workspace_ptr_[shape_ptr_idx_start_ + 1]);
    size_t *out_shape_ptr = reinterpret_cast<size_t *>(workspace_ptr_[shape_ptr_idx_start_ + 2]);
    for (size_t i = 0; i < lft_shape.size(); i++) {
      if (lft_shape[i] != rht_shape[i]) {
        need_broadcast = true;
      }
      lft_num *= lft_shape[i];
      rht_num *= rht_shape[i];
      out_num *= out_shape[i];
    }
    CHECK_CUDA_RET_WITH_ERROR_NOTRACE(
      cudaMemcpyAsync(lft_shape_ptr, &lft_shape[0], lft_shape.size() * sizeof(size_t), cudaMemcpyHostToDevice,
                      reinterpret_cast<cudaStream_t>(stream_ptr)),
      "For " + node_name_ + ", Mul's cudaMemcpyAsync failed.");
    CHECK_CUDA_RET_WITH_ERROR_NOTRACE(
      cudaMemcpyAsync(rht_shape_ptr, &rht_shape[0], rht_shape.size() * sizeof(size_t), cudaMemcpyHostToDevice,
                      reinterpret_cast<cudaStream_t>(stream_ptr)),
      "For " + node_name_ + ", Mul's cudaMemcpyAsync failed.");
    CHECK_CUDA_RET_WITH_ERROR_NOTRACE(
      cudaMemcpyAsync(out_shape_ptr, &out_shape[0], out_shape.size() * sizeof(size_t), cudaMemcpyHostToDevice,
                      reinterpret_cast<cudaStream_t>(stream_ptr)),
      "For " + node_name_ + ", Mul's cudaMemcpyAsync failed.");
    CalMul<T>(need_broadcast, out_shape.size(), lft_shape_ptr, lft_num, rht_shape_ptr, rht_num, out_shape_ptr, out_num,
              lft_input, rht_input, output, reinterpret_cast<cudaStream_t>(stream_ptr));
  }
  void Bmm(T *input_a, T *input_b, T *output, const std::vector<size_t> &a_shape, const std::vector<size_t> &b_shape,
           const std::vector<size_t> &out_shape, void *stream_ptr) {
    cublasHandle_t handle = device::gpu::GPUDeviceManager::GetInstance().GetCublasHandle();
    CHECK_CUBLAS_RET_WITH_ERROR(cublasSetStream(handle, reinterpret_cast<cudaStream_t>(stream_ptr)),
                                "cublasSetStream failed in primitive[Einsum].");
    cublasGemmAlgo_t algo = CUBLAS_GEMM_DEFAULT;
    size_t b = out_shape[out_shape.size() - DIM_THREE];
    size_t m = out_shape[out_shape.size() - DIM_TWO];
    size_t n = out_shape[out_shape.size() - 1];
    size_t k;
    if (transpose_x1_ == CUBLAS_OP_T) {
      k = a_shape[a_shape.size() - DIM_TWO];
    } else {
      k = a_shape[a_shape.size() - 1];
    }
    const int lda = (transpose_x1_ == CUBLAS_OP_T) ? SizeToInt(m) : SizeToInt(k);
    const int ldb = (transpose_x2_ == CUBLAS_OP_T) ? SizeToInt(k) : SizeToInt(n);

    auto stride_a = SizeToInt(m * k);
    auto stride_b = SizeToInt(k * n);
    auto stride_c = SizeToInt(m * n);
    cudaDataType_t cu_type = kCudaDtypeMap[TypeIdLabel(data_type_)];
    cudaDataType_t comp_type = (cu_type == CUDA_R_64F) ? CUDA_R_64F : CUDA_R_32F;
    if (cu_type == CUDA_R_16F) {
      algo = CUBLAS_GEMM_DEFAULT_TENSOR_OP;
      float alpha = 1.0f;
      float beta = 0.0f;
      CHECK_CUBLAS_RET_WITH_ERROR(
        cublasGemmStridedBatchedEx(handle, transpose_x2_, transpose_x1_, SizeToInt(n), SizeToInt(m), SizeToInt(k),
                                   &alpha, input_b, cu_type, ldb, stride_b, input_a, cu_type, lda, stride_a, &beta,
                                   output, cu_type, n, stride_c, b, comp_type, algo),
        "For " + node_name_ + ", cublasGemmStridedBatchedEx failed.");
    } else {
      T alpha = static_cast<T>(1.0f);
      T beta = static_cast<T>(0.0f);
      CHECK_CUBLAS_RET_WITH_ERROR(
        cublasGemmStridedBatchedEx(handle, transpose_x2_, transpose_x1_, SizeToInt(n), SizeToInt(m), SizeToInt(k),
                                   &alpha, input_b, cu_type, ldb, stride_b, input_a, cu_type, lda, stride_a, &beta,
                                   output, cu_type, n, stride_c, b, comp_type, algo),
        "For " + node_name_ + ", cublasGemmStridedBatchedEx failed.");
    }
    handle = nullptr;
  }
  void Dot(T *input_a, T *input_b, T *output, const std::vector<size_t> &a_shape, const std::vector<size_t> &b_shape,
           const std::vector<size_t> &out_shape, void *stream_ptr) {
    size_t size = 1;
    for (size_t idx = 0; idx < a_shape.size(); ++idx) {
      size *= a_shape[idx];
    }
    CHECK_CUDA_RET_WITH_ERROR_NOTRACE(cudaMemset(output, 0, sizeof(T) * 1),
                                      "Dot's cudaMemset failed in primitive[Einsum].");
    CalDot<T>(size, input_a, input_b, output, reinterpret_cast<cudaStream_t>(stream_ptr));
  }
  void DotGrad(T *output, T *mid_res, T *input_b, T *input_a, const std::vector<size_t> &out_shape,
               const std::vector<size_t> &a_shape, const std::vector<size_t> &b_shape, void *stream_ptr) {
    size_t size = 1;
    for (size_t idx = 0; idx < a_shape.size(); ++idx) {
      size *= a_shape[idx];
    }
    T out_val;
    CHECK_CUDA_RET_WITH_ERROR_NOTRACE(
      cudaMemcpyAsync(&out_val, output, sizeof(T), cudaMemcpyDeviceToHost, reinterpret_cast<cudaStream_t>(stream_ptr)),
      "For " + node_name_ + ", Diagonal's cudaMemcpyAsync failed.");
    CalDotGrad<T>(size, out_val, mid_res, input_b, input_a, reinterpret_cast<cudaStream_t>(stream_ptr));
  }
  void Diagonal(T *input_ptr, T *output_ptr, const std::vector<size_t> &inp_shape,
                const std::vector<size_t> &operate_info, void *stream_ptr) {
    // operate_info[0]: dim1
    size_t out_size = 1;
    size_t dim1 = operate_info[0];
    size_t dim2 = operate_info[1];
    size_t *shape_ptr = reinterpret_cast<size_t *>(workspace_ptr_[shape_ptr_idx_start_]);
    for (size_t idx = 0; idx < inp_shape.size(); ++idx) {
      out_size *= inp_shape[idx];
    }
    out_size /= inp_shape[dim2];
    CHECK_CUDA_RET_WITH_ERROR_NOTRACE(
      cudaMemcpyAsync(shape_ptr, &inp_shape[0], inp_shape.size() * sizeof(size_t), cudaMemcpyHostToDevice,
                      reinterpret_cast<cudaStream_t>(stream_ptr)),
      "For " + node_name_ + ", Diagonal's cudaMemcpyAsync failed.");
    CalDiagonal<T>(out_size, input_ptr, shape_ptr, inp_shape.size(), dim1, dim2, output_ptr,
                   reinterpret_cast<cudaStream_t>(stream_ptr));
  }
  void DiagonalGrad(T *dout_ptr, T *dinp_ptr, const std::vector<size_t> &dinp_shape,
                    const std::vector<size_t> &operate_info, void *stream_ptr) {
    size_t dinp_size = 1;
    size_t dout_size = 1;
    size_t dim1 = operate_info[0];
    size_t dim2 = operate_info[1];
    size_t *d_shape_ptr = reinterpret_cast<size_t *>(workspace_ptr_[shape_ptr_idx_start_]);
    for (size_t idx = 0; idx < dinp_shape.size(); ++idx) {
      dinp_size *= dinp_shape[idx];
    }
    dout_size = dinp_size / dinp_shape[dim2];
    CHECK_CUDA_RET_WITH_ERROR_NOTRACE(
      cudaMemcpyAsync(d_shape_ptr, &dinp_shape[0], dinp_shape.size() * sizeof(size_t), cudaMemcpyHostToDevice,
                      reinterpret_cast<cudaStream_t>(stream_ptr)),
      "For " + node_name_ + ", Diagonal's cudaMemcpyAsync failed.");
    CHECK_CUDA_RET_WITH_ERROR_NOTRACE(
      cudaMemsetAsync(dinp_ptr, 0, dinp_size * sizeof(T), reinterpret_cast<cudaStream_t>(stream_ptr)),
      "For " + node_name_ + ", Diagonal's cudaMemsetAsync failed.");

    CalDiagonalGrad<T>(dout_size, dout_ptr, d_shape_ptr, dinp_shape.size(), dim1, dim2, dinp_ptr,
                       reinterpret_cast<cudaStream_t>(stream_ptr));
  }
  void CudnnSetTensorNdDescriptor(const std::vector<size_t> &shape, cudnnTensorDescriptor_t descriptor,
                                  cudnnDataType_t data_type) {
    if (shape.size() < DIM_THREE) {
      MS_EXCEPTION(ValueError) << "For " << node_name_ << ", cudnnSetTensorNdDescriptor support 3 dim, but got "
                               << shape.size() << "D.";
    }
    const int nbDims = shape.size();
    std::unique_ptr<int[]> dim = std::make_unique<int[]>(nbDims);
    std::unique_ptr<int[]> stride = std::make_unique<int[]>(nbDims);

    for (int i = 0; i < nbDims; i++) {
      dim[i] = SizeToInt(shape[i]);
      stride[i] = 1;
    }

    for (int i = nbDims - DIM_TWO; i >= 0; i--) {
      stride[i] = stride[i + 1] * SizeToInt(shape[i + 1]);
    }

    CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(
      cudnnSetTensorNdDescriptor(descriptor, data_type, nbDims, dim.get(), stride.get()),
      "For " + node_name_ + ", cudnnSetTensorNdDescriptor failed.");
  }
  // expand Nd Shape to 4d (N in [0,4])
  void ShapeNdTo4d(const std::vector<size_t> &src, std::vector<size_t> *dst) {
    dst->push_back(src.size() < DIM_FOUR ? 1 : src[src.size() - DIM_FOUR]);
    dst->push_back(src.size() < DIM_THREE ? 1 : src[src.size() - DIM_THREE]);
    dst->push_back(src.size() < DIM_TWO ? 1 : src[src.size() - DIM_TWO]);
    dst->push_back(src.size() == 0 ? 1 : src[src.size() - 1]);
  }
  void ReduceSumCuda(T *input_ptr, T *output_ptr, const std::vector<size_t> &inp_shape,
                     const std::vector<size_t> &operate_info, void *stream_ptr) {
    cudnnHandle_t cudnn_handle = device::gpu::GPUDeviceManager::GetInstance().GetCudnnHandle();
    cudnnReduceTensorDescriptor_t reduce_tensor_descriptor;
    cudnnTensorDescriptor_t inputA_descriptor;
    cudnnTensorDescriptor_t outputC_descriptor;
    CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(cudnnCreateReduceTensorDescriptor(&reduce_tensor_descriptor),
                                        "For " + node_name_ + ", cudnnCreateReduceTensorDescriptor failed.");
    CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(cudnnCreateTensorDescriptor(&inputA_descriptor),
                                        "For " + node_name_ + ", cudnnCreateTensorDescriptor failed.");
    CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(cudnnCreateTensorDescriptor(&outputC_descriptor),
                                        "For " + node_name_ + ", cudnnCreateTensorDescriptor failed.");
    std::vector<size_t> out_shape(inp_shape);
    bool copy_flag = true;
    for (auto &axis : operate_info) {
      out_shape[axis] = 1;
      if (inp_shape[axis] != 1) {
        copy_flag = false;
      }
    }
    std::vector<size_t> shrink_inp_shape;
    std::vector<size_t> shrink_out_shape;
    size_t copy_size = sizeof(T);
    for (size_t idx = 0; idx < inp_shape.size(); ++idx) {
      if (inp_shape[idx] != 1) {
        copy_size *= inp_shape[idx];
        shrink_inp_shape.emplace_back(inp_shape[idx]);
        shrink_out_shape.emplace_back(out_shape[idx]);
      }
    }
    if (copy_flag) {
      CHECK_CUDA_RET_WITH_ERROR_NOTRACE(cudaMemcpyAsync(output_ptr, input_ptr, copy_size, cudaMemcpyDeviceToDevice,
                                                        reinterpret_cast<cudaStream_t>(stream_ptr)),
                                        "For " + node_name_ + ", ReduceSum's cudaMemcpyAsync failed.");
      CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(cudnnDestroyReduceTensorDescriptor(reduce_tensor_descriptor),
                                          "For " + node_name_ + ", cudnnDestroyReduceTensorDescriptor failed.");
      CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(cudnnDestroyTensorDescriptor(inputA_descriptor),
                                          "For " + node_name_ + ", cudnnDestroyTensorDescriptor failed.");
      CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(cudnnDestroyTensorDescriptor(outputC_descriptor),
                                          "For " + node_name_ + ", cudnnDestroyTensorDescriptor failed.");
      return;
    }
    cudnnDataType_t data_type = kCudnnDtypeMap[TypeIdLabel(data_type_)];
    std::vector<size_t> inputA;
    if (shrink_inp_shape.size() <= SPLIT_DIM) {
      ShapeNdTo4d(shrink_inp_shape, &inputA);
      CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(
        cudnnSetTensor4dDescriptor(inputA_descriptor, CUDNN_TENSOR_NCHW, data_type, SizeToInt(inputA[0]),
                                   SizeToInt(inputA[1]), SizeToInt(inputA[DIM_TWO]), SizeToInt(inputA[DIM_THREE])),
        "For " + node_name_ + ", cudnnSetTensor4dDescriptor failed.");
    } else {
      CudnnSetTensorNdDescriptor(shrink_inp_shape, inputA_descriptor, data_type);
      for (auto dim : shrink_inp_shape) {
        inputA.emplace_back(SizeToInt(dim));
      }
    }
    std::vector<size_t> outputC;
    if (shrink_out_shape.size() <= SPLIT_DIM) {
      ShapeNdTo4d(shrink_out_shape, &outputC);
      CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(
        cudnnSetTensor4dDescriptor(outputC_descriptor, CUDNN_TENSOR_NCHW, data_type, SizeToInt(outputC[0]),
                                   SizeToInt(outputC[1]), SizeToInt(outputC[DIM_TWO]), SizeToInt(outputC[DIM_THREE])),
        "For " + node_name_ + ", cudnnSetTensor4dDescriptor failed.");
    } else {
      CudnnSetTensorNdDescriptor(shrink_out_shape, outputC_descriptor, data_type);
      for (auto dim : shrink_out_shape) {
        outputC.emplace_back(SizeToInt(dim));
      }
    }
    T *workspace_addr = reinterpret_cast<T *>(workspace_ptr_[shape_ptr_idx_start_ - 1]);
    cudnnDataType_t comp_type = (data_type == CUDNN_DATA_DOUBLE) ? CUDNN_DATA_DOUBLE : CUDNN_DATA_FLOAT;
    CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(
      cudnnSetReduceTensorDescriptor(reduce_tensor_descriptor, CUDNN_REDUCE_TENSOR_ADD, comp_type,
                                     CUDNN_NOT_PROPAGATE_NAN, CUDNN_REDUCE_TENSOR_NO_INDICES, CUDNN_32BIT_INDICES),
      "For " + node_name_ + ", cudnnSetReduceTensorDescriptor failed.");
    T alpha = static_cast<T>(1.0f);
    T beta = static_cast<T>(0.0f);
    if (data_type == CUDNN_DATA_DOUBLE) {
      CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(
        cudnnReduceTensor(cudnn_handle, reduce_tensor_descriptor, nullptr, 0, workspace_addr, copy_size, &alpha,
                          inputA_descriptor, input_ptr, &beta, outputC_descriptor, output_ptr),
        "For " + node_name_ + ", cudnnReduceTensor failed.");
    } else {
      const float alphaf = static_cast<float>(alpha);
      const float betaf = static_cast<float>(beta);
      if (data_type == CUDNN_DATA_HALF) {
        copy_size *= HALF_TYPE_WORK_SIZE_MUL;
      }
      CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(
        cudnnReduceTensor(cudnn_handle, reduce_tensor_descriptor, nullptr, 0, workspace_addr, copy_size, &alphaf,
                          inputA_descriptor, input_ptr, &betaf, outputC_descriptor, output_ptr),
        "For " + node_name_ + ", cudnnReduceTensor failed.");
    }
    CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(cudnnDestroyReduceTensorDescriptor(reduce_tensor_descriptor),
                                        "For " + node_name_ + ", cudnnDestroyReduceTensorDescriptor failed.");
    CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(cudnnDestroyTensorDescriptor(inputA_descriptor),
                                        "For " + node_name_ + ", cudnnDestroyTensorDescriptor failed.");
    CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(cudnnDestroyTensorDescriptor(outputC_descriptor),
                                        "For " + node_name_ + ", cudnnDestroyTensorDescriptor failed.");
  }
  void ReduceSumCudaGrad(T *dout, T *d_input, const std::vector<size_t> &dinp_shape,
                         const std::vector<size_t> &operate_info, void *stream_ptr) {
    std::vector<size_t> dout_shape(dinp_shape);
    for (auto &axis : operate_info) {
      dout_shape[axis] = 1;
    }
    Tile(dout, d_input, dout_shape, dinp_shape, stream_ptr);
  }

  void Tile(T *input_ptr, T *output_ptr, const std::vector<size_t> &inp_shape, const std::vector<size_t> &out_shape,
            void *stream_ptr) {
    size_t *input_shape_ptr = reinterpret_cast<size_t *>(workspace_ptr_[shape_ptr_idx_start_]);
    size_t *output_shape_ptr = reinterpret_cast<size_t *>(workspace_ptr_[shape_ptr_idx_start_ + 1]);
    CHECK_CUDA_RET_WITH_ERROR_NOTRACE(
      cudaMemcpyAsync(input_shape_ptr, &inp_shape[0], inp_shape.size() * sizeof(size_t), cudaMemcpyHostToDevice,
                      reinterpret_cast<cudaStream_t>(stream_ptr)),
      "For " + node_name_ + ", cudaMemcpyAsync input_shape failed.");
    CHECK_CUDA_RET_WITH_ERROR_NOTRACE(
      cudaMemcpyAsync(output_shape_ptr, &out_shape[0], out_shape.size() * sizeof(size_t), cudaMemcpyHostToDevice,
                      reinterpret_cast<cudaStream_t>(stream_ptr)),
      "For " + node_name_ + ", cudaMemcpyAsync output_shape failed.");
    size_t inp_size = 1;
    size_t out_size = 1;
    for (auto &v : inp_shape) {
      inp_size *= v;
    }
    for (auto &v : out_shape) {
      out_size *= v;
    }
    CalTile(out_size, inp_size, inp_shape.size(), input_shape_ptr, output_shape_ptr, input_ptr, output_ptr,
            reinterpret_cast<cudaStream_t>(stream_ptr));
  }

  void MulGrad(T *dout, T *mid_res, T *drht, T *dlft, const std::vector<size_t> &dlft_shape,
               const std::vector<size_t> &drht_shape, const std::vector<size_t> &dout_shape, void *stream_ptr) {
    bool broadcast_flag = false;
    for (size_t idx = 0; idx < dlft_shape.size(); ++idx) {
      if (dlft_shape[idx] != drht_shape[idx]) {
        broadcast_flag = true;
        break;
      }
    }
    if (!broadcast_flag) {
      Mul(drht, dout, dlft, drht_shape, dout_shape, dlft_shape, stream_ptr);
      Mul(mid_res, dout, drht, dlft_shape, dout_shape, drht_shape, stream_ptr);
      return;
    }
    std::vector<size_t> reverse_lft(dlft_shape);
    std::vector<size_t> reverse_rht(drht_shape);
    std::reverse(reverse_lft.begin(), reverse_lft.end());
    std::reverse(reverse_rht.begin(), reverse_rht.end());
    if (reverse_lft.size() > reverse_rht.size()) {
      reverse_rht.resize(reverse_lft.size(), 1);
    } else {
      reverse_lft.resize(reverse_rht.size(), 1);
    }
    std::vector<size_t> grad_x_reduce_idx;
    std::vector<size_t> grad_y_reduce_idy;
    const size_t n = reverse_lft.size();
    for (size_t i = 0; i < n; ++i) {
      const int64_t x_i = reverse_lft[i];
      const int64_t y_i = reverse_rht[i];
      const int64_t reduce_idx = SizeToLong(n - 1 - i);
      bool same = false;
      if (x_i == y_i) {
        same = true;
      } else if (x_i == 1) {
        grad_x_reduce_idx.emplace_back(reduce_idx);
      } else if (y_i == 1) {
        grad_y_reduce_idy.emplace_back(reduce_idx);
      } else {
        MS_LOG(EXCEPTION) << "For " << node_name_ << ", not compatible shape input for BroadcastGradientArgs.";
      }
      if (same && x_i == 1) {
        grad_x_reduce_idx.emplace_back(reduce_idx);
        grad_y_reduce_idy.emplace_back(reduce_idx);
        continue;
      }
    }
    std::reverse(grad_x_reduce_idx.begin(), grad_x_reduce_idx.end());
    std::reverse(grad_y_reduce_idy.begin(), grad_y_reduce_idy.end());
    T *work_ptr = reinterpret_cast<T *>(workspace_ptr_[shape_ptr_idx_start_ - 2]);
    if (grad_x_reduce_idx.size() > 0) {
      Mul(drht, dout, work_ptr, drht_shape, dout_shape, dout_shape, stream_ptr);
      ReduceSumCuda(work_ptr, dlft, dout_shape, grad_x_reduce_idx, stream_ptr);
    } else {
      Mul(drht, dout, dlft, drht_shape, dout_shape, dlft_shape, stream_ptr);
    }
    if (grad_y_reduce_idy.size() > 0) {
      Mul(mid_res, dout, work_ptr, dlft_shape, dout_shape, dout_shape, stream_ptr);
      ReduceSumCuda(work_ptr, drht, dout_shape, grad_y_reduce_idy, stream_ptr);
    } else {
      Mul(mid_res, dout, drht, dlft_shape, dout_shape, drht_shape, stream_ptr);
    }
  }
  void Transpose(T *input_ptr, T *output_ptr, const std::vector<size_t> &inp_shape,
                 const std::vector<size_t> &operate_info, void *stream_ptr) {
    // operate_info: input_axis
    size_t size = 1;
    size_t *d_shape_ptr = reinterpret_cast<size_t *>(workspace_ptr_[shape_ptr_idx_start_]);
    size_t *d_info_ptr = reinterpret_cast<size_t *>(workspace_ptr_[shape_ptr_idx_start_ + 1]);
    for (size_t idx = 0; idx < inp_shape.size(); ++idx) {
      size *= inp_shape[idx];
    }
    CHECK_CUDA_RET_WITH_ERROR_NOTRACE(
      cudaMemcpyAsync(d_shape_ptr, &inp_shape[0], inp_shape.size() * sizeof(size_t), cudaMemcpyHostToDevice,
                      reinterpret_cast<cudaStream_t>(stream_ptr)),
      "For " + node_name_ + ", Transpose's cudaMemcpyAsync failed.");
    CHECK_CUDA_RET_WITH_ERROR_NOTRACE(
      cudaMemcpyAsync(d_info_ptr, &operate_info[0], operate_info.size() * sizeof(size_t), cudaMemcpyHostToDevice,
                      reinterpret_cast<cudaStream_t>(stream_ptr)),
      "For " + node_name_ + ", Transpose's cudaMemcpyAsync failed.");
    CalTranspose<T>(size, input_ptr, d_shape_ptr, d_info_ptr, inp_shape.size(), output_ptr,
                    reinterpret_cast<cudaStream_t>(stream_ptr));
  }
  bool SegLeftEquation(const std::string &left_equation, const std::vector<std::vector<int64_t>> &input_shapes) {
    size_t cur_element = 0;
    auto found_ell = false;
    for (size_t idx = 0; idx < left_equation.length(); ++idx) {
      auto label = left_equation[idx];
      if (isalpha(label)) {
        left_elements_[cur_element].emplace_back(static_cast<size_t>(char_to_index(label)));
        element_count_[char_to_index(label)] += 1;
      } else if (label == '.') {
        if (found_ell) {
          MS_LOG(ERROR) << "For " << node_name_
                        << ", each operand can contain contain only one ellipsis, but it has been found again.";
          return false;
        }
        if ((idx + ELL_LEN - 1 >= left_equation.length()) || (left_equation[idx + 1] != '.') ||
            (left_equation[idx + ELL_LEN - 1] != '.')) {
          MS_LOG(ERROR) << "For " << node_name_
                        << ", An ellipsis in the equation should consist three \'.\', but got less than 3.";
          return false;
        }
        idx += (ELL_LEN - 1);
        found_ell = true;
        left_elements_[cur_element].emplace_back(ELL_VAL);
      } else if (label == ',') {
        if (found_ell) {
          if (left_elements_[cur_element].size() > input_shapes[cur_element].size() + 1) {
            MS_LOG(ERROR) << "For " << node_name_ << ", The number of subscript in " << cur_element
                          << " operand in the eqaution should match inputs[" << cur_element
                          << "].dim(), but it does not.";
            return false;
          }
          ell_dim_ = ell_dim_ > (input_shapes[cur_element].size() - left_elements_[cur_element].size() + 1)
                       ? ell_dim_
                       : (input_shapes[cur_element].size() - left_elements_[cur_element].size() + 1);
        } else if (left_elements_[cur_element].size() != input_shapes[cur_element].size()) {
          MS_LOG(ERROR) << "For " << node_name_ << ", The number of subscript in " << cur_element
                        << " operand in the eqaution should match inputs[" << cur_element
                        << "].dim(), but it does not.";
          return false;
        }
        ++cur_element;
        if (cur_element >= input_shapes.size()) {
          MS_LOG(ERROR) << "For " << node_name_
                        << ", the number of inputs should be equal to the number of inputs and equation's operand, but "
                           "it does not.";
          return false;
        }
        found_ell = false;
      } else {
        MS_LOG(ERROR) << "For " << node_name_ << ", Operand " << cur_element
                      << " in the equation contains invalid subscript, which can only consist of [a-zA-z].";
        return false;
      }
    }
    if (cur_element != input_shapes.size() - 1) {
      MS_LOG(ERROR)
        << "For " << node_name_
        << ", the number of inputs should be equal to the number of inputs and equation's operand, but it does not.";
      return false;
    }
    for (size_t i = 0; i < left_elements_.size(); ++i) {
      auto it = std::find(left_elements_[i].begin(), left_elements_[i].end(), ELL_VAL);
      if (left_elements_[i].size() != input_shapes[i].size() && it == left_elements_[i].end()) {
        MS_LOG(ERROR) << "For " << node_name_ << ", The number of subscript in " << i
                      << " operand in the eqaution should match inputs[" << i << "].dim(), but it does not.";
        return false;
      }
    }
    return true;
  }

  bool SegRightEquationWithArrow(const std::string &left_equation, const std::string &right_equation,
                                 ShapeVector *out_shape) {
    auto found_ell = false;
    if (right_equation.length() == 0) {
      out_size_ = 0;
      ell_idx_ = 0;
      perm_idx_ = element_shape_map_[ELL_VAL].size();
      out_shape->emplace_back(1);
      return true;
    }
    int64_t label_idx = 0;
    for (size_t idx = 0; idx < right_equation.length(); ++idx) {
      if (left_equation.find(right_equation[idx]) == std::string::npos) {
        MS_LOG(ERROR) << "For " << node_name_
                      << ", The label to the right of arrow in the equation must have appeared on the left, but the "
                      << right_equation[idx] << " not.";
        return false;
      }
      if (right_equation[idx] == '.') {
        if (found_ell) {
          MS_LOG(ERROR) << "For " << node_name_
                        << ", each operand can contain contain only one ellipsis, but it has been found again.";
          return false;
        }
        if ((idx + ELL_LEN - 1 >= right_equation.length()) || (right_equation[idx + 1] != '.') ||
            (right_equation[idx + ELL_LEN - 1] != '.')) {
          MS_LOG(ERROR) << "For " << node_name_
                        << ", An ellipsis in the equation should consist three \'.\', but got less than 3.";
          return false;
        }
        found_ell = true;
        ell_idx_ = idx;
        idx += (ELL_LEN - 1);
        label_idx = ELL_VAL;
        perm_idx_ += element_shape_map_[ELL_VAL].size();
      } else if (isalpha(right_equation[idx])) {
        label_idx = char_to_index(right_equation[idx]);
        label_perm_idx_[label_idx] = static_cast<int64_t>(perm_idx_);
        ++perm_idx_;
      } else {
        MS_LOG(ERROR) << "For " << node_name_ << ", Operand " << right_equation
                      << " in the equation contains invalid subscript, which can only consist of [a-zA-z].";
        return false;
      }
      out_shape->insert(out_shape->end(), element_shape_map_[label_idx].begin(), element_shape_map_[label_idx].end());
    }
    out_size_ = perm_idx_;
    if (!found_ell) {
      ell_idx_ = perm_idx_;
      perm_idx_ += element_shape_map_[ELL_VAL].size();
    }
    return true;
  }
  bool SegRightEquationWithoutArrow(const std::string &left_equation, std::vector<int64_t> *out_shape) {
    if (left_equation.find('.') != std::string::npos) {
      perm_idx_ = element_shape_map_[ELL_VAL].size();
      out_shape->insert(out_shape->begin(), element_shape_map_[ELL_VAL].begin(), element_shape_map_[ELL_VAL].end());
    }
    for (size_t idx = 0; idx < element_count_.size(); ++idx) {
      if (element_count_[idx] == 1) {
        out_shape->insert(out_shape->end(), element_shape_map_[idx].begin(), element_shape_map_[idx].end());
        label_perm_idx_[idx] = static_cast<int64_t>(perm_idx_);
        ++perm_idx_;
      }
    }
    out_size_ = perm_idx_;
    return true;
  }

  inline bool SetValue(size_t cur_element, int64_t value) {
    if (element_shape_map_.find(cur_element) != element_shape_map_.end()) {
      if (element_shape_map_[cur_element][0] != value) {
        MS_LOG(ERROR) << "For " << node_name_
                      << ", the same label in equation can only represent the same dimension in inputs, but the "
                      << static_cast<char>(cur_element + 'a') << " in equation not.";
        return false;
      }
    } else {
      element_shape_map_[cur_element] = {value};
    }

    return true;
  }

  bool ElementMapShape(const std::vector<std::vector<int64_t>> &input_shapes) {
    for (size_t idx_input = 0; idx_input < input_shapes.size(); ++idx_input) {
      auto cur_shape = input_shapes[idx_input];
      size_t idx_left = 0;
      while (idx_left < left_elements_[idx_input].size() && left_elements_[idx_input][idx_left] != ELL_VAL) {
        auto cur_element = left_elements_[idx_input][idx_left];
        if (!SetValue(cur_element, input_shapes[idx_input][idx_left])) {
          return false;
        }
        ++idx_left;
      }

      if (idx_left != left_elements_[idx_input].size()) {
        auto idx_element_right = left_elements_[idx_input].size() - 1;
        auto idx_shape_right = input_shapes[idx_input].size() - 1;
        while (idx_element_right > idx_left && left_elements_[idx_input][idx_element_right] != ELL_VAL) {
          auto cur_element = left_elements_[idx_input][idx_element_right];
          if (!SetValue(cur_element, input_shapes[idx_input][idx_shape_right])) {
            return false;
          }
          --idx_shape_right;
          --idx_element_right;
        }
        ShapeVector temp_vec(input_shapes[idx_input].begin() + idx_left,
                             input_shapes[idx_input].begin() + idx_shape_right + 1);
        if (!AdjustElementMapShape(temp_vec)) {
          return false;
        }
      }
    }
    return true;
  }

  bool AdjustElementMapShape(const ShapeVector &temp_vec) {
    const size_t ellipsis_val_num = 52;
    auto iter = element_shape_map_.find(ellipsis_val_num);
    if (iter != element_shape_map_.end()) {
      ShapeVector cur_vec = iter->second;
      if (cur_vec != temp_vec) {
        if (temp_vec.empty() || cur_vec.empty()) {
          element_shape_map_[ellipsis_val_num] = temp_vec.empty() ? cur_vec : temp_vec;
        } else {
          MS_LOG(ERROR)
            << "For " << node_name_
            << ", the same ellipsis in equation can only represent the same dimension in inputs, but it does not.";
          return false;
        }
      }
    } else {
      element_shape_map_[ellipsis_val_num] = temp_vec;
    }
    return true;
  }

  void CalAxisShape(const std::vector<size_t> &axis_val, const ShapeVector &shape_val, size_t *idx,
                    ShapeVector *re_shape, std::vector<size_t> *res_trans_axis) {
    for (auto val : axis_val) {
      (*re_shape)[*idx] = shape_val[val];
      (*res_trans_axis)[val] = (*idx)++;
    }
  }
  bool MulOrDot(size_t sum_dims_size, const ShapeVector &sig_src_shape, ShapeVector *res_inp_shape,
                std::vector<OpStruct> *res_op) {
    ShapeVector res_out_shape;
    if (sum_dims_size == 0) {
      res_out_shape = ShapeVector(res_inp_shape->size());
      for (size_t idx = 0; idx < res_inp_shape->size(); ++idx) {
        res_out_shape[idx] = (*res_inp_shape)[idx] == 1 ? sig_src_shape[idx] : (*res_inp_shape)[idx];
      }
      res_op->emplace_back(
        std::make_tuple("Mul", (*res_inp_shape), Convert2SizeTClipNeg(sig_src_shape), res_out_shape));
      (*res_inp_shape) = res_out_shape;
      return true;
    }

    if (sum_dims_size == res_inp_shape->size()) {
      res_out_shape = {1};
      res_op->emplace_back(
        std::make_tuple("Dot", (*res_inp_shape), Convert2SizeTClipNeg(sig_src_shape), res_out_shape));
      (*res_inp_shape) = res_out_shape;
      return true;
    }
    return false;
  }
  void SumPair(std::vector<OpStruct> *res_op, std::vector<OpStruct> *single_op, std::vector<size_t> *sum_dims,
               ShapeVector *res_inp_shape, const ShapeVector &sig_src_shape) {
    if (MulOrDot(sum_dims->size(), sig_src_shape, res_inp_shape, res_op)) {
      return;
    }
    size_t size = res_inp_shape->size();
    std::vector<bool> sum_dims_bool = std::vector<bool>(size, false);
    for (auto idx : (*sum_dims)) {
      sum_dims_bool[idx] = true;
    }

    std::vector<size_t> lo;
    std::vector<size_t> ro;
    std::vector<size_t> lro;
    int64_t lo_size = 1;
    int64_t ro_size = 1;
    int64_t lro_size = 1;
    int64_t sum_size = 1;

    ShapeVector sig_out_shape;
    auto sig_inp_shape = sig_src_shape;
    std::vector<size_t> op_info;
    ShapeVector res_out_shape;
    for (size_t idx = 0; idx < res_inp_shape->size(); ++idx) {
      bool sl = (*res_inp_shape)[idx] > 1;
      bool sr = sig_inp_shape[idx] > 1;

      if (sum_dims_bool[idx]) {
        if (sl && sr) {
          sum_size *= (*res_inp_shape)[idx];
        } else if (sl) {
          op_info = {static_cast<size_t>(idx)};
          res_out_shape = (*res_inp_shape);
          res_out_shape[idx] = 1;
          res_op->emplace_back(std::make_tuple("ReduceSum", (*res_inp_shape), op_info, res_out_shape));
          (*res_inp_shape) = res_out_shape;
        } else if (sr) {
          op_info = {static_cast<size_t>(idx)};
          sig_out_shape = sig_inp_shape;
          sig_out_shape[idx] = 1;
          single_op->emplace_back(std::make_tuple("ReduceSum", sig_inp_shape, op_info, sig_out_shape));
          sig_inp_shape = sig_out_shape;
        }
      } else if (sl && sr) {
        lro.emplace_back(idx);
        lro_size *= (*res_inp_shape)[idx];
      } else if (sl) {
        lo.emplace_back(idx);
        lo_size *= (*res_inp_shape)[idx];
      } else {
        ro.emplace_back(idx);
        ro_size *= sig_inp_shape[idx];
      }
    }

    std::vector<size_t> res_trans_axis;
    res_trans_axis.insert(res_trans_axis.end(), lro.begin(), lro.end());
    res_trans_axis.insert(res_trans_axis.end(), lo.begin(), lo.end());
    res_trans_axis.insert(res_trans_axis.end(), sum_dims->begin(), sum_dims->end());
    res_trans_axis.insert(res_trans_axis.end(), ro.begin(), ro.end());
    // tranpose
    res_out_shape = (*res_inp_shape);
    for (size_t idx_axis = 0; idx_axis < res_trans_axis.size(); ++idx_axis) {
      res_out_shape[idx_axis] = (*res_inp_shape)[res_trans_axis[idx_axis]];
    }
    res_op->emplace_back(std::make_tuple("Transpose", (*res_inp_shape), res_trans_axis, res_out_shape));

    std::vector<size_t> sig_trans_axis;
    sig_trans_axis.insert(sig_trans_axis.end(), lro.begin(), lro.end());
    sig_trans_axis.insert(sig_trans_axis.end(), sum_dims->begin(), sum_dims->end());
    sig_trans_axis.insert(sig_trans_axis.end(), ro.begin(), ro.end());
    sig_trans_axis.insert(sig_trans_axis.end(), lo.begin(), lo.end());
    // tranpose
    sig_out_shape = sig_inp_shape;
    for (size_t idx_axis = 0; idx_axis < sig_trans_axis.size(); ++idx_axis) {
      sig_out_shape[idx_axis] = sig_inp_shape[sig_trans_axis[idx_axis]];
    }
    single_op->emplace_back(std::make_tuple("Transpose", sig_inp_shape, sig_trans_axis, sig_out_shape));

    ShapeVector res_inp_reshape = {lro_size, lo_size, sum_size};
    ShapeVector sig_inp_reshape = {lro_size, sum_size, ro_size};
    ShapeVector res_out_reshape = {lro_size, lo_size, ro_size};
    res_op->emplace_back(
      std::make_tuple("Bmm", res_inp_reshape, Convert2SizeTClipNeg(sig_inp_reshape), res_out_reshape));

    ShapeVector res_re_shape(LongToSize(lro.size() + lo.size() + sum_dims->size() + ro.size()));
    size_t idx = 0;
    CalAxisShape(lro, (*res_inp_shape), &idx, &res_re_shape, &res_trans_axis);
    CalAxisShape(lo, (*res_inp_shape), &idx, &res_re_shape, &res_trans_axis);
    ShapeVector shape_val(sum_dims_bool.size(), 1);
    CalAxisShape((*sum_dims), shape_val, &idx, &res_re_shape, &res_trans_axis);
    CalAxisShape(ro, sig_inp_shape, &idx, &res_re_shape, &res_trans_axis);
    (*res_inp_shape) = res_re_shape;
    res_out_shape = (*res_inp_shape);
    for (size_t idx_axis = 0; idx_axis < res_trans_axis.size(); ++idx_axis) {
      res_out_shape[idx_axis] = (*res_inp_shape)[res_trans_axis[idx_axis]];
    }
    res_op->emplace_back(std::make_tuple("Transpose", (*res_inp_shape), res_trans_axis, res_out_shape));
    res_inp_shape->clear();
    idx = 0;
    while (idx < size) {
      if (!sum_dims_bool[idx]) {
        res_inp_shape->emplace_back(res_out_shape[idx]);
      }
      ++idx;
    }
    return;
  }

  bool CalOutShape(const std::string &equation, const std::vector<std::vector<int64_t>> &input_shapes,
                   std::vector<int64_t> *out_shape) {
    std::string seg_arrow = "->";
    auto seg_pos = equation.find(seg_arrow);
    std::string left_equation = equation.substr(0, seg_pos);
    bool ret_flag = true;
    ret_flag = SegLeftEquation(left_equation, input_shapes);
    if (!ret_flag) {
      return ret_flag;
    }
    ret_flag = ElementMapShape(input_shapes);
    if (!ret_flag) {
      return ret_flag;
    }
    ell_dim_ = element_shape_map_[ELL_VAL].size();

    if (seg_pos == std::string::npos) {
      ret_flag = SegRightEquationWithoutArrow(left_equation, out_shape);
    } else {
      auto right_equation = equation.substr(seg_pos + 2, equation.length());
      ret_flag = SegRightEquationWithArrow(left_equation, right_equation, out_shape);
    }
    for (size_t idx = 0; idx < LABEL_NUM; ++idx) {
      if (element_count_[idx] > 0 && label_perm_idx_[idx] == -1) {
        label_perm_idx_[idx] = static_cast<int64_t>(perm_idx_);
        ++perm_idx_;
      }
    }
    return ret_flag;
  }
  void StatSingleOp(const std::vector<std::vector<int64_t>> &input_shapes,
                    std::vector<std::vector<OpStruct>> *single_op) {
    std::vector<size_t> op_info;
    for (size_t idx = 0; idx < input_shapes.size(); ++idx) {
      std::vector<int64_t> trans_axis(perm_idx_, -1);
      auto &elements = left_elements_[idx];
      std::vector<int64_t> label_dim(LABEL_NUM, -1);
      int64_t j = 0;
      auto sig_inp_shape = input_shapes[idx];
      auto sig_out_shape = input_shapes[idx];
      for (auto element : elements) {
        if (element == ELL_VAL) {
          for (size_t k = 0; k < ell_dim_; ++k) {
            trans_axis[ell_idx_ + k] = j++;
          }
        } else if (label_dim[element] != -1) {
          // diagonal,
          op_info = {static_cast<size_t>(label_dim[element]), static_cast<size_t>(j), 0};
          sig_out_shape.erase(sig_out_shape.begin() + j);
          (*single_op)[idx].emplace_back(std::make_tuple("Diagonal", sig_inp_shape, op_info, sig_out_shape));
          sig_inp_shape = sig_out_shape;
        } else {
          label_dim[element] = j;
          trans_axis[label_perm_idx_[element]] = j++;
        }
      }
      op_info.clear();
      for (auto &val : trans_axis) {
        if (val == -1) {
          sig_inp_shape.insert(sig_inp_shape.end(), 1);
          val = j++;
        }
        op_info.emplace_back(static_cast<size_t>(val));
      }
      // tranpose
      sig_out_shape = sig_inp_shape;
      for (size_t idx_axis = 0; idx_axis < trans_axis.size(); ++idx_axis) {
        sig_out_shape[idx_axis] = sig_inp_shape[trans_axis[idx_axis]];
      }
      (*single_op)[idx].emplace_back(std::make_tuple("Transpose", sig_inp_shape, op_info, sig_out_shape));
    }
  }
  void StatCalProcess(const std::vector<std::vector<int64_t>> &input_shapes,
                      std::vector<std::vector<OpStruct>> *single_op, std::vector<OpStruct> *res_op) {
    StatSingleOp(input_shapes, single_op);
    // dim_last_op
    std::vector<size_t> dim_last_op(perm_idx_);
    for (size_t dim = 0; dim < perm_idx_; ++dim) {
      auto shape_a = std::get<3>((*single_op)[0].back());
      for (size_t idx = 1; idx < input_shapes.size(); ++idx) {
        auto shape_b = std::get<3>((*single_op)[idx].back());
        auto dim_val = shape_b[dim];
        if (dim_val != 1) {
          dim_last_op[dim] = idx;
        }
      }
    }
    size_t dim = out_size_;
    auto sig_inp_shape = std::get<3>((*single_op)[0].back());
    for (size_t idx = dim; idx < perm_idx_; ++idx, ++dim) {
      if (dim_last_op[idx] == 0) {
        if (sig_inp_shape[dim] == 1) {
          sig_inp_shape.erase(sig_inp_shape.begin() + dim);
        } else {
          std::vector<size_t> op_info = {dim};
          auto sig_out_shape = sig_inp_shape;
          sig_out_shape.erase(sig_out_shape.begin() + dim);
          (*single_op)[0].emplace_back(std::make_tuple("ReduceSum", sig_inp_shape, op_info, sig_out_shape));
          sig_inp_shape = sig_out_shape;
        }
        --dim;
      }
    }
    auto res_inp_shape = sig_inp_shape;
    for (size_t idx = 1; idx < input_shapes.size(); ++idx) {
      size_t dim = out_size_;
      auto sig_inp_shape = std::get<3>((*single_op)[idx].back());
      std::vector<size_t> sum_dims;
      for (size_t j = dim; j < perm_idx_; ++j, ++dim) {
        if (dim_last_op[j] < idx) {
          sig_inp_shape.erase(sig_inp_shape.begin() + dim);
          --dim;
        } else if (dim_last_op[j] == idx) {
          if (res_inp_shape[dim] == 1) {
            std::vector<size_t> op_info = {dim};
            auto sig_out_shape = sig_inp_shape;
            sig_out_shape.erase(sig_out_shape.begin() + dim);
            (*single_op)[idx].emplace_back(std::make_tuple("ReduceSum", sig_inp_shape, op_info, sig_out_shape));
            sig_inp_shape = sig_out_shape;
            res_inp_shape.erase(res_inp_shape.begin() + dim);
            --dim;
          } else {
            sum_dims.emplace_back(dim);
          }
        }
      }
      SumPair(res_op, &(*single_op)[idx], &sum_dims, &res_inp_shape, sig_inp_shape);
    }
    if (res_op->size() > 0) {
      auto name = std::get<0>((*res_op)[0]);
      while (two_op_func_.count(name) == 0) {
        (*single_op)[0].emplace_back((*res_op)[0]);
        res_op->erase(res_op->begin());
        name = std::get<0>((*res_op)[0]);
      }
    }
  }

 private:
  size_t ell_idx_ = 0;
  size_t ell_dim_ = 0;
  size_t perm_idx_ = 0;
  size_t out_size_ = 0;
  std::string type_name_;
  std::string node_name_;
  size_t shape_ptr_idx_start_ = 0;
  TypeId data_type_;
  cublasOperation_t transpose_x1_;
  cublasOperation_t transpose_x2_;
  std::vector<size_t> element_count_;
  std::unordered_map<size_t, std::vector<int64_t>> element_shape_map_;
  std::vector<std::vector<size_t>> left_elements_;
  std::vector<int64_t> label_perm_idx_;
  std::vector<void *> workspace_ptr_;
  std::set<std::string> two_op_func_ = {"Mul", "Dot", "Bmm"};
};
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_EINSUM_HELPER_H_
