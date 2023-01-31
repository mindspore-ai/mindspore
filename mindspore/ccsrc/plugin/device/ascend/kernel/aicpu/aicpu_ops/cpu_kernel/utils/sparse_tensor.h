/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2022. All rights reserved.
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

#ifndef AICPU_SPARSETENSOR_H
#define AICPU_SPARSETENSOR_H

#include <algorithm>
#include <memory>

#include "cpu_tensor.h"
#include "eigen_tensor.h"
#include "utils/kernel_util.h"
#include "cpu_kernel_utils.h"
#include "kernel_log.h"
#include "sparse_group.h"
#include "status.h"

namespace aicpu {
template <typename T>
const T SubtleMustCopy(const T &x) {
  auto *to_x = reinterpret_cast<const volatile T *>(&x);
  return *to_x;
}
}  // namespace aicpu

namespace aicpu {
class SparseTensor {
 public:
  SparseTensor() : dims_(0) {}
  ~SparseTensor() = default;

  /*
   * create sparse tensor
   * @param ix: index tensor
   * @param tensorvals: tensorvals tensor
   * @param shape: shape vec
   * @param order: order vec
   * @return uint32_t: 0->success other->failed
   */
  uint32_t CreateSparseTensor(Tensor *ix, Tensor *tensorvals, std::vector<int64_t> shape, std::vector<int64_t> order);

  /*
   * sparse indices valid
   * @return uint32_t: 0->success other->failed
   */
  uint32_t IndicesValid(CpuKernelContext &ctx) const;

  /*
   * group sparse tensor
   * @return GroupIterable
   */
  GroupIterable group(const std::vector<int64_t> &group_ix) const;
  /*
   * sparse eigen tensor indices valid
   * @return uint32_t: 0->success other->failed
   */
  int dims() const { return dims_; }

  std::shared_ptr<EigenTensor> indices() const { return ix_; }

  std::shared_ptr<EigenTensor> values() const { return vals_; }

  std::vector<int64_t> shape() const { return shape_; }

  template <typename T>
  uint32_t EigenTensorIndicesValidCheck(int64_t dims_size) const {
    const auto ix_t = ix_->matrix<T>();
    for (int64_t n = 1; n < dims_size; ++n) {
      bool valid = true;
      bool different = false;
      bool increasing = true;
      for (int32_t di = 0; di < dims_; ++di) {
        if (ix_t(n, di) < 0 || ix_t(n, di) >= shape_[di]) {
          valid = false;
        }
        int64_t diff = ix_t(n, order_[di]) - ix_t(n - 1, order_[di]);
        if (diff > 0) {
          different = true;
        }
        if (!different && diff < 0) {
          increasing = false;
        }
      }
      if (!valid) {
        KERNEL_LOG_ERROR("Indices is out of bounds, index=%lld.", n);
        return static_cast<uint32_t>(KERNEL_STATUS_PARAM_INVALID);
      }
      if (!increasing) {
        KERNEL_LOG_ERROR("indices is out of order, index=%lld.", n);
        return static_cast<uint32_t>(KERNEL_STATUS_PARAM_INVALID);
      }
      if (!different) {
        KERNEL_LOG_ERROR("indices is repeated, index=%lld.", n);
        return static_cast<uint32_t>(KERNEL_STATUS_PARAM_INVALID);
      }
    }
    return static_cast<uint32_t>(KERNEL_STATUS_OK);
  }
  /*
   * sparse eigen tensor indices valid
   * @return uint32_t: 0->success other->failed
   */
  template <typename T>
  uint32_t EigenTensorIndicesValidParaCheck(const CpuKernelContext &ctx, int64_t dims_size) const {
    uint32_t min_core_num = 1;
    int64_t max_core_num = std::max(min_core_num, aicpu::CpuKernelUtils::GetCPUNum(ctx) - kResvCpuNum);
    uint32_t result = static_cast<uint32_t>(KERNEL_STATUS_OK);
    (void)aicpu::CpuKernelUtils::ParallelFor(ctx, dims_size, dims_size / max_core_num,
                                             [&](std::int64_t begin, std::int64_t end) {
                                               int64_t start = begin;
                                               if (begin == 0) {
                                                 start = begin + 1;
                                               }
                                               const auto ix_t = ix_->matrix<T>();
                                               for (int64_t n = start; n < end; ++n) {
                                                 bool valid = true;
                                                 bool different = false;
                                                 bool increasing = true;
                                                 for (int32_t di = 0; di < dims_; ++di) {
                                                   if (ix_t(n, di) < 0 || ix_t(n, di) >= shape_[di]) {
                                                     valid = false;
                                                   }
                                                   int64_t diff = ix_t(n, order_[di]) - ix_t(n - 1, order_[di]);
                                                   if (diff > 0) {
                                                     different = true;
                                                   }
                                                   if (!different && diff < 0) {
                                                     increasing = false;
                                                   }
                                                 }
                                                 if (!valid) {
                                                   KERNEL_LOG_ERROR("Indices is out of bounds, index=%lld.", n);
                                                   result = static_cast<uint32_t>(KERNEL_STATUS_PARAM_INVALID);
                                                   return;
                                                 }
                                                 if (!increasing) {
                                                   KERNEL_LOG_ERROR("indices is out of order, index=%lld.", n);
                                                   result = static_cast<uint32_t>(KERNEL_STATUS_PARAM_INVALID);
                                                   return;
                                                 }
                                                 if (!different) {
                                                   KERNEL_LOG_ERROR("indices is repeated, index=%lld.", n);
                                                   result = static_cast<uint32_t>(KERNEL_STATUS_PARAM_INVALID);
                                                   return;
                                                 }
                                               }
                                             });
    return result;
  }
  /*
   * sparse eigen tensor indices valid
   * @return uint32_t: 0->success other->failed
   */
  template <typename T>
  uint32_t EigenTensorIndicesValid(const CpuKernelContext &ctx) const {
    const auto ix_t = ix_->matrix<T>();
    int64_t dims_size =
      (ix_->GetTensor()->GetTensorShape()->GetDims() == 0) ? 1 : ix_->GetTensor()->GetTensorShape()->GetDimSize(0);
    if (dims_size > 0) {
      for (int32_t di = 0; di < dims_; ++di) {
        if ((ix_t(0, di) < 0) || (ix_t(0, di) >= shape_[di])) {
          KERNEL_LOG_ERROR("Indices is out of bounds, index=0.");
          return KERNEL_STATUS_PARAM_INVALID;
        }
      }
    }
    const int64_t paralled_data_size = 16 * 1024;
    if (dims_size < paralled_data_size) {
      return EigenTensorIndicesValidCheck<T>(dims_size);
    } else {
      return EigenTensorIndicesValidParaCheck<T>(ctx, dims_size);
    }
  }

  /*
   * validate sparse to dense
   * @param output: output tensor
   * @return bool: true->success false->failed
   */
  bool ValidateToDense(const Tensor *out) const;

  /*
   * sparse tensor to dense tensor
   * @param output: output tensor
   * @return uint32_t: 0->success other->failed
   */
  template <typename IndiceT, typename ValueT>
  uint32_t ToDenseParallel(const CpuKernelContext &ctx, Tensor *output) {
    EigenTensor outputET(output, output->GetData());
    auto output_t = outputET.flat<ValueT>();
    auto ix_t = ix_->matrix<IndiceT>();
    std::vector<int64_t> strides(dims_);
    const auto &out_shape = output->GetTensorShape();
    if (dims_ > 0) {
      strides[dims_ - 1] = 1;
    }
    for (int32_t d = dims_ - 2; d >= 0; --d) {
      strides[d] = strides[d + 1] * out_shape->GetDimSize(d + 1);
    }
    auto vals_t = vals_->vec<ValueT>();
    int64_t vals_size = vals_t.dimension(0);
    uint32_t min_core_num = 1;
    int64_t max_core_num = std::max(min_core_num, aicpu::CpuKernelUtils::GetCPUNum(ctx) - kResvCpuNum);
    uint32_t result = static_cast<uint32_t>(KERNEL_STATUS_OK);
    auto parallel_proc = [&](std::int64_t begin, std::int64_t end) {
      for (int64_t n = begin; n < end; ++n) {
        bool invalid_dims = false;
        int64_t ix = 0;
        for (int d = 0; d < dims_; ++d) {
          const int64_t ix_n_d = ix_t(n, d);
          if (ix_n_d > out_shape->GetDimSize(d)) {
            invalid_dims = true;
          }
          ix += strides[d] * ix_n_d;
        }
        if (invalid_dims) {
          result = static_cast<uint32_t>(KERNEL_STATUS_INNER_ERROR);
          KERNEL_LOG_ERROR("Sparse to dense got invalid dims.");
          return;
        }
        output_t(ix) = vals_t(n);
      }
      return;
    };
    KERNEL_HANDLE_ERROR(aicpu::CpuKernelUtils::ParallelFor(ctx, vals_size, vals_size / max_core_num, parallel_proc),
                        "SparseToDense Compute failed.");
    return result;
  }

  /*
   * sparse tensor to dense tensor
   * @param output: output tensor
   * @return uint32_t: 0->success other->failed
   */
  template <typename IndiceT, typename ValueT>
  uint32_t ToDense(const CpuKernelContext &ctx, Tensor *output) {
    KERNEL_LOG_INFO("Start to execute ToDense.");
    if (output == nullptr || output->GetData() == nullptr) {
      KERNEL_LOG_ERROR("Output tensor is nullptr.");
      return KERNEL_STATUS_INNER_ERROR;
    }
    if (!ValidateToDense(output)) {
      KERNEL_LOG_ERROR("Validate to dense param failed.");
      return KERNEL_STATUS_INNER_ERROR;
    }
    auto vals_t = vals_->vec<ValueT>();
    int64_t vals_size = vals_t.dimension(0);
    const int64_t paralled_data_size = 16 * 1024;
    if (vals_size >= paralled_data_size) {
      return ToDenseParallel<IndiceT, ValueT>(ctx, output);
    }
    EigenTensor outputET(output, output->GetData());
    auto output_t = outputET.flat<ValueT>();
    auto ix_t = ix_->matrix<IndiceT>();
    std::vector<int64_t> strides(dims_);
    const auto &out_shape = output->GetTensorShape();
    if (dims_ > 0) {
      strides[dims_ - 1] = 1;
    }
    for (int32_t d = dims_ - 2; d >= 0; --d) {
      strides[d] = strides[d + 1] * out_shape->GetDimSize(d + 1);
    }
    for (int64_t n = 0; n < vals_size; ++n) {
      bool invalid_dims = false;
      int64_t ix = 0;
      for (int d = 0; d < dims_; ++d) {
        const int64_t ix_n_d = ix_t(n, d);
        if (ix_n_d > out_shape->GetDimSize(d)) {
          invalid_dims = true;
        }
        ix += strides[d] * ix_n_d;
      }
      if (invalid_dims) {
        KERNEL_LOG_ERROR("Sparse to dense got invalid dims.");
        return KERNEL_STATUS_INNER_ERROR;
      }
      output_t(ix) = vals_t(n);
    }
    return KERNEL_STATUS_OK;
  }

 private:
  std::shared_ptr<EigenTensor> ix_;
  std::shared_ptr<EigenTensor> vals_;
  std::vector<int64_t> shape_;
  std::vector<int64_t> order_;
  int32_t dims_;
};
}  // namespace aicpu

#endif  // AICPU_SPARSETENSOR_H
