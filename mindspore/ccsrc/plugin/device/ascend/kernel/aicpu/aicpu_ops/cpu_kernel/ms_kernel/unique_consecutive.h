/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2021-2022. All rights reserved.
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

#include <functional>
#include <map>

#include "cpu_ops_kernel.h"

namespace aicpu {
class UniqueConsecutiveCpuKernel : public CpuKernel {
 public:
  UniqueConsecutiveCpuKernel() = default;
  ~UniqueConsecutiveCpuKernel() override = default;
  uint32_t Compute(CpuKernelContext &ctx) override;

 private:
  template <typename T1, typename T2>
  uint32_t DoComputeNone(const CpuKernelContext &ctx);
  template <typename T1, typename T2>
  uint32_t DoComputeDim(const CpuKernelContext &ctx, const int32_t tmp_axis);
  void DefaultSet(const CpuKernelContext &ctx);
  template <typename T1>
  void OutputYSet(const std::vector<int64_t> &y_shape_, const std::vector<int64_t> &input_shape_, int32_t axis,
                  T1 *y_dataptr, const std::vector<std::vector<T1>> &out_data_);
  uint32_t ExtraParamCheck(CpuKernelContext &ctx);
  template <typename T2>
  void SetOuputIdxandCount(const CpuKernelContext &ctx, const std::vector<int64_t> &idx_shape_,
                           const std::vector<int64_t> &count_shape_, T2 *idx_dataptr, T2 *count_dataptr);
  template <typename T2>
  uint32_t DtypeMapNone(const CpuKernelContext &ctx, DataType x_dtype);
  template <typename T2>
  uint32_t DtypeMapDim(const CpuKernelContext &ctx, int32_t tmp_axis, DataType x_dtype);

  uint32_t DoCompute(const CpuKernelContext &ctx);
  int32_t axis_;
  bool return_idx_;
  bool return_counts_;
  DataType input_type_ = DT_INT32;
  DataType idx_dtype_ = DT_INT32;
  DataType count_dtype_ = DT_INT32;
};
}  // namespace aicpu
