/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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
#ifndef AICPU_KERNELS_NORMALIZED_DCTN_H_
#define AICPU_KERNELS_NORMALIZED_DCTN_H_

#include <complex>
#include <utility>
#include <map>
#include <functional>
#include <algorithm>
#include <memory>
#include <vector>
#include <securec.h>
#include "inc/ms_cpu_kernel.h"
#include "cpu_kernel/utils/fft_helper.h"

namespace aicpu {
const uint32_t kIndex0 = 0;
const uint32_t kDCTTypeIndex = 1;
const uint32_t kDCTSIndex = 2;
const uint32_t kDCTDimIndex = 3;
const uint32_t kDCTNormIndex = 4;
class DCTNCpuKernel : public CpuKernel {
 public:
  ~DCTNCpuKernel() = default;

  DataType input_type_;
  DataType output_type_;
  std::string op_name_;
  std::size_t dim_index_ = kDCTDimIndex;
  std::size_t norm_index_ = kDCTNormIndex;

 protected:
  uint32_t Compute(CpuKernelContext &ctx) override;

 private:
  uint32_t ParseKernelParam(CpuKernelContext &ctx);
  void DCTNGetInputs(CpuKernelContext &ctx);
  void DCTNGetAttr();

  bool s_is_none_;
  bool dim_is_none_;
  bool forward_;
  bool is_ortho_;
  int64_t x_rank_;
  int64_t input_element_nums_;
  int64_t calculate_element_nums_;
  int64_t fft_nums_;
  double norm_weight_;
  NormMode norm_;

  int64_t dct_type_;
  std::vector<int64_t> dim_;
  std::vector<int64_t> s_;
  std::vector<int64_t> tensor_shape_;
  std::vector<int64_t> calculate_shape_;
};
}  // namespace aicpu
#endif  //  AICPU_DCTN_H
