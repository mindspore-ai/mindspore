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

#include "backend/kernel_compiler/cpu/matrix_band_part_cpu_kernel.h"
#include <algorithm>
#include "utils/ms_utils.h"

namespace mindspore {
namespace kernel {
template <typename T>
void MatrixBandPartCPUKernel<T>::InitKernel(const CNodePtr &kernel_node) {
  shapes_ = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
  dim_size_ = shapes_.size();
  if (shapes_.size() < kDim2) {
    MS_LOG(EXCEPTION) << "Wrong array shape, A should be a matrix max than 2.";
  }
  m_ = shapes_[dim_size_ - kDim2];
  n_ = shapes_[dim_size_ - kDim1];
  for (size_t i = 0; i < shapes_.size() - kDim2; i++) {
    out_range_size_ *= shapes_[i];
  }
  matrix_size_ = out_range_size_ * m_ * n_;
}

template <typename T>
bool MatrixBandPartCPUKernel<T>::Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
                                        const std::vector<AddressPtr> &outputs) {
  T *in_value = reinterpret_cast<T *>(inputs[0]->addr);
  const int64_t *lower = reinterpret_cast<int64_t *>(inputs[1]->addr);
  const int64_t *upper = reinterpret_cast<int64_t *>(inputs[2]->addr);
  T *out_value = reinterpret_cast<T *>(outputs[0]->addr);

  const size_t l = (*lower < 0 || *lower > static_cast<int64_t>(m_)) ? m_ : *lower;
  const size_t u = (*upper < 0 || *upper > static_cast<int64_t>(n_)) ? n_ : *upper;
  memset(out_value, 0, matrix_size_ * sizeof(T));
  if (l >= m_ && u >= n_) {
    memcpy_s(out_value, matrix_size_ * sizeof(T), in_value, matrix_size_ * sizeof(T));
    return true;
  }
  for (size_t k = 0; k < out_range_size_; k++) {
    for (size_t i = 0; i < std::min(m_, l + n_); i++) {
      const size_t s = i < l ? 0 : i - l;
      // When i = n - u, end is n -1, because end pos is start from 0
      const size_t e = i >= n_ - u ? n_ - 1 : i + u;
      const size_t offset = k * m_ * n_ + i * n_;
      memcpy_s(out_value + offset + s, matrix_size_ * sizeof(T), in_value + offset + s, (e - s + 1) * sizeof(T));
    }
  }
  return true;
}
}  // namespace kernel
}  // namespace mindspore
