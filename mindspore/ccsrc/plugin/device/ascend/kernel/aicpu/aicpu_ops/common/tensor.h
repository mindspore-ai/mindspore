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
#ifndef AICPU_OPS_COMMON_TENSOR_H_
#define AICPU_OPS_COMMON_TENSOR_H_

#include <atomic>
#include <memory>
#include <string>
#include <vector>
#include <map>

namespace aicpu {
namespace ms {
class Tensor {
 public:
  Tensor() = default;
  ~Tensor() = default;
  const uint8_t *GetData() const;
  size_t GetSize() const;
  void SetData(uint8_t *data, size_t size);

 private:
  uint8_t *tensor_ptr_;
  size_t tensor_len_;
};
}  // namespace ms
}  // namespace aicpu
#endif  // AICPU_OPS_COMMON_TENSOR_H_
