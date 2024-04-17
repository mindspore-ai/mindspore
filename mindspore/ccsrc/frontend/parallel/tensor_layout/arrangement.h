/**
 * Copyright 2019-2024 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_FRONTEND_PARALLEL_TENSOR_LAYOUT_ARRANGEMENT_H_
#define MINDSPORE_CCSRC_FRONTEND_PARALLEL_TENSOR_LAYOUT_ARRANGEMENT_H_

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>
#include <utility>
#include "frontend/parallel/status.h"
#include "frontend/parallel/tensor_layout/array.h"

namespace mindspore {
namespace parallel {
class Arrangement : public Array {
 public:
  Arrangement() : size_(1) {}
  ~Arrangement() override = default;
  Status Init(const Shape &array) override;
  Status UpdateTensorShape(size_t index, int64_t update_value);
  int64_t size() const { return size_; }
  Shape GetFrontElementByValue(int64_t value) const;
  std::shared_ptr<std::vector<Arrangement>> GetExpandShapeList(const Arrangement &expand_shape) const;
  Shape ComputeReverseAccumulateSumInReverseOrder() const;
  std::shared_ptr<Arrangement> GetExpandedShapeByExpandListReserveLeft(
    const std::vector<Arrangement> &expand_list) const;
  std::shared_ptr<Arrangement> GetExpandedShapeByExpandListRemoveLeft(
    const std::vector<Arrangement> &expand_list) const;
  std::shared_ptr<std::pair<std::vector<Arrangement>, Arrangement>> GetExpandShapeListPair(
    const Arrangement &expand_shape) const;
  std::shared_ptr<Arrangement> GetUnifiedShape(const Arrangement &in2) const;
  std::vector<size_t> GetSqueezeIdx() const;
  Arrangement GetSqueezeArrangement() const;

 private:
  bool IsValidArrangement();
  void ComputeSize();
  int64_t size_;
};
}  // namespace parallel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_FRONTEND_PARALLEL_TENSOR_LAYOUT_ARRANGEMENT_H_
