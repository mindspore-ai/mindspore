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
#ifndef MINDSPORE_CORE_OPS_MATRIX_BAND_PART_H_
#define MINDSPORE_CORE_OPS_MATRIX_BAND_PART_H_
#include <map>
#include <memory>
#include <string>
#include <vector>
#include "mindapi/base/types.h"
#include "ops/base_operator.h"

namespace mindspore {
namespace ops {
constexpr auto kNameMatrixBandPart = "MatrixBandPart";

template <typename T>
std::vector<T> GetExpandedShape(const std::vector<T> &shape, const size_t expended_rank) {
  std::vector<T> expanded_shape;
  expanded_shape.resize(expended_rank, 1);
  for (size_t i = 0; i < shape.size(); i++) {
    expanded_shape[i] = shape[i];
  }
  return expanded_shape;
}

class OPS_API MatrixBandPart : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(MatrixBandPart);
  MatrixBandPart() : BaseOperator(kNameMatrixBandPart) { InitIOName({"x"}, {"y"}); }
  void Init() const {}
};
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_MATRIX_BAND_PART_H_
