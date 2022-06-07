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
#include <vector>
#include <string>
#include <memory>
#include "ops/base_operator.h"
#include "mindapi/base/types.h"

namespace mindspore {
namespace ops {
constexpr auto kNameMatrixBandPart = "MatrixBandPart";
constexpr int64_t kXMinShapeSize = 2;

template <typename T>
std::vector<T> GetExpandedShape(const std::vector<T> &shape) {
  if (shape.size() == 0) {
    return {1, 1};
  }
  size_t expanded_dim_num = 0;
  size_t visit_count = 0;
  for (auto it = shape.end() - 1; it >= shape.begin(); it--) {
    visit_count++;
    if (*it != 1 && visit_count == 1) {
      expanded_dim_num += kXMinShapeSize;
      break;
    }
    if (*it != 1) {
      expanded_dim_num++;
    }
    if (it == shape.begin() || visit_count == kXMinShapeSize) {
      break;
    }
  }
  if (shape.size() < kXMinShapeSize && expanded_dim_num < kXMinShapeSize) {
    expanded_dim_num++;
  }
  auto expanded_shape = shape;
  for (size_t i = 0; i < expanded_dim_num; ++i) {
    expanded_shape.emplace_back(1);
  }
  return expanded_shape;
}

class MIND_API MatrixBandPart : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(MatrixBandPart);
  MatrixBandPart() : BaseOperator(kNameMatrixBandPart) { InitIOName({"x"}, {"y"}); }
  void Init() {}
};
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_MATRIX_BAND_PART_H_
