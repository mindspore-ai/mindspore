/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_SHAPE_UTILS_INFO_H_
#define MINDSPORE_SHAPE_UTILS_INFO_H_

#include <algorithm>
#include <string>
#include <vector>
#include "abstract/dshape.h"
#include "utils/log_adapter.h"

namespace mindspore {
inline std::string ShapeVectorToString(const ShapeVector &shape) {
  std::string str_shape = "";
  for (auto &item : shape) {
    str_shape += std::to_string(item) + ", ";
  }
  str_shape = str_shape.length() >= 2 ? str_shape.substr(0, str_shape.length() - 2) : str_shape;
  return str_shape;
}

inline size_t SizeOf(const ShapeVector &shape) {
  size_t data_size = 1;
  for (auto dim : shape) {
    if (dim <= 0) {
      // For dynamic shape which has negative dimensions, data size should be zero.
      return 0;
    }
    if (SIZE_MAX / dim < data_size) {
      MS_EXCEPTION(ValueError) << "The product value of shape (" << ShapeVectorToString(shape)
                               << ") exceeds the maximum value of size_t: " << SIZE_MAX;
    }
    data_size *= static_cast<size_t>(dim);
  }
  return data_size;
}

inline bool IsOneElementShape(const ShapeVector &shape) {
  if (shape.empty()) {
    return true;
  } else if (shape.size() == 1 && shape[0] == 1) {
    return true;
  } else {
    return false;
  }
}

inline bool IsMactchedShapeInferValue(const ShapeVector &shape1, const ShapeVector &shape2) {
  if (IsOneElementShape(shape1) && IsOneElementShape(shape2)) {
    return true;
  }
  if (shape1 == shape2) {
    return true;
  }
  return false;
}

inline bool IsDynamicRank(const ShapeVector &shape) {
  for (size_t i = 0; i < shape.size(); ++i) {
    if (shape[i] > abstract::Shape::kShapeRankAny) {
      continue;
    }

    if (shape.size() == abstract::Shape::kDynamicRankLen) {
      return true;
    } else if (i == 1) {
      MS_LOG(DEBUG) << "Shape(" << ShapeVectorToString(shape) << ") is a valid shape for real tuple tensor.";
      return true;
    } else {
      MS_EXCEPTION(ValueError) << "Shape should have only one -2 for normal tensor,or [not -2, -2] for real tuple "
                                  "tensor, or no -2 at all, but got ("
                               << ShapeVectorToString(shape) << ").";
    }
  }

  return false;
}

inline bool IsDynamicShape(const ShapeVector &shape) {
  return std::any_of(shape.cbegin(), shape.cend(),
                     [](ShapeValueDType s) { return s == abstract::Shape::kShapeDimAny; });
}

inline bool IsDynamic(const ShapeVector &shape) {
  for (auto &s : shape) {
    if (s > abstract::Shape::kShapeDimAny) {
      continue;
    }

    if (s < abstract::Shape::kShapeRankAny) {
      MS_EXCEPTION(ValueError) << "Shape should not have values less than -2 but got (" << ShapeVectorToString(shape)
                               << ").";
    }

    return true;
  }

  return false;
}

inline bool IsShapeEmpty(const ShapeVector &shape) {
  constexpr size_t kOne = 1;
  constexpr size_t kZero = 0;
  return shape.size() == kOne && shape[0] == kZero;
}

inline bool IsShapeNone(const ShapeVector &shape) {
  return std::any_of(shape.begin(), shape.end(), [](const auto &dim) { return dim == 0; });
}

// use for the op with the constraint that output shape must be same as input shape
inline ShapeVector InferOutShapeSameAsInShape(const ShapeArray &input_shapes) {
  ShapeVector out_shape{};
  for (size_t i = 0; i < input_shapes.size(); i++) {
    auto in_shape = input_shapes[i];
    // scalar case
    if (in_shape.empty()) {
      return out_shape;
    }
    // skip to next input shape if current shape is dynamic rank
    if (IsDynamicRank(in_shape)) {
      continue;
    }
    // initialize output shape
    auto rank = in_shape.size();
    if (out_shape.empty()) {
      out_shape.resize(rank, abstract::Shape::kShapeDimAny);
    }
    if (out_shape.size() != rank) {
      MS_EXCEPTION(ValueError) << "Ranks of inputs must be all same if they are not dynamic.";
    }
    for (size_t j = 0; j < rank; j++) {
      if (out_shape[j] != abstract::Shape::kShapeDimAny && in_shape[j] != abstract::Shape::kShapeDimAny &&
          out_shape[j] != in_shape[j]) {
        MS_EXCEPTION(ValueError) << "Corresponding axis of input shapes must be same if they are not dynamic.";
      }
      if (out_shape[j] == abstract::Shape::kShapeDimAny && in_shape[j] != abstract::Shape::kShapeDimAny) {
        out_shape[j] = in_shape[j];
      }
    }
  }
  // if all input shapes are dynamic rank, return dynamic rank output
  if (out_shape.empty()) {
    return {abstract::Shape::kShapeRankAny};
  }
  return out_shape;
}

template <typename T>
std::string VectorToString(const std::vector<T> &values) {
  std::stringstream ss;
  ss << "[";
  auto size = values.size();
  for (size_t i = 0; i < size; ++i) {
    ss << values[i];
    if (i != size - 1) {
      ss << ", ";
    }
  }
  ss << "]";
  return ss.str();
}
}  // namespace mindspore

#endif  // MINDSPORE_SHAPE_UTILS_INFO_H_
