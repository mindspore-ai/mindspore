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
#include <map>
#include <functional>
#include <algorithm>
#include <set>
#include "frontend/parallel/tensor_layout/layout_utils.h"

namespace mindspore::parallel {
int64_t GetTensorSize(const Shape &shape) {
  int64_t size = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int64_t>());
  return std::abs(size);
}

bool RecordDimsChange(size_t key, int64_t value, std::map<size_t, int64_t> *memo, bool update) {
  auto iter = memo->find(key);
  if (!update && iter != memo->end()) {
    return false;
  }
  if (update && memo->find(key) != memo->end()) {
    (*memo)[key] = value;
    return true;
  }
  memo->insert({key, value});
  return true;
}

Status GetFactors(const TensorLayout &layout, Array *array) {
  std::vector<int64_t> factors(layout.tensor_shape().array().size());
  for (uint64_t i = 0; i < layout.tensor_map().GetDimSize(); i++) {
    if (layout.tensor_map().GetDimByIdx(i) != -1) {
      int64_t divisor = layout.GetSliceNumByTensorDimensionIndex(i);
      if (divisor == 0) {
        MS_LOG(ERROR) << "GetSliceNumByTensorDimensionIndex is 0";
        return Status::FAILED;
      }
      factors[i] = divisor;
    } else {
      factors[i] = 1;
    }
  }
  array->Init(factors);
  return Status::SUCCESS;
}

bool UseStrictMode(const Shape &from_shape, const Shape &to_shape) {
  if (from_shape.size() == to_shape.size()) {
    for (size_t i = 0; i < from_shape.size(); ++i) {
      if (from_shape[i] != to_shape[i]) {
        return false;
      }
    }
    return true;
  }
  return false;
}

int64_t GetLeastFactorWithoutConstDims(const Shape &to_shape, const Array &to_factors) {
  Shape new_to_factors;
  for (size_t i = 0; i < to_shape.size(); i++) {
    if (to_shape.at(i) == -1 && to_factors.GetDimByIdx(i) != -1) {
      new_to_factors.emplace_back(to_factors.GetDimByIdx(i));
    }
  }
  if (new_to_factors.empty()) {
    return 1;
  }
  int64_t factor = std::accumulate(new_to_factors.begin(), new_to_factors.end(), 1, std::multiplies<int64_t>());
  return factor;
}

void InitShapeVec(const Shape &src_shape, Shape *tgt_shape) {
  size_t src_size = src_shape.size();
  size_t tgt_size = tgt_shape->size();
  size_t copy_size = std::min(src_size, tgt_size);
  std::copy(src_shape.begin(), src_shape.begin() + copy_size, tgt_shape->begin());
  if (tgt_size >= src_size) {
    return;
  }
  for (size_t i = tgt_size; i < src_size; ++i) {
    (*tgt_shape)[tgt_size - 1] *= src_shape[i];
  }
  if (GetTensorSize(src_shape) != GetTensorSize(*tgt_shape)) {
    MS_LOG(ERROR) << "Failed to copy init tensor.";
  }
}

bool CheckDynamicShape(const TensorLayout &from_in, const TensorLayout &to_in) {
  Shape from_shape = from_in.tensor_shape().array();
  Shape to_shape = to_in.tensor_shape().array();
  auto func = [](const Shape &shape) -> bool { return std::find(shape.begin(), shape.end(), -1) != shape.end(); };
  return func(from_shape) && func(to_shape);
}

void UnifyFromAndToShape(Shape *new_from_shape, Shape *new_to_shape, const TensorLayout &from_in,
                         const TensorLayout &to_in, ReplacementMemo *from_dims_replace_memo) {
  Shape original_from_shape = from_in.tensor_shape().array();
  Shape original_to_shape = to_in.tensor_shape().array();
  for (size_t i = 0; i < new_from_shape->size(); ++i) {
    if (original_from_shape[i] == -1) {
      if (i < new_to_shape->size() && new_from_shape->at(i) < new_to_shape->at(i) &&
          new_to_shape->at(i) % new_from_shape->at(i) == 0) {
        int64_t scalar = new_to_shape->at(i) / new_from_shape->at(i);
        for (size_t j = i + 1; j < new_from_shape->size(); ++j) {
          if (original_from_shape[j] != -1) {
            continue;
          }
          if (new_from_shape->at(j) > scalar && new_from_shape->at(j) % scalar == 0) {
            (*new_from_shape)[j] = new_from_shape->at(j) / scalar;
            (*new_from_shape)[i] = new_from_shape->at(i) * scalar;
            RecordDimsChange(i, new_from_shape->at(i), from_dims_replace_memo, true);
            RecordDimsChange(j, new_from_shape->at(j), from_dims_replace_memo, true);
            break;
          }
        }
      }
    }
  }
}

void IntroduceConstraints(const Shape &expected_tgt_shape, Shape *tgt_shape) {
  // ([80,7,768,16], [-1,-1,3072,-1]) -> [80,7,3072,4]
  // ([20480,768,1,1], [-1, 1024, 12, 64]) -> [20, 1024, 12, 64]
  // Record fix dim index.
  std::set<size_t> index;
  std::vector<size_t> dynamic_dim_index;
  for (size_t i = 0; i < expected_tgt_shape.size(); ++i) {
    if (expected_tgt_shape[i] == -1) {
      dynamic_dim_index.emplace_back(i);
    }
  }
  for (size_t i = 0; i < expected_tgt_shape.size(); ++i) {
    if (expected_tgt_shape[i] == -1) {
      continue;
    }
    if (tgt_shape->at(i) == expected_tgt_shape[i]) {
      index.insert(i);
      continue;
    }
    if (tgt_shape->at(i) > expected_tgt_shape[i]) {
      if (tgt_shape->at(i) % expected_tgt_shape[i] == 0) {
        int64_t f = tgt_shape->at(i) / expected_tgt_shape[i];
        for (int32_t j = static_cast<int32_t>(tgt_shape->size()) - 1; j >= 0; --j) {
          if (j == static_cast<int32_t>(i) || index.find(j) != index.end()) {
            continue;
          }
          (*tgt_shape)[j] *= f;
          break;
        }
        (*tgt_shape)[i] = expected_tgt_shape[i];
      } else {
        MS_LOG(ERROR) << "Can't be divided.";
      }
    } else {
      if (expected_tgt_shape[i] % tgt_shape->at(i) == 0) {
        int64_t f = expected_tgt_shape[i] / tgt_shape->at(i);
        for (int32_t j = static_cast<int32_t>(tgt_shape->size()) - 1; j >= 0; --j) {
          if (j == static_cast<int32_t>(i) || index.find(j) != index.end()) {
            continue;
          }
          int64_t divider = std::gcd(f, tgt_shape->at(j));
          (*tgt_shape)[j] /= divider;
          f /= divider;
          if (f == 1) {
            break;
          }
        }
        if (f != 1) {
          MS_LOG(ERROR) << "Can't merge shape.";
        }
        (*tgt_shape)[i] = expected_tgt_shape[i];
      } else {
        int64_t target_dim = expected_tgt_shape[i];  // 1024
        for (int32_t j = static_cast<int32_t>(tgt_shape->size()) - 1; j >= 0; --j) {
          if (index.find(j) != index.end()) {
            continue;
          }
          int64_t divider = std::gcd(target_dim, tgt_shape->at(j));
          (*tgt_shape)[j] /= divider;
          target_dim /= divider;
          if (target_dim == 1) {
            break;
          }
        }
        if (target_dim != 1) {
          MS_LOG(ERROR) << "Can't be divided.";
        } else {
          // find last dyn dim on right and put tgt_shape->at(i) to it
          (*tgt_shape)[dynamic_dim_index.back()] = tgt_shape->at(dynamic_dim_index.back()) * tgt_shape->at(i);
          (*tgt_shape)[i] = expected_tgt_shape[i];
        }
      }
    }
    index.insert(i);
  }
}

bool ForwardMatching(const Shape &src_shape, const Shape &expected_tgt_shape, Shape *tgt_shape,
                     const Array &tgt_factors) {
  // Borrow the size from right dim, then borrow the size from left dim.
  // tgt_shape must be inited with value 1 and has fixed size.
  InitShapeVec(src_shape, tgt_shape);
  IntroduceConstraints(expected_tgt_shape, tgt_shape);
  int64_t tensor_size = GetTensorSize(*tgt_shape);
  size_t src_size = tgt_shape->size();
  std::set<size_t> fix_index;
  for (size_t i = 0; i < expected_tgt_shape.size(); ++i) {
    if (expected_tgt_shape[i] != -1) {
      fix_index.insert(i);
    }
  }
  for (size_t i = 0; i < tgt_shape->size(); ++i) {
    if (tgt_shape->at(i) % tgt_factors.GetDimByIdx(i) == 0) {
      tensor_size /= tgt_shape->at(i);
      continue;
    }
    // Borrow the size from right dim.
    int64_t factor = tgt_factors.GetDimByIdx(i);
    int64_t val = tgt_shape->at(i) * factor;
    if (val > tensor_size) {
      MS_LOG(DEBUG) << "Out of size when calculate index " << i;
      return false;
    }
    size_t ptr = i + 1;
    while (ptr < src_size) {
      if (fix_index.find(ptr) != fix_index.end()) {
        ++ptr;
        continue;
      }
      if (tgt_shape->at(ptr) >= factor && tgt_shape->at(ptr) % factor == 0) {
        (*tgt_shape)[ptr] /= factor;
        factor = 1;
        break;
      }
      int64_t divisor = std::gcd(tgt_shape->at(ptr), factor);
      factor /= divisor;
      (*tgt_shape)[ptr] /= divisor;
      ++ptr;
    }
    if (factor != 1) {
      MS_LOG(DEBUG) << "Out of size when calculate index " << i << ". Can't borrow dim from right.";
      return false;
    }
    (*tgt_shape)[i] = val;
    tensor_size /= val;
  }
  if (tensor_size != 1) {
    MS_LOG(ERROR) << "Failed to forward matching.";
    return false;
  }
  return true;
}

bool BackwardMatching(const Shape &expected_tgt_shape, Shape *tgt_shape, const Array &tgt_factors) {
  // Borrow the size from right dim.
  // Then borrow the size from left dim.
  int64_t ori_tensor_size = GetTensorSize(*tgt_shape);
  int64_t dst_size = SizeToLong(tgt_shape->size());
  std::set<size_t> fix_index;
  for (size_t i = 0; i < expected_tgt_shape.size(); ++i) {
    if (expected_tgt_shape[i] != -1) {
      fix_index.insert(i);
    }
  }
  for (int32_t i = dst_size - 1; i >= 0; --i) {
    // Borrow the size from left dim.
    int64_t factor = tgt_factors.GetDimByIdx(i);
    if (tgt_shape->at(i) % factor == 0) {
      continue;
    }
    int64_t to_be_filled_dim = tgt_shape->at(i) * factor;
    int32_t ptr = i - 1;
    while (ptr >= 0) {
      if (fix_index.find(ptr) != fix_index.end()) {
        --ptr;
        continue;
      }
      if (tgt_shape->at(ptr) % factor == 0 && tgt_shape->at(ptr) / factor % tgt_factors.GetDimByIdx(ptr) == 0) {
        (*tgt_shape)[ptr] /= factor;
        factor = 1;
        break;
      }
      int64_t divisor = std::gcd(tgt_shape->at(ptr), factor);
      factor /= divisor;
      (*tgt_shape)[ptr] /= divisor;
      --ptr;
    }
    if (factor != 1) {
      MS_LOG(ERROR) << "Can't borrow factor from left.";
      return false;
    }
    (*tgt_shape)[i] = to_be_filled_dim;
  }
  if (ori_tensor_size != GetTensorSize(*tgt_shape)) {
    MS_LOG(ERROR) << "After backward matching, tensor size is not equal.";
    return false;
  }
  return true;
}

bool SolveCombination(const Shape &src_shape_arr, size_t src_index,
                      const std::vector<std::vector<int64_t>> &enum_numbers, size_t offset, int64_t target,
                      std::vector<int64_t> *candidates_values) {
  bool is_last = (enum_numbers.size() - offset - 1) == 0;
  if (src_index < src_shape_arr.size()) {
    constexpr size_t MAX_DIM = 8;
    for (size_t factor = 1; factor < MAX_DIM; ++factor) {
      int64_t preferred_choose = SizeToLong(factor) * src_shape_arr[src_index];
      if (std::find(enum_numbers[offset].begin(), enum_numbers[offset].end(), preferred_choose) !=
            enum_numbers[offset].end() &&
          preferred_choose <= target && target % preferred_choose == 0) {
        (*candidates_values)[offset] = preferred_choose;
        if (!is_last && SolveCombination(src_shape_arr, src_index + 1, enum_numbers, offset + 1,
                                         target / candidates_values->at(offset), candidates_values)) {
          return true;
        }
      }
    }
  }
  for (size_t i = 0; i < enum_numbers[offset].size(); ++i) {
    if (enum_numbers[offset][i] > target) {
      break;
    }
    if (target % enum_numbers[offset][i] != 0) {
      continue;
    }
    (*candidates_values)[offset] = enum_numbers[offset][i];
    if (is_last && target / enum_numbers[offset][i] == 1) {
      return true;
    }
    if (!is_last && SolveCombination(src_shape_arr, src_index, enum_numbers, offset + 1,
                                     target / enum_numbers[offset][i], candidates_values)) {
      return true;
    }
  }
  return false;
}
}  // namespace mindspore::parallel
