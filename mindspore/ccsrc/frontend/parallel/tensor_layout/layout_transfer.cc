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
#include <functional>
#include <algorithm>
#include <set>
#include <vector>
#include "frontend/parallel/tensor_layout/layout_transfer.h"
#include "frontend/parallel/status.h"
#include "frontend/parallel/tensor_layout/prime_generator.h"

namespace mindspore {
namespace parallel {
std::string LayoutTransfer::ToString() const {
  std::ostringstream buffer;
  buffer << std::endl << std::string("from_in_ tensor layout:" + from_in_.ToString());
  buffer << std::endl << std::string("to_in_ tensor layout:" + to_in_.ToString());
  return buffer.str();
}

LayoutTransfer::~LayoutTransfer() = default;

bool LayoutTransfer::IsDynamicShape() const { return this->is_dynamic_shape_; }

bool LayoutTransfer::IsAssembledStaticShape() const { return this->assembled_static_shape_; }

ReplacementMemo LayoutTransfer::FromLayoutDimsReplacementMemo() const { return this->from_dims_replace_memo_; }

ReplacementMemo LayoutTransfer::ToLayoutDimsReplacementMemo() const { return this->to_dims_replace_memo_; }

bool CheckDynamicShape(const TensorLayout &from_in, const TensorLayout &to_in) {
  Shape from_shape = from_in.tensor_shape().array();
  Shape to_shape = to_in.tensor_shape().array();
  auto func = [](const Shape &shape) -> bool { return std::find(shape.begin(), shape.end(), -1) != shape.end(); };
  return func(from_shape) && func(to_shape);
}

int64_t GetTensorSize(const Shape &shape) {
  int64_t size = 1;
  size = std::accumulate(shape.begin(), shape.end(), size, std::multiplies<int64_t>());
  return std::abs(size);
}

bool RecordDimsChange(size_t key, int64_t value, std::map<size_t, int64_t> *memo) {
  auto iter = memo->find(key);
  if (iter != memo->end()) {
    return false;
  }
  memo->insert({key, value});
  return true;
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

static std::vector<int64_t> EnumerateArray(int64_t base_n, size_t length = 100) {
  static std::map<int64_t, std::vector<int64_t>> enum_numbers;
  if (enum_numbers.find(base_n) != enum_numbers.end()) {
    return enum_numbers.at(base_n);
  }
  std::vector<int64_t> array(length);
  for (size_t i = 1; i < length + 1; ++i) {
    array[i - 1] = base_n * i;
  }
  return array;
}

bool SolveCombination(const Shape &src_shape_arr, size_t src_index,
                      const std::vector<std::vector<int64_t>> &enum_numbers, size_t offset, int64_t target,
                      std::vector<int64_t> *candidates_values) {
  bool is_last = (enum_numbers.size() - offset - 1) == 0;
  if (src_index < src_shape_arr.size()) {
    for (size_t factor = 1; factor < 8; ++factor) {
      int64_t preferred_choose = factor * src_shape_arr[src_index];
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
        for (int32_t j = tgt_shape->size() - 1; j >= 0; --j) {
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
        for (int32_t j = tgt_shape->size() - 1; j >= 0; --j) {
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
        for (int32_t j = tgt_shape->size() - 1; j >= 0; --j) {
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
  int64_t dst_size = tgt_shape->size();
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

Status LayoutTransfer::CalculateFromTensorShape(Shape *from_shape, const Array &from_factors, const Shape &to_shape,
                                                const Array &to_factors) {
  if (from_shape->size() != from_factors.GetDimSize() || to_shape.size() != to_factors.GetDimSize()) {
    MS_LOG(ERROR) << "Shape size is not equal to factor size.";
    return Status::FAILED;
  }
  int64_t to_layout_added_factor = GetLeastFactorWithoutConstDims(to_shape, to_factors);
  int64_t to_layout_const_size = GetTensorSize(to_shape);
  int64_t from_layout_const_size = GetTensorSize(*from_shape);
  if (to_layout_const_size > from_layout_const_size && to_layout_const_size % from_layout_const_size == 0) {
    to_layout_added_factor *= (to_layout_const_size / from_layout_const_size);
  }
  MS_LOG(DEBUG) << "from_shape=" << (*from_shape) << ", from_factors=" << from_factors.array()
                << ", to_shape=" << to_shape << ", to_factors=" << to_factors.array()
                << ", to_layout_added_factor=" << to_layout_added_factor << std::endl;
  std::vector<int64_t> known_dims;
  (void)std::copy_if(from_shape->begin(), from_shape->end(), std::back_inserter(known_dims),
                     [](int64_t dim) -> bool { return dim != -1; });
  for (size_t i = 0; i < from_shape->size(); ++i) {
    if (from_shape->at(i) != -1) {
      continue;
    }
    int64_t prime_num = PrimeGenerator::GetInstance()->GetCoprimeNum(known_dims);
    if (prime_num == -1) {
      return Status::FAILED;
    }
    (*from_shape)[i] = prime_num * from_factors.GetDimByIdx(i);
    if (to_layout_added_factor > 0) {
      (*from_shape)[i] *= to_layout_added_factor;
      to_layout_added_factor = -1;
    }
    known_dims.emplace_back(from_shape->at(i));
    if (!RecordDimsChange(i, from_shape->at(i), &this->from_dims_replace_memo_)) {
      MS_LOG(ERROR) << "Index " << i << " conflicts.";
      return Status::FAILED;
    }
  }
  return Status::SUCCESS;
}

Status LayoutTransfer::CalculateToTensorShapeUsingEnumeration(const Shape &from_tsr_shape, Shape *to_tsr_shape,
                                                              const Array &factors) {
  int64_t src_element_size = GetTensorSize(from_tsr_shape);
  int64_t dst_element_size = GetTensorSize(*to_tsr_shape);
  if (src_element_size % dst_element_size != 0) {
    MS_LOG(ERROR) << "Calculate to tensor shape failed. Tensor shape size is not matched.";
    return Status::FAILED;
  }
  const int64_t dyn_dim_val = -1;
  int64_t dyn_axis_cnt = std::count(to_tsr_shape->begin(), to_tsr_shape->end(), dyn_dim_val);
  int64_t left_size = src_element_size / dst_element_size;

  if (dyn_axis_cnt == 0) {
    if (left_size != 1) {
      MS_LOG(ERROR) << "Calculate to tensor shape failed. Tensor shape size is not matched.";
      return Status::FAILED;
    }
    return Status::SUCCESS;
  }

  if (dyn_axis_cnt == 1) {
    /**
     * Case1:
     * from: c1, -1(32), c3, c4; to: c1/2, -1(32)*c3, c4
     */
    auto iter = std::find(to_tsr_shape->begin(), to_tsr_shape->end(), dyn_dim_val);
    size_t index = iter - to_tsr_shape->begin();
    if (left_size % factors.GetDimByIdx(index) != 0) {
      MS_LOG(ERROR) << "Generate static shape failed, the shape cannot be divided by factor. dim=" << left_size
                    << ", factor=" << factors.GetDimByIdx(index);
      return Status::FAILED;
    }
    (*iter) = left_size;
    if (!RecordDimsChange(index, left_size, &this->to_dims_replace_memo_)) {
      MS_LOG(ERROR) << "Index " << iter - to_tsr_shape->begin() << " conflicts.";
      return Status::FAILED;
    }
    return Status::SUCCESS;
  } else {
    /**
     * Case2:
     * from: -1(16), c1, c2; to: -1(2), c1*c2/2, 2*-1(8)
     * Solution:
     * -1(16), c1*c2/2, 2
     *      A,       B, c1*c2/2, 2
     *      A, c1*c2/2, 2* B
     *
     * A*B=3*16 && A%2=0 && B%8=0
     */
    std::vector<std::vector<int64_t>> enum_numbers;
    for (size_t i = 0; i < to_tsr_shape->size(); ++i) {
      if (to_tsr_shape->at(i) == -1) {
        std::vector<int64_t> array = EnumerateArray(factors.GetDimByIdx(i));
        enum_numbers.emplace_back(array);
      }
    }
    std::vector<int64_t> candidates(enum_numbers.size());
    if (!SolveCombination(from_tsr_shape, 0, enum_numbers, 0, left_size, &candidates)) {
      MS_LOG(ERROR) << "Not supported for now.";
      return Status::FAILED;
    }
    size_t cnt = 0;
    for (size_t i = 0; i < to_tsr_shape->size(); ++i) {
      if (to_tsr_shape->at(i) == -1) {
        (*to_tsr_shape)[i] = candidates[cnt++];
        if (!RecordDimsChange(i, to_tsr_shape->at(i), &this->to_dims_replace_memo_)) {
          MS_LOG(ERROR) << "Index " << i << " conflicts.";
          return Status::FAILED;
        }
      }
    }
    return Status::SUCCESS;
  }
}

Status LayoutTransfer::CalculateToTensorShape(const Shape &from_shape, const Shape &origin_to_shape,
                                              const Array &to_in_factors, Shape *to_shape) {
  // Use forward and backward matching first, if failed, turn to enumeration.
  // FIXME: record to_layout change
  bool flag_forward_match = ForwardMatching(from_shape, origin_to_shape, to_shape, to_in_factors);
  if (!flag_forward_match && !BackwardMatching(origin_to_shape, to_shape, to_in_factors)) {
    MS_LOG(DEBUG) << "Backward matching failed.";
    if (CalculateToTensorShapeUsingEnumeration(from_shape, to_shape, to_in_factors) != Status::SUCCESS) {
      MS_LOG(ERROR) << "Calculate to tensor shape failed trying to use enumeration method.";
      return Status::FAILED;
    }
  }
  return Status::SUCCESS;
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

Status LayoutTransfer::AssembleStaticTensorShape(const TensorLayout &from_in, const TensorLayout &to_in,
                                                 TensorLayout *new_from_layout, TensorLayout *new_to_layout) {
  Shape new_from_shape(from_in.tensor_shape().array());
  Shape original_to_shape = to_in.tensor_shape().array();
  Array from_in_factors;
  if (GetFactors(from_in, &from_in_factors) != Status::SUCCESS) {
    MS_LOG(ERROR) << "Get from_in factors failed.";
    return Status::FAILED;
  }
  Array to_in_factors;
  if (GetFactors(to_in, &to_in_factors) != Status::SUCCESS) {
    MS_LOG(ERROR) << "Get to_in factors failed.";
    return Status::FAILED;
  }
  if (CalculateFromTensorShape(&new_from_shape, from_in_factors, original_to_shape, to_in_factors) != Status::SUCCESS) {
    MS_LOG(ERROR) << "Failed to generate static shape for from_tensor layout: " << from_in.ToString();
    return Status::FAILED;
  }
  Shape new_to_shape(to_in_factors.GetDimSize(), 1);
  if (CalculateToTensorShape(new_from_shape, original_to_shape, to_in_factors, &new_to_shape)) {
    MS_LOG(ERROR) << "Failed to generate static shape for to_tensor layout: " << to_in.ToString() << std::endl
                  << "from_in layout: " << from_in.ToString() << std::endl
                  << "Already generate from_in shape: " << new_from_shape;
    return Status::FAILED;
  }
  size_t size = std::min(new_from_shape.size(), new_to_shape.size());
  int64_t acc_scalar = 1;
  for (size_t i = 0; i < size; ++i) {
    if (new_from_shape.at(i) > new_to_shape.at(i) && new_from_shape.at(i) % new_to_shape.at(i) == 0) {
      int64_t scalar = new_from_shape.at(i) / new_to_shape.at(i);
      new_to_shape[i] = new_to_shape[i] * scalar;
      acc_scalar *= scalar;
    }
  }
  int64_t last_dyn_dim = -1;
  for (size_t i = 0; i < from_in.tensor_shape().array().size(); ++i) {
    if (from_in.tensor_shape().array()[i] == -1) {
      last_dyn_dim = static_cast<int64_t>(i);
    }
  }
  if (last_dyn_dim != -1) {
    new_from_shape[static_cast<size_t>(last_dyn_dim)] *= acc_scalar;
  }

  if (new_from_layout->InitFromVector(from_in.device_arrangement().array(), from_in.tensor_map().array(),
                                      new_from_shape) != Status::SUCCESS) {
    MS_LOG(ERROR) << "Failed to init new from_tensor layout.";
    return Status::FAILED;
  }
  MS_LOG(DEBUG) << "Init new_from_tensor layout, origin:" << from_in.ToString()
                << ", new:" << new_from_layout->ToString();

  if (new_to_layout->InitFromVector(to_in.device_arrangement().array(), to_in.tensor_map().array(), new_to_shape) !=
      Status::SUCCESS) {
    MS_LOG(ERROR) << "Failed to init new to_tensor layout.";
    return Status::FAILED;
  }
  MS_LOG(DEBUG) << "Init new_to_layout layout, origin:" << to_in.ToString() << ", new:" << new_to_layout->ToString();

  return Status::SUCCESS;
}

Status LayoutTransfer::RollbackToDynamicShape() {
  if (!this->IsAssembledStaticShape()) {
    return Status::SUCCESS;
  }
  for (auto &iter : this->from_dims_replace_memo_) {
    MS_LOG(DEBUG) << "from index=" << iter.first << ", value=" << iter.second << std::endl;
  }
  for (auto &iter : this->to_dims_replace_memo_) {
    MS_LOG(DEBUG) << "to index=" << iter.first << ", value=" << iter.second << std::endl;
  }
  this->from_in_ = this->origin_from_in_;
  this->to_in_ = this->origin_to_in_;
  MS_LOG(DEBUG) << "RollbackToDynamicShape: from_in_=" << this->from_in_.ToString() << std::endl
                << "to_in_=" << this->to_in_.ToString() << std::endl;
  return Status::SUCCESS;
}

Status LayoutTransfer::Init(const TensorLayout &from_in, const TensorLayout &to_in, bool keep_state) {
  // Modify dynamic shape to static shape.
  this->assembled_static_shape_ = keep_state && this->assembled_static_shape_;
  is_dynamic_shape_ = CheckDynamicShape(from_in, to_in);
  this->origin_from_in_ = from_in;
  this->origin_to_in_ = to_in;
  if (is_dynamic_shape_) {
    MS_LOG(DEBUG) << "LayoutTransfer inited with dynamic shape.";
    Status ret = this->AssembleStaticTensorShape(from_in, to_in, &this->from_in_, &this->to_in_);
    if (ret != Status::SUCCESS) {
      return ret;
    }
    this->assembled_static_shape_ = true;
  } else {
    MS_LOG(DEBUG) << "LayoutTransfer inited with static shape.";
    from_in_ = from_in;
    to_in_ = to_in;
  }
  MS_LOG(DEBUG) << "LayoutTransfer init finish: " << this->ToString();
  Status status = CheckValidTransfer();
  return status;
}
}  // namespace parallel
}  // namespace mindspore
