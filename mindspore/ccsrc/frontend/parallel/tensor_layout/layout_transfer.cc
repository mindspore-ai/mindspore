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
#include "frontend/parallel/tensor_layout/layout_utils.h"

namespace mindspore {
namespace parallel {
constexpr size_t INVALID_TENSOR_RANK = 9999;

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

static std::vector<int64_t> EnumerateArray(int64_t base_n, size_t length = 100) {
  static std::map<int64_t, std::vector<int64_t>> enum_numbers;
  if (enum_numbers.find(base_n) != enum_numbers.end()) {
    return enum_numbers.at(base_n);
  }
  std::vector<int64_t> array(length);
  for (size_t i = 1; i < length + 1; ++i) {
    array[i - 1] = base_n * SizeToLong(i);
  }
  return array;
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
                << ", to_layout_added_factor=" << to_layout_added_factor;
  if (from_layout_const_size > to_layout_const_size && from_layout_const_size % to_layout_added_factor == 0) {
    from_layout_const_size /= to_layout_added_factor;
    // Existed dim in from_layout already satisfy to_layout_added_factor.
    to_layout_added_factor = -1;
  }
  MS_LOG(INFO) << "from_shape=" << (*from_shape) << ", from_factors=" << from_factors.array()
               << ", to_shape=" << to_shape << ", to_factors=" << to_factors.array()
               << ", to_layout_added_factor=" << to_layout_added_factor;
  bool strict_mode = UseStrictMode(*from_shape, to_shape);
  std::vector<int64_t> known_dims;
  (void)std::copy_if(from_shape->begin(), from_shape->end(), std::back_inserter(known_dims),
                     [](int64_t dim) -> bool { return dim != -1; });
  size_t last_dyn_dim = INVALID_TENSOR_RANK;
  auto last_dyn_dim_iter = std::find(from_shape->rbegin(), from_shape->rend(), -1);
  if (last_dyn_dim_iter != from_shape->rend()) {
    last_dyn_dim = from_shape->size() - (last_dyn_dim_iter - from_shape->rbegin()) - 1;
  }
  for (size_t i = 0; i < from_shape->size(); ++i) {
    if (from_shape->at(i) != -1) {
      continue;
    }
    int64_t prime_num = PrimeGenerator::GetInstance()->GetCoprimeNum(known_dims);
    if (prime_num == -1) {
      return Status::FAILED;
    }
    (*from_shape)[i] = prime_num * from_factors.GetDimByIdx(i);
    if (strict_mode && from_shape->at(i) % to_factors.GetDimByIdx(i) != 0) {
      (*from_shape)[i] *= to_factors.GetDimByIdx(i);
      if (to_layout_added_factor >= to_factors.GetDimByIdx(i) &&
          to_layout_added_factor % to_factors.GetDimByIdx(i) == 0) {
        to_layout_added_factor /= to_factors.GetDimByIdx(i);
      }
    }
    if (i == last_dyn_dim && to_layout_added_factor > 0) {
      if (from_shape->at(i) % to_layout_added_factor != 0) {
        (*from_shape)[i] *= to_layout_added_factor;
      }
      to_layout_added_factor = -1;
    }
    known_dims.emplace_back(from_shape->at(i));
    MS_LOG(DEBUG) << "Replace  " << i << " with value " << from_shape->at(i) << " prime " << prime_num;
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
    size_t index = static_cast<size_t>(iter - to_tsr_shape->begin());
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
  if (GetTensorSize(new_from_shape) != GetTensorSize(new_to_shape)) {
    int64_t acc_scalar = 1;
    for (size_t i = 0; i < size; ++i) {
      if (new_from_shape.at(i) > new_to_shape.at(i) && new_from_shape.at(i) % new_to_shape.at(i) == 0) {
        int64_t scalar = new_from_shape.at(i) / new_to_shape.at(i);
        new_to_shape[i] = new_to_shape[i] * scalar;
        acc_scalar *= scalar;
      }
    }
    const Shape &f_in_tensor_shape = from_in.tensor_shape().array();
    auto last_dyn_dim_iter = std::find(f_in_tensor_shape.rbegin(), f_in_tensor_shape.rend(), -1);
    if (last_dyn_dim_iter != f_in_tensor_shape.rend()) {
      size_t last_dyn_dim =
        f_in_tensor_shape.size() - static_cast<size_t>(last_dyn_dim_iter - f_in_tensor_shape.rbegin()) - 1;
      new_from_shape[static_cast<size_t>(last_dyn_dim)] *= acc_scalar;
    }
  }

  // Unify shape from begin to end.
  UnifyFromAndToShape(&new_from_shape, &new_to_shape, from_in, to_in, &this->from_dims_replace_memo_);

  MS_LOG(INFO) << "new_from_shape=" << new_from_shape << ", new_to_shape=" << new_to_shape;
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
    return Status::FAILED;
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
