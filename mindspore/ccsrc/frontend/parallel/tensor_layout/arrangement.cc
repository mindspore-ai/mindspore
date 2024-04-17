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

#include "frontend/parallel/tensor_layout/arrangement.h"

#include <algorithm>
#include <utility>

#include "frontend/parallel/status.h"
#include "frontend/parallel/tensor_layout/shape_util.h"
#include "include/common/utils/convert_utils.h"
#include "utils/log_adapter.h"

namespace mindspore {
namespace parallel {
Status Arrangement::Init(const Shape &array) {
  Status status = Array::Init(array);
  if (status != Status::SUCCESS) {
    return Status::FAILED;
  }
  if (!IsValidArrangement()) {
    MS_LOG(ERROR) << "invalid arrangement " << this->ToString();
    return Status::FAILED;
  }
  ComputeSize();
  return Status::SUCCESS;
}

Status Arrangement::UpdateTensorShape(size_t index, int64_t update_value) {
  if (index >= this->array_.size()) {
    return Status::FAILED;
  }
  this->array_[index] = update_value;
  return Status::SUCCESS;
}

bool Arrangement::IsValidArrangement() {
  return !std::any_of(array_.begin(), array_.end(), [](int64_t value) { return value <= 0 && value != -1; });
}

void Arrangement::ComputeSize() {
  size_ = 1;
  for (auto &value : array_) {
    size_ *= value;
  }
}

/*
 * if GetDimSize() = 0, return []
 * if value <= array_[0], return [value]
 * if array_[0] < value <= size_[i], return [shape[0], shape[1], ..., shape[i-1], value/size_[i-1]],
 * where size_[i-1] = shape[0] * shape[1] * ... * shape[i-1],
 * if value > size_, return []
 */
Shape Arrangement::GetFrontElementByValue(int64_t value) const {
  Shape out;
  if (GetDimSize() == 0) {
    return out;
  }
  if (value <= size_) {
    int64_t size = 1;
    size_t shape_list_idx = 0;
    while (size < value) {
      size *= array_[shape_list_idx];
      if (size <= value) {
        out.push_back(array_[shape_list_idx]);
      } else {
        if (size == 0) {
          MS_LOG(ERROR) << "The size is 0";
          out.clear();
          return out;
        }
        out.push_back(value * array_[shape_list_idx] / size);
      }
      shape_list_idx++;
    }
  }
  return out;
}

std::shared_ptr<Arrangement> Arrangement::GetExpandedShapeByExpandListRemoveLeft(
  const std::vector<Arrangement> &expand_list) const {
  if (expand_list.size() != GetDimSize()) {
    return nullptr;
  }
  Shape new_shape;
  for (size_t i = 0; i < expand_list.size(); i++) {
    Shape expand_shape = expand_list[i].GetFrontElementByValue(GetDimByIdx(i));
    if (expand_shape.empty()) {
      new_shape.push_back(GetDimByIdx(i));
    } else {
      (void)new_shape.insert(new_shape.cend(), expand_shape.cbegin(), expand_shape.cend());
    }
  }
  Arrangement arrangement_new;
  (void)arrangement_new.Init(new_shape);
  return std::make_shared<Arrangement>(arrangement_new);
}

/*
 *  example:
 *    expand_shape = [4, 2, 2, 2]
 *    array_ = [8, 4],
 *    arrangement_list = [[4, 2], [2, 2]]
 */
std::shared_ptr<std::vector<Arrangement>> Arrangement::GetExpandShapeList(const Arrangement &expand_shape) const {
  int64_t size = 1;
  size_t ind = 0;
  std::vector<Arrangement> arrangement_list;
  Shape shape;
  for (size_t i = 0; i < expand_shape.GetDimSize(); i++) {
    size *= expand_shape.GetDimByIdx(i);
    if (size > GetDimByIdx(ind)) {
      MS_LOG(INFO) << "invalid expand_shape:" << expand_shape.array();
      return nullptr;
    } else if (size < GetDimByIdx(ind)) {
      shape.push_back(expand_shape.GetDimByIdx(i));
      continue;
    } else {
      shape.push_back(expand_shape.GetDimByIdx(i));
      Arrangement arrangement;
      (void)arrangement.Init(shape);
      arrangement_list.push_back(arrangement);
      shape.clear();
      ind++;
      size = 1;
    }
  }
  if (ind != GetDimSize()) {
    MS_LOG(INFO) << "invalid expand_shape:" << expand_shape.array();
    return nullptr;
  }
  auto arrangement_new = std::make_shared<std::vector<Arrangement>>(arrangement_list);
  return arrangement_new;
}

std::shared_ptr<std::pair<std::vector<Arrangement>, Arrangement>> Arrangement::GetExpandShapeListPair(
  const Arrangement &expand_shape) const {
  std::shared_ptr<std::vector<Arrangement>> expand_shape_list_ptr = GetExpandShapeList(expand_shape);
  if (expand_shape_list_ptr == nullptr) {
    return nullptr;
  }
  Shape expand_num_list_shape;
  (void)std::transform(expand_shape_list_ptr->begin(), expand_shape_list_ptr->end(),
                       std::back_inserter(expand_num_list_shape),
                       [](const Arrangement &arr) { return SizeToLong(arr.GetDimSize()); });
  Arrangement expand_num_list;
  Status status = expand_num_list.Init(expand_num_list_shape);
  if (status != Status::SUCCESS) {
    return nullptr;
  }
  auto out_value = std::make_pair(*expand_shape_list_ptr, expand_num_list);
  return std::make_shared<std::pair<std::vector<Arrangement>, Arrangement>>(out_value);
}

Shape Arrangement::ComputeReverseAccumulateSumInReverseOrder() const {
  Shape shape_accum;
  int64_t size = 0;
  for (auto iter = array_.end() - 1; iter >= array_.begin(); --iter) {
    shape_accum.push_back(size);
    size += *iter;
  }
  return shape_accum;
}

std::shared_ptr<Arrangement> Arrangement::GetExpandedShapeByExpandListReserveLeft(
  const std::vector<Arrangement> &expand_list) const {
  if (expand_list.size() != GetDimSize()) {
    return nullptr;
  }
  Shape new_shape;
  for (size_t i = 0; i < expand_list.size(); i++) {
    if (expand_list[i].GetDimSize() >= 1) {
      int64_t size = 1;
      for (size_t k = 0; k < expand_list[i].GetDimSize() - 1; k++) {
        new_shape.push_back(expand_list[i].GetDimByIdx(k));
        size *= expand_list[i].GetDimByIdx(k);
      }
      new_shape.push_back(GetDimByIdx(i) / size);
    } else {
      new_shape.push_back(GetDimByIdx(i));
    }
  }
  Arrangement arrangement_new;
  (void)arrangement_new.Init(new_shape);
  return std::make_shared<Arrangement>(arrangement_new);
}

std::shared_ptr<Arrangement> Arrangement::GetUnifiedShape(const Arrangement &in2) const {
  std::vector<int64_t> in1_accum;
  Status status = ShapeToAccumulateProduct(array_, &in1_accum);
  if (status != Status::SUCCESS) {
    return nullptr;
  }
  std::vector<int64_t> in2_accum;
  status = ShapeToAccumulateProduct(in2.array(), &in2_accum);
  if (status != Status::SUCCESS) {
    return nullptr;
  }
  std::vector<int64_t> out_accum;
  status = UnifyAccumulateProduct(in1_accum, in2_accum, &out_accum);
  if (status != Status::SUCCESS) {
    return nullptr;
  }
  Shape out_shape;
  status = AccumulateProductToShape(out_accum, &out_shape);
  if (status != Status::SUCCESS) {
    return nullptr;
  }
  Arrangement out;
  status = out.Init(out_shape);
  if (status != Status::SUCCESS) {
    return nullptr;
  }
  return std::make_shared<Arrangement>(out);
}

std::vector<size_t> Arrangement::GetSqueezeIdx() const {
  std::vector<size_t> out;
  for (size_t i = 0; i < GetDimSize(); i++) {
    if (GetDimByIdx(SizeToUlong(i)) == 1) {
      out.push_back(i);
    }
  }
  return out;
}

Arrangement Arrangement::GetSqueezeArrangement() const {
  Shape out_shape(array_.size());
  auto it = std::copy_if(array_.begin(), array_.end(), out_shape.begin(), [](int64_t value) { return value != 1; });
  out_shape.resize(LongToSize(std::distance(out_shape.begin(), it)));

  // if all elements are 1, out_shape = {1}
  if (out_shape.empty()) {
    MS_LOG(ERROR) << "out_shape size is 0, this may not happen under current situation";
    out_shape.push_back(1);
  }
  Arrangement out;
  (void)out.Init(out_shape);
  return out;
}
}  // namespace parallel
}  // namespace mindspore
