/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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
#define MAX_INTEGER_DTYPE 9223372036854775807

#include "minddata/dataset/core/tensor_shape.h"

#include <limits>

#include "utils/ms_utils.h"
#ifndef ENABLE_ANDROID
#include "utils/log_adapter.h"
#else
#include "mindspore/lite/src/common/log_adapter.h"
#endif
#include "minddata/dataset/include/constants.h"

namespace mindspore {
namespace dataset {
constexpr dsize_t TensorShape::kDimUnknown;

bool multi_ok(dsize_t x, dsize_t y) {
  dsize_t p = x * y;
  if (x == 0) {
    return true;
  }
  return p / x == y;
}

dsize_t TensorShape::NumOfElements() const {
  if (!known()) {
    return 0;
  }
  return strides_[0];
}

void TensorShape::Print(std::ostream &out) const {
  if (!known() && raw_shape_.empty()) {
    out << "<kUnknown>";
  } else {
    out << "<";
    for (auto i = 0; i < this->Rank(); i++) {
      if (raw_shape_[i] == kDimUnknown) {
        out << "*";
      } else {
        out << raw_shape_[i];
      }
      if (i != this->Rank() - 1) {
        out << ",";
      }
    }
    out << ">";
  }
}

TensorShape::TensorShape(const std::initializer_list<dsize_t> &list)
    : raw_shape_(*GlobalContext::Instance()->int_allocator()), strides_(*GlobalContext::Instance()->int_allocator()) {
  AddListToShape(list);
}

TensorShape::TensorShape(const std::vector<dsize_t> &list)
    : raw_shape_(*GlobalContext::Instance()->int_allocator()), strides_(*GlobalContext::Instance()->int_allocator()) {
  AddListToShape(list);
}

TensorShape::TensorShape(const TensorShape &shape)
    : raw_shape_(*GlobalContext::Instance()->int_allocator()), strides_(*GlobalContext::Instance()->int_allocator()) {
  AddListToShape(shape.AsVector());
  known_ = shape.known_;  // override with the input shape in case of unknown-rank tensor shape.
}

#ifdef ENABLE_PYTHON
TensorShape::TensorShape(py::list l)
    : raw_shape_(*GlobalContext::Instance()->int_allocator()), strides_(*GlobalContext::Instance()->int_allocator()) {
  std::vector<dsize_t> list_c;
  for (auto &i : l) {
    if (!i.is_none()) {
      list_c.push_back(i.cast<int>());
    } else {
      list_c.push_back(TensorShape::kDimUnknown);
    }
  }
  AddListToShape(list_c);
}
#endif

#ifndef ENABLE_ANDROID
TensorShape::TensorShape(cv::MatSize cv_size, uint32_t type)
    : raw_shape_(*GlobalContext::Instance()->int_allocator()), strides_(*GlobalContext::Instance()->int_allocator()) {
  for (int i = 0; i < cv_size.dims(); i++) {
    raw_shape_.push_back(cv_size[i]);
  }
  auto channels = static_cast<uint8_t>(1 + (type >> static_cast<uint8_t>(CV_CN_SHIFT)));
  if (channels != 1) {
    raw_shape_.push_back(channels);
  }
  known_ = true;
}
#endif

TensorShape TensorShape::CreateUnknownRankShape() {
  TensorShape s({});
  s.known_ = false;
  return s;
}

TensorShape TensorShape::InsertDim(dsize_t axis, dsize_t dim) const {
  std::vector<dsize_t> tmp = AsVector();
  (void)tmp.insert(tmp.begin() + axis, dim);
  return TensorShape(tmp);
}

std::vector<dsize_t> TensorShape::AsVector() const {
  return std::vector<dsize_t>(raw_shape_.begin(), raw_shape_.end());
}

bool TensorShape::IsValidIndex(const std::vector<dsize_t> &index) const {
  dsize_t s_rank = Rank();
  if (index.size() != s_rank) {
    return false;
  }
  for (dsize_t i = 0; i < s_rank; i++) {
    if (index[i] < 0 || raw_shape_[i] <= index[i]) {
      return false;
    }
  }
  return true;
}

template <typename T>
void TensorShape::AddListToShape(const T &list) {
  raw_shape_.resize(list.size());
  strides_.resize(list.size() + 1);
  strides_[list.size()] = 1;
  known_ = true;
  dsize_t size = 0;
  auto itr = std::rbegin(list);  // iterate over the list in reverse order
  auto s = list.size() - 1;      // to compute strides while adding dims
  for (; itr != std::rend(list); itr++, s--) {
    dsize_t dim = *itr;
    if (dim > 0) {
      if (strides_[s + 1] > std::numeric_limits<int64_t>::max() / dim) {
        MS_LOG(ERROR) << "Invalid shape data, overflow occurred!";
        known_ = false;
        raw_shape_.clear();
        return;
      }
      strides_[s] = dim * strides_[s + 1];
    }
    if (dim < 0) {
      known_ = false;
    }
    if (dim > kDeMaxDim) {
      std::stringstream ss;
      ss << "Invalid shape data, dim (" << dim << ") is larger than the maximum dim size(" << kDeMaxDim << ")!";
      MS_LOG(ERROR) << ss.str().c_str();
      known_ = false;
      raw_shape_.clear();
      return;
    }
    raw_shape_[s] = dim;
    size++;
  }
  if (size > kDeMaxRank) {
    std::stringstream ss;
    ss << "Invalid shape data, rank (" << size << ") is larger than the maximum rank size(" << kDeMaxRank << ").";
    MS_LOG(ERROR) << ss.str().c_str();
    known_ = false;
    raw_shape_.clear();
    return;
  }
}

TensorShape TensorShape::CreateUnknownShapeWithRank(dsize_t rank) {
  TensorShape s({});
  for (dsize_t i = 0; i < rank; i++) {
    s.raw_shape_.push_back(kDimUnknown);
  }
  s.known_ = false;
  return s;
}

TensorShape TensorShape::PrependDim(dsize_t dim) const {
  if (Size() == 0) {
    return TensorShape({dim});
  }
  return InsertDim(0, dim);
}

TensorShape TensorShape::AppendDim(dsize_t dim) const {
  auto vec = AsVector();
  vec.push_back(dim);
  return TensorShape(vec);
}

#ifdef ENABLE_PYTHON
py::list TensorShape::AsPyList() {
  py::list list;
  for (auto i : raw_shape_) {
    list.append(i);
  }
  return list;
}
#endif

TensorShape TensorShape::Squeeze() const {
  std::vector<dsize_t> new_shape;
  for (auto s : AsVector()) {
    if (s != 1) {
      new_shape.push_back(s);
    }
  }
  return TensorShape(new_shape);
}

std::vector<dsize_t> TensorShape::Strides() const { return std::vector<dsize_t>{strides_.begin() + 1, strides_.end()}; }

// Name: ToFlatIndex()
// Description: convert a vector style index to number, used to access memory internal use only
Status TensorShape::ToFlatIndex(const std::vector<dsize_t> &index, dsize_t *flat_index) const {
  *flat_index = 0;
  for (size_t k = 0; k < index.size(); k++) {
    *flat_index += index[k] * strides_[k + 1];  // skip the first element of strides_ which is numOfElements
  }
  CHECK_FAIL_RETURN_UNEXPECTED(*flat_index < NumOfElements(), "Not a valid index");
  return Status::OK();
}
}  // namespace dataset
}  // namespace mindspore
