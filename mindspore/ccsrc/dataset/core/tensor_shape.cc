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

#include "dataset/core/tensor_shape.h"

#include <limits>

#include "common/utils.h"
#include "utils/log_adapter.h"
#include "dataset/core/constants.h"
#include "dataset/util/de_error.h"

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
  dsize_t num = 1;
  for (auto i : raw_shape_) {
    if (multi_ok(num, i)) {
      num *= i;
    } else {
      // dsize_t can wrap since it is signed int, we double check here
      MS_LOG(ERROR) << "Tensor shape larger than maximum allowed value!";
    }
  }
  return num;
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
    : raw_shape_(*GlobalContext::Instance()->int_allocator()) {
  AddListToShape(list);
}

TensorShape::TensorShape(const std::vector<dsize_t> &list) : raw_shape_(*GlobalContext::Instance()->int_allocator()) {
  AddListToShape(list);
}

TensorShape::TensorShape(const TensorShape &shape) : raw_shape_(*GlobalContext::Instance()->int_allocator()) {
  AddListToShape(shape.AsVector());
  known_ = shape.known_;  // override with the input shape in case of unknown-rank tensor shape.
}

TensorShape::TensorShape(py::list l) : raw_shape_(*GlobalContext::Instance()->int_allocator()) {
  std::vector<dsize_t> list_c;
  for (auto i : l) {
    list_c.push_back(i.cast<int>());
  }
  AddListToShape(list_c);
}

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

TensorShape::TensorShape(cv::MatSize cv_size, uint32_t type) : raw_shape_(*GlobalContext::Instance()->int_allocator()) {
  for (int i = 0; i < cv_size.dims(); i++) {
    raw_shape_.push_back(cv_size[i]);
  }
  auto channels = static_cast<uint8_t>(1 + (type >> static_cast<uint8_t>(CV_CN_SHIFT)));
  if (channels != 1) {
    raw_shape_.push_back(channels);
  }
  known_ = true;
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
  known_ = true;
  dsize_t num = 1;
  dsize_t size = 0;
  for (const auto &itr : list) {
    if (itr > 0) {
      if (num > std::numeric_limits<int64_t>::max() / itr) {
        MS_LOG(ERROR) << "Invalid shape data, overflow occurred!";
        known_ = false;
        raw_shape_.clear();
        return;
      }
      num *= itr;
    }
    if (itr < 0) {
      known_ = false;
    }
    if (itr > kDeMaxDim) {
      std::stringstream ss;
      ss << "Invalid shape data, dim (" << size << ") is larger than the maximum dim size(" << kDeMaxDim << ")!";
      MS_LOG(ERROR) << ss.str().c_str();
      known_ = false;
      raw_shape_.clear();
      return;
    }
    raw_shape_.push_back(itr);
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

py::list TensorShape::AsPyList() {
  py::list list;
  for (auto i : raw_shape_) {
    list.append(i);
  }
  return list;
}

TensorShape TensorShape::Squeeze() const {
  std::vector<dsize_t> new_shape;
  for (auto s : AsVector()) {
    if (s != 1) {
      new_shape.push_back(s);
    }
  }
  return TensorShape(new_shape);
}
}  // namespace dataset
}  // namespace mindspore
