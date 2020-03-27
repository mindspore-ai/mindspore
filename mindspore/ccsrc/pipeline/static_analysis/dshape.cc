/**
 * This is the C++ adaptation and derivative work of Myia (https://github.com/mila-iqia/myia/).
 *
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

#include "pipeline/static_analysis/dshape.h"

#include <exception>
#include <iostream>

#include "utils/log_adapter.h"

namespace mindspore {
namespace abstract {
// used for print BaseShape content
std::ostream& operator<<(std::ostream& os, const BaseShape& bs) {
  os << bs.ToString();
  return os;
}

std::ostream& operator<<(std::ostream& os, const std::shared_ptr<BaseShape> bs) {
  MS_EXCEPTION_IF_NULL(bs);
  os << bs->ToString();
  return os;
}

bool BaseShape::operator==(const BaseShape& other) const {
  if (tid() != other.tid()) {
    return false;
  }
  return true;
}

bool BaseShape::operator!=(const BaseShape& other) const { return !(*this == other); }

std::string Shape::ToString() const {
  std::ostringstream buffer;
  bool f_begin = true;
  buffer << "(";
  for (auto& x : shape_) {
    if (!f_begin) {
      buffer << ", ";
    } else {
      f_begin = false;
    }
    buffer << x;
  }
  buffer << ")";
  return buffer.str();
}

std::string Shape::DumpText() const {
  std::ostringstream buffer;
  buffer << "[";
  for (size_t i = 0; i < shape_.size(); i++) {
    buffer << (i > 0 ? ", " : "") << shape_[i];
  }
  buffer << "]";
  return buffer.str();
}

bool Shape::operator==(const BaseShape& other) const {
  if (tid() != other.tid()) {
    return false;
  }
  return shape_ == static_cast<const Shape&>(other).shape_;
}

const int Shape::SHP_ANY;
void Shape::Broaden() {
  for (size_t i = 0; i < shape_.size(); i++) {
    shape_[i] = SHP_ANY;
  }
}

std::string SequeueShape::ToString() const {
  std::ostringstream buffer;
  bool f_begin = true;
  for (auto p_shp : p_shapes_) {
    if (!f_begin) {
      buffer << ", ";
    } else {
      f_begin = false;
    }
    MS_EXCEPTION_IF_NULL(p_shp);
    buffer << p_shp->ToString();
  }
  return buffer.str();
}

BaseShapePtrList SequeueShape::ElementsClone() const {
  BaseShapePtrList ele_list;
  for (auto p_shp : p_shapes_) {
    MS_EXCEPTION_IF_NULL(p_shp);
    ele_list.push_back(p_shp->Clone());
  }
  return ele_list;
}

template <typename T>
bool SequeueShape::SequeueEqual(const BaseShape& other) const {
  if (tid() != other.tid()) {
    return false;
  }
  auto other_shapes = static_cast<const T&>(other).p_shapes_;
  if (other_shapes.size() != p_shapes_.size()) {
    return false;
  }
  for (unsigned int i = 0; i < p_shapes_.size(); ++i) {
    if (!(*p_shapes_[i] == *other_shapes[i])) {
      return false;
    }
  }
  return true;
}
template bool SequeueShape::SequeueEqual<TupleShape>(const BaseShape&) const;
template bool SequeueShape::SequeueEqual<ListShape>(const BaseShape&) const;

const std::shared_ptr<NoShape> kNoShape = std::make_shared<NoShape>();
}  // namespace abstract
}  // namespace mindspore
