/**
 * This is the C++ adaptation and derivative work of Myia (https://github.com/mila-iqia/myia/).
 *
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

#include "abstract/dshape.h"
#include "utils/shape_utils.h"
#include "abstract/symbolic_shape/int_symbol.h"

namespace mindspore {
namespace abstract {
namespace {
void ShapeVectorToStr(const std::vector<int64_t> &shp, std::ostringstream &buffer) {
  bool f_begin = true;
  buffer << "(";
  for (auto &x : shp) {
    if (!f_begin) {
      buffer << ", ";
    } else {
      f_begin = false;
    }
    buffer << x;
  }
  buffer << ")";
}
}  // namespace

// used for print BaseShape content
std::ostream &operator<<(std::ostream &os, const BaseShape &bs) {
  os << bs.ToString();
  return os;
}

std::ostream &operator<<(std::ostream &os, const std::shared_ptr<BaseShape> bs) {
  MS_EXCEPTION_IF_NULL(bs);
  os << bs->ToString();
  return os;
}

bool BaseShape::operator==(const BaseShape &other) const { return tid() == other.tid(); }

bool BaseShape::operator!=(const BaseShape &other) const { return !(*this == other); }

std::string TensorShape::ToString() const {
  std::ostringstream buffer;
  ShapeVectorToStr(shape_, buffer);
  return buffer.str();
}

void TensorShape::ToStringWithBuffer(std::ostringstream &buffer) const { ShapeVectorToStr(shape_, buffer); }

std::string TensorShape::DumpText() const {
  std::ostringstream buffer;
  buffer << "[";
  for (size_t i = 0; i < shape_.size(); i++) {
    buffer << (i > 0 ? ", " : "") << shape_[i];
  }
  buffer << "]";
  return buffer.str();
}

bool TensorShape::IsDynamic() const { return mindspore::IsDynamic(shape_); }

bool TensorShape::operator==(const BaseShape &other) const {
  if (tid() != other.tid()) {
    return false;
  }
  const TensorShape &other_shape = static_cast<const TensorShape &>(other);
  if (shape_ != other_shape.shape_) {
    return false;
  }
  return true;
}

void TensorShape::Broaden() {
  for (size_t i = 0; i < shape_.size(); i++) {
    shape_[i] = kShapeDimAny;
  }
}

ListSymbolPtr TensorShape::BuildSymbolicShape() const {
  if (mindspore::IsDynamicRank(shape_)) {
    return ListSymbol::Make();
  }
  symshape::SymbolPtrList symlist(shape_.size());
  (void)std::transform(shape_.begin(), shape_.end(), symlist.begin(), [](int64_t s) {
    if (s == abstract::Shape::kShapeDimAny) {
      auto ints = IntSymbol::Make();
      ints->SetRangeMin(1);
      return ints;
    }
    return IntSymbol::Make(s);
  });
  return ListSymbol::Make(std::move(symlist));
}

bool DynamicSequenceShape::IsDynamic() const { return true; }

bool DynamicSequenceShape::IsDimZero() const {
  if (element_shape_ == nullptr) {
    return false;
  }
  return element_shape_->IsDimZero();
}

bool DynamicSequenceShape::IsDimUnknown() const {
  if (element_shape_ == nullptr) {
    return false;
  }
  return element_shape_->IsDimUnknown();
}

size_t DynamicSequenceShape::hash() const {
  auto hash_code = static_cast<std::size_t>(tid());
  hash_code = hash_combine(hash_code, element_shape_->hash());
  return hash_code;
}

bool DynamicSequenceShape::operator==(const BaseShape &other) const {
  if (!other.isa<DynamicSequenceShape>()) {
    return false;
  }
  const auto &other_shape = dynamic_cast<const DynamicSequenceShape &>(other);
  if (element_shape_ == nullptr && other_shape.element_shape_ == nullptr) {
    return true;
  }
  if (element_shape_ == nullptr || other_shape.element_shape_ == nullptr) {
    return false;
  }
  return *element_shape_ == *other_shape.element_shape_;
}

std::string SequenceShape::ToString() const {
  std::ostringstream buffer;
  bool f_begin = true;
  for (const auto &p_shp : p_shapes_) {
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

void SequenceShape::ToStringWithBuffer(std::ostringstream &buffer) const {
  bool f_begin = true;
  for (const auto &p_shp : p_shapes_) {
    if (!f_begin) {
      buffer << ", ";
    } else {
      f_begin = false;
    }
    MS_EXCEPTION_IF_NULL(p_shp);
    p_shp->ToStringWithBuffer(buffer);
  }
}

BaseShapePtrList SequenceShape::ElementsClone() const {
  BaseShapePtrList ele_list;
  for (const auto &p_shp : p_shapes_) {
    MS_EXCEPTION_IF_NULL(p_shp);
    ele_list.push_back(p_shp->Clone());
  }
  return ele_list;
}

ListSymbolPtr SequenceShape::BuildSymbolicShape() const {
  symshape::SymbolPtrList symlist(p_shapes_.size());
  (void)std::transform(p_shapes_.begin(), p_shapes_.end(), symlist.begin(),
                       [](const BaseShapePtr &s) { return s->BuildSymbolicShape(); });
  return ListSymbol::Make(std::move(symlist));
}

template bool SequenceShape::SequenceEqual<TupleShape>(const BaseShape &) const;
template bool SequenceShape::SequenceEqual<ListShape>(const BaseShape &) const;
}  // namespace abstract
}  // namespace mindspore
