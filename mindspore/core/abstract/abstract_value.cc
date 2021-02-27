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

#include "abstract/abstract_value.h"

#include <algorithm>

#include "utils/symbolic.h"
#include "abstract/utils.h"
#include "utils/ms_context.h"

namespace mindspore {
namespace abstract {
bool AbstractBase::operator==(const AbstractBase &other) const {
  if (tid() != other.tid()) {
    return false;
  }
  if (BuildType()->type_id() == kObjectTypeUndeterminedType &&
      other.BuildType()->type_id() == kObjectTypeUndeterminedType) {
    return true;
  }
  if (value_ == nullptr || other.value_ == nullptr) {
    MS_LOG(EXCEPTION) << "If value_ is nullptr, AbstractBase::operator== should not be called. this: "
                      << this->ToString() << ", other: " << other.ToString();
  }

  bool value_equal = false;
  if (value_ == other.value_) {
    value_equal = true;
  } else if (*value_ == *other.value_) {
    value_equal = true;
  }
  bool type_equal = false;
  if (type_ == other.type_) {
    type_equal = true;
  } else if (*type_ == *other.type_) {
    type_equal = true;
  }
  bool shape_equal = false;
  if (shape_ == other.shape_) {
    shape_equal = true;
  } else if (*shape_ == *other.shape_) {
    shape_equal = true;
  }
  return value_equal && type_equal && shape_equal;
}

ValuePtr AbstractBase::BuildValue() const {
  if (value_ == nullptr) {
    return RealBuildValue();
  }
  return value_;
}

AbstractBasePtr AbstractBase::Broaden(uint8_t config) const {
  AbstractBasePtr clone = Clone();
  auto not_broaden = config & (kBroadenTensorOnly | kBroadenParameterOnly);
  if (not_broaden == 0) {
    clone->set_value(kAnyValue);
  }
  return clone;
}

std::string AbstractBase::ToString() const {
  std::ostringstream buffer;
  std::string value = std::string("value is null");
  if (value_ != nullptr) {
    value = value_->ToString();
  }
  MS_EXCEPTION_IF_NULL(type_);
  MS_EXCEPTION_IF_NULL(shape_);
  buffer << type_name() << "("
         << "Type: " << type_->ToString() << " Value: " << value << " Shape: " << shape_->ToString() << ")";
  return buffer.str();
}

AbstractBasePtr AbstractScalar::Broaden(uint8_t config) const {
  if (MsContext::GetInstance()->get_param<bool>(MS_CTX_GRAD_FOR_SCALAR)) {
    return AbstractBase::Broaden(config);
  } else {
    return Clone();
  }
}

AbstractBasePtr AbstractScalar::Join(const AbstractBasePtr &other) {
  MS_EXCEPTION_IF_NULL(other);
  if (*this == *other) {
    return shared_from_base<AbstractBase>();
  }
  auto value_self = GetValueTrack();
  MS_EXCEPTION_IF_NULL(value_self);
  TypePtr res_type = TypeJoin(GetTypeTrack(), other->GetTypeTrack());
  if (res_type == kAnyType) {
    MS_EXCEPTION(TypeError) << "Type join failed, type1 = " << GetTypeTrack()->ToString()
                            << ", type2 = " << other->GetTypeTrack()->ToString();
  }
  ValuePtr res_value = ValueJoin(value_self, other->GetValueTrack());
  if (res_value == value_self) {
    return shared_from_base<AbstractBase>();
  }
  return std::make_shared<AbstractScalar>(res_value, res_type);
}

AbstractBasePtr AbstractType::Clone() const {
  ValuePtr value_self = GetValueTrack();
  if (value_self == nullptr || !value_self->isa<Type>()) {
    return nullptr;
  }
  TypePtr type_self = value_self->cast<TypePtr>();
  return std::make_shared<AbstractType>(type_self->Clone());
}

bool AbstractType::operator==(const AbstractBase &other) const {
  if (tid() != other.tid()) {
    return false;
  }
  // Have to compare TypePtr with value;
  ValuePtr value_self = GetValueTrack();
  ValuePtr value_other = other.GetValueTrack();
  if (value_self == nullptr || value_other == nullptr) {
    MS_LOG(EXCEPTION) << "AbstractType value should not be nullptr. this: " << this->ToString()
                      << ", other: " << other.ToString();
  }
  if (!value_self->isa<Type>() || !value_other->isa<Type>()) {
    return false;
  }
  TypePtr type_self = value_self->cast<TypePtr>();
  TypePtr type_other = value_other->cast<TypePtr>();
  bool value_equal = *type_self == *type_other;
  return value_equal;
}

std::string AbstractType::ToString() const {
  std::ostringstream buffer;
  ValuePtr value_self = GetValueTrack();
  if (value_self == nullptr) {
    buffer << "AbstractType value: nullptr";
    return buffer.str();
  }
  if (!value_self->isa<Type>()) {
    buffer << type_name() << "(Value: nullptr)";
    return buffer.str();
  }
  TypePtr type_self = value_self->cast<TypePtr>();
  MS_EXCEPTION_IF_NULL(type_self);
  buffer << type_name() << "("
         << "Value: " << type_self->ToString() << ")";
  return buffer.str();
}

std::string AbstractError::ToString() const {
  std::ostringstream buffer;
  auto value_track = GetValueTrack();
  MS_EXCEPTION_IF_NULL(value_track);
  buffer << type_name() << "("
         << "Value: " << value_track->ToString() << ", Node: " << node_->DebugString() << ")";
  return buffer.str();
}

AbstractBasePtr AbstractFunction::Join(const AbstractBasePtr &other) {
  MS_EXCEPTION_IF_NULL(other);
  auto other_func = dyn_cast<AbstractFunction>(other);
  if (other_func == nullptr) {
    MS_LOG(EXCEPTION) << "Join failed as type mismatch, this: " << ToString() << ", other: " << other->ToString();
  }
  return Join(other_func);
}

bool AbstractFunction::operator==(const AbstractBase &other) const {
  if (!other.isa<AbstractFunction>()) {
    return false;
  }
  const auto &other_func = static_cast<const AbstractFunction &>(other);
  bool value_equal = (*this == other_func);
  return value_equal;
}

const AbstractBasePtr AbstractSequeue::operator[](const std::size_t &dim) const {
  if (dim >= size()) {
    MS_LOG(EXCEPTION) << "Index [" << dim << "] Out of the size [" << size() << "] of the list.";
  }
  return elements_[dim];
}

std::string AbstractSequeue::ToString() const {
  std::ostringstream buffer;
  int64_t i = 0;
  for (const auto &ele : elements_) {
    MS_EXCEPTION_IF_NULL(ele);
    buffer << "element[" << i << "]: " << ele->ToString() << ",";
    i++;
  }
  return buffer.str();
}

TypePtrList AbstractSequeue::ElementsType() const {
  TypePtrList element_type_list;
  for (const auto &ele : elements_) {
    MS_EXCEPTION_IF_NULL(ele);
    TypePtr element_type = ele->BuildType();
    element_type_list.push_back(element_type);
  }
  return element_type_list;
}

BaseShapePtrList AbstractSequeue::ElementsShape() const {
  BaseShapePtrList element_shape_list;
  for (const auto &ele : elements_) {
    MS_EXCEPTION_IF_NULL(ele);
    BaseShapePtr element_shape = ele->BuildShape();
    element_shape_list.push_back(element_shape);
  }
  return element_shape_list;
}

AbstractBasePtrList AbstractSequeue::ElementsClone() const {
  AbstractBasePtrList ele_list;
  for (const auto &ele : elements_) {
    MS_EXCEPTION_IF_NULL(ele);
    AbstractBasePtr clone = ele->Clone();
    ele_list.push_back(clone);
  }
  return ele_list;
}

AbstractBasePtrList AbstractSequeue::ElementsBroaden(uint8_t config) const {
  AbstractBasePtrList ele_list;
  for (const auto &ele : elements_) {
    MS_EXCEPTION_IF_NULL(ele);
    AbstractBasePtr broadend = ele->Broaden(config);
    ele_list.push_back(broadend);
  }
  return ele_list;
}

template <typename T>
ValuePtr AbstractSequeue::ElementsBuildValue() const {
  std::vector<ValuePtr> element_value_list;
  for (const auto &ele : elements_) {
    ValuePtr element_value = ele->BuildValue();
    if (element_value->isa<AnyValue>()) {
      return kAnyValue;
    }
    element_value_list.push_back(element_value);
  }
  return std::make_shared<T>(element_value_list);
}
template ValuePtr AbstractSequeue::ElementsBuildValue<ValueTuple>() const;
template ValuePtr AbstractSequeue::ElementsBuildValue<ValueList>() const;

template <typename T>
AbstractBasePtr AbstractSequeue::ElementsJoin(const AbstractBasePtr &other) {
  auto other_sequeue = dyn_cast<T>(other);
  if (other_sequeue == nullptr) {
    MS_LOG(EXCEPTION) << "Join failed as type mismatch, this: " << ToString() << ", other: " << other->ToString();
  }
  auto joined_list = AbstractJoin(elements_, other_sequeue->elements_);
  bool changes = false;
  for (std::size_t i = 0; i < elements_.size(); i++) {
    if (elements_[i] != joined_list[i]) {
      changes = true;
      break;
    }
  }
  if (!changes) {
    return shared_from_base<AbstractBase>();
  }
  return std::make_shared<T>(joined_list);
}
template AbstractBasePtr AbstractSequeue::ElementsJoin<AbstractList>(const AbstractBasePtr &);
template AbstractBasePtr AbstractSequeue::ElementsJoin<AbstractTuple>(const AbstractBasePtr &);

std::size_t AbstractSequeue::hash() const {
  std::size_t hash_sum = hash_combine(tid(), std::hash<size_t>{}(elements_.size()));
  // Hashing all elements is costly, so only take at most 4 elements into account based on
  // some experiments.
  for (size_t i = 0; (i < elements_.size()) && (i < 4); i++) {
    hash_sum = hash_combine(hash_sum, elements_[i]->hash());
  }
  return hash_sum;
}

bool AbstractTuple::operator==(const AbstractTuple &other) const {
  if (&other == this) {
    return true;
  }

  if (elements_.size() != other.elements_.size()) {
    return false;
  }
  for (size_t i = 0; i < elements_.size(); i++) {
    if (!(*(elements_[i]) == *(other.elements_[i]))) {
      return false;
    }
  }
  return true;
}

bool AbstractTuple::operator==(const AbstractBase &other) const {
  if (&other == this) {
    return true;
  }

  if (other.isa<AbstractTuple>()) {
    auto other_tuple = static_cast<const AbstractTuple *>(&other);
    return *this == *other_tuple;
  }

  return false;
}

bool AbstractList::operator==(const AbstractList &other) const {
  if (&other == this) {
    return true;
  }

  if (elements_.size() != other.elements_.size()) {
    return false;
  }
  for (size_t i = 0; i < elements_.size(); i++) {
    if (!(*(elements_[i]) == *(other.elements_[i]))) {
      return false;
    }
  }
  return true;
}

bool AbstractList::operator==(const AbstractBase &other) const {
  if (&other == this) {
    return true;
  }

  if (other.isa<AbstractList>()) {
    auto other_list = static_cast<const AbstractList *>(&other);
    return *this == *other_list;
  }
  return false;
}

TypePtr AbstractSlice::BuildType() const {
  MS_EXCEPTION_IF_NULL(start_);
  MS_EXCEPTION_IF_NULL(stop_);
  MS_EXCEPTION_IF_NULL(step_);
  TypePtr start = start_->BuildType();
  TypePtr stop = stop_->BuildType();
  TypePtr step = step_->BuildType();
  return std::make_shared<Slice>(start, stop, step);
}

bool AbstractSlice::operator==(const AbstractSlice &other) const {
  if (&other == this) {
    return true;
  }
  return (*start_ == *other.start_ && *stop_ == *other.stop_ && *step_ == *other.step_);
}

bool AbstractSlice::operator==(const AbstractBase &other) const {
  if (&other == this) {
    return true;
  }
  if (!other.isa<AbstractSlice>()) {
    return false;
  }
  auto other_slice = static_cast<const AbstractSlice *>(&other);
  return *this == *other_slice;
}

AbstractBasePtr AbstractSlice::Clone() const {
  MS_EXCEPTION_IF_NULL(start_);
  MS_EXCEPTION_IF_NULL(stop_);
  MS_EXCEPTION_IF_NULL(step_);
  AbstractBasePtr start = start_->Clone();
  AbstractBasePtr stop = stop_->Clone();
  AbstractBasePtr step = step_->Clone();
  return std::make_shared<AbstractSlice>(start, stop, step);
}

AbstractBasePtr AbstractSlice::Broaden(uint8_t config) const {
  MS_EXCEPTION_IF_NULL(start_);
  MS_EXCEPTION_IF_NULL(stop_);
  MS_EXCEPTION_IF_NULL(step_);
  AbstractBasePtr start = start_->Broaden(config);
  AbstractBasePtr stop = stop_->Broaden(config);
  AbstractBasePtr step = step_->Broaden(config);
  return std::make_shared<AbstractSlice>(start, stop, step);
}

std::string AbstractSlice::ToString() const {
  std::ostringstream buffer;
  buffer << type_name() << "[";
  MS_EXCEPTION_IF_NULL(start_);
  buffer << start_->ToString() << " : ";
  MS_EXCEPTION_IF_NULL(stop_);
  buffer << stop_->ToString() << " : ";
  MS_EXCEPTION_IF_NULL(step_);
  buffer << step_->ToString();
  buffer << "]";
  return buffer.str();
}

ValuePtr AbstractSlice::RealBuildValue() const {
  MS_EXCEPTION_IF_NULL(start_);
  MS_EXCEPTION_IF_NULL(stop_);
  MS_EXCEPTION_IF_NULL(step_);
  ValuePtr start = start_->BuildValue();
  ValuePtr stop = stop_->BuildValue();
  ValuePtr step = step_->BuildValue();
  if (start->isa<AnyValue>() || stop->isa<AnyValue>() || step->isa<AnyValue>()) {
    return kAnyValue;
  }
  return std::make_shared<ValueSlice>(start, stop, step);
}

std::size_t AbstractSlice::hash() const {
  MS_EXCEPTION_IF_NULL(start_);
  MS_EXCEPTION_IF_NULL(stop_);
  MS_EXCEPTION_IF_NULL(step_);
  return hash_combine({tid(), start_->hash(), stop_->hash(), step_->hash()});
}

ShapePtr AbstractUndetermined::shape() const {
  auto shp = dyn_cast<Shape>(GetShapeTrack());
  if (shp == nullptr) {
    MS_LOG(EXCEPTION) << "Tensor should have a shape.";
  }
  return shp;
}

TypePtr AbstractTensor::BuildType() const {
  MS_EXCEPTION_IF_NULL(element_);
  TypePtr element_type = element_->BuildType();
  return std::make_shared<TensorType>(element_type);
}

BaseShapePtr AbstractTensor::BuildShape() const {
  auto shape = GetShapeTrack();
  // Guard from using set_shape(nullptr)
  if (shape == nullptr) {
    return kNoShape;
  }
  return shape;
}

AbstractBasePtr AbstractTensor::Join(const AbstractBasePtr &other) {
  if (other->BuildType()->type_id() == kObjectTypeUndeterminedType) {
    auto other_tensor = dyn_cast<AbstractUndetermined>(other);
    auto element = element_->Join(other_tensor->element());
    auto shape = ShapeJoin(this->shape(), other_tensor->shape());
    auto ret = std::make_shared<AbstractUndetermined>(element, shape);
    return ret;
  }
  auto other_tensor = dyn_cast<AbstractTensor>(other);
  if (other_tensor == nullptr) {
    MS_LOG(EXCEPTION) << "Join failed as type mismatch, this: " << ToString() << ", other: " << other->ToString();
  }
  if (*this == *other) {
    return shared_from_base<AbstractBase>();
  }
  auto element = element_->Join(other_tensor->element_);
  auto shape = ShapeJoin(this->shape(), other_tensor->shape());
  return std::make_shared<AbstractTensor>(element, shape);
}

bool AbstractTensor::equal_to(const AbstractTensor &other) const {
  if (&other == this) {
    return true;
  }

  auto v1 = GetValueTrack();
  auto v2 = other.GetValueTrack();
  if (v1 == nullptr || v2 == nullptr) {
    MS_LOG(EXCEPTION) << "The value of AbstractTensor is nullptr";
  }

  bool is_value_equal = (v1 == v2);
  if (v1->isa<AnyValue>() && v2->isa<AnyValue>()) {
    is_value_equal = true;
  }
  return (*element_ == *other.element_) && (*shape() == *other.shape()) && is_value_equal;
}

bool AbstractTensor::operator==(const AbstractTensor &other) const { return equal_to(other); }

bool AbstractTensor::operator==(const AbstractBase &other) const {
  if (&other == this) {
    return true;
  }

  if (other.tid() == tid()) {
    auto other_tensor = static_cast<const AbstractTensor *>(&other);
    return *this == *other_tensor;
  } else {
    return false;
  }
}

AbstractBasePtr AbstractTensor::Clone() const {
  MS_EXCEPTION_IF_NULL(element_);
  auto clone = std::make_shared<AbstractTensor>(element_->Clone());
  ShapePtr shp = shape();
  clone->set_shape(shp->Clone());
  clone->set_value(GetValueTrack());
  clone->set_value_range(get_min_value(), get_max_value());
  return clone;
}

AbstractBasePtr AbstractTensor::Broaden(uint8_t config) const {
  MS_EXCEPTION_IF_NULL(element_);
  auto broaden = std::make_shared<AbstractTensor>(element_->Broaden());
  auto shp = shape();
  broaden->set_shape(shp->Clone());
  auto not_broaden = config & kBroadenParameterOnly;
  if (not_broaden == 0) {
    broaden->set_value(kAnyValue);
  }
  return broaden;
}

AbstractBasePtr AbstractTensor::BroadenWithShape() const {
  MS_EXCEPTION_IF_NULL(element_);
  auto broaden = std::make_shared<AbstractTensor>(element_->Broaden());
  auto shp = shape()->Clone();
  shp->Broaden();
  broaden->set_shape(shp);
  broaden->set_value(kAnyValue);
  return broaden;
}

std::string AbstractTensor::ToString() const {
  std::ostringstream buffer;
  BaseShapePtr shape_track = GetShapeTrack();
  MS_EXCEPTION_IF_NULL(shape_track);
  MS_EXCEPTION_IF_NULL(element_);
  auto value_track = GetValueTrack();
  MS_EXCEPTION_IF_NULL(value_track);
  buffer << type_name() << "("
         << "shape: " << shape_track->ToString() << ", element: " << element_->ToString()
         << ", value_ptr: " << value_track << ", value: " << value_track->ToString() << ")";
  return buffer.str();
}

TypePtr AbstractDictionary::BuildType() const {
  std::vector<std::pair<std::string, TypePtr>> key_values;
  for (const auto &item : key_values_) {
    MS_EXCEPTION_IF_NULL(item.second);
    TypePtr type = item.second->BuildType();
    key_values.emplace_back(item.first, type);
  }
  return std::make_shared<Dictionary>(key_values);
}

bool AbstractDictionary::operator==(const AbstractDictionary &other) const {
  if (key_values_.size() != other.key_values_.size()) {
    return false;
  }

  for (size_t index = 0; index < key_values_.size(); index++) {
    if (key_values_[index].first != other.key_values_[index].first) {
      return false;
    }
    if (!(*key_values_[index].second == *other.key_values_[index].second)) {
      return false;
    }
  }
  return true;
}

bool AbstractDictionary::operator==(const AbstractBase &other) const {
  if (&other == this) {
    return true;
  }
  if (other.isa<AbstractDictionary>()) {
    auto other_class = static_cast<const AbstractDictionary *>(&other);
    return *this == *other_class;
  }
  return false;
}

AbstractBasePtr AbstractDictionary::Clone() const {
  std::vector<AbstractAttribute> kv;
  (void)std::transform(key_values_.begin(), key_values_.end(), std::back_inserter(kv),
                       [](const AbstractAttribute &item) {
                         MS_EXCEPTION_IF_NULL(item.second);
                         return std::make_pair(item.first, item.second->Clone());
                       });
  return std::make_shared<AbstractDictionary>(kv);
}

AbstractBasePtr AbstractDictionary::Broaden(uint8_t config) const {
  std::vector<AbstractAttribute> kv;
  (void)std::transform(key_values_.begin(), key_values_.end(), std::back_inserter(kv),
                       [config](const AbstractAttribute &item) {
                         MS_EXCEPTION_IF_NULL(item.second);
                         return std::make_pair(item.first, item.second->Broaden(config));
                       });
  return std::make_shared<AbstractDictionary>(kv);
}

std::string AbstractDictionary::ToString() const {
  std::ostringstream buffer;
  buffer << type_name() << "{ ";
  for (const auto &kv : key_values_) {
    MS_EXCEPTION_IF_NULL(kv.second);
    buffer << "(" << kv.first << ": " << kv.second->ToString() << ") ";
  }
  buffer << "}";
  return buffer.str();
}

std::size_t AbstractDictionary::hash() const {
  std::size_t hash_sum = std::accumulate(key_values_.begin(), key_values_.end(), tid(),
                                         [](std::size_t hash_sum, const AbstractAttribute &item) {
                                           hash_sum = hash_combine(hash_sum, std::hash<std::string>()(item.first));
                                           MS_EXCEPTION_IF_NULL(item.second);
                                           hash_sum = hash_combine(hash_sum, item.second->hash());
                                           return hash_sum;
                                         });
  return hash_sum;
}

ValuePtr AbstractDictionary::RealBuildValue() const {
  std::vector<std::pair<std::string, ValuePtr>> key_values;
  for (const auto &item : key_values_) {
    MS_EXCEPTION_IF_NULL(item.second);
    auto element_value = item.second->BuildValue();
    MS_EXCEPTION_IF_NULL(element_value);
    if (element_value->isa<AnyValue>()) {
      return kAnyValue;
    }
    key_values.emplace_back(item.first, element_value);
  }
  return std::make_shared<ValueDictionary>(key_values);
}

TypePtr AbstractClass::BuildType() const {
  ClassAttrVector attributes_type;
  for (auto attr : attributes_) {
    MS_EXCEPTION_IF_NULL(attr.second);
    TypePtr type = attr.second->BuildType();
    std::pair<std::string, TypePtr> elem(attr.first, type);
    attributes_type.push_back(elem);
  }

  return std::make_shared<Class>(tag_, attributes_type, methods_);
}

bool AbstractClass::operator==(const AbstractClass &other) const {
  if (!(tag_ == other.tag_)) {
    return false;
  }
  if (attributes_.size() != other.attributes_.size()) {
    return false;
  }
  for (size_t i = 0; i < attributes_.size(); i++) {
    MS_EXCEPTION_IF_NULL(attributes_[i].second);
    MS_EXCEPTION_IF_NULL(other.attributes_[i].second);
    if (!(*attributes_[i].second == *other.attributes_[i].second)) {
      MS_LOG(DEBUG) << "attr " << attributes_[i].first << " not equal, arg1:" << attributes_[i].second->ToString()
                    << " arg2:" << other.attributes_[i].second->ToString();
      return false;
    }
  }
  // method compare;
  if (methods_.size() != other.methods_.size()) {
    return false;
  }
  for (const auto &iter : methods_) {
    auto iter_other = other.methods_.find(iter.first);
    if (iter_other == other.methods_.end()) {
      return false;
    }
    if (!(*iter.second == *iter_other->second)) {
      return false;
    }
  }
  return true;
}

bool AbstractClass::operator==(const AbstractBase &other) const {
  if (other.isa<AbstractClass>()) {
    auto other_class = static_cast<const AbstractClass *>(&other);
    return *this == *other_class;
  }
  return false;
}

AbstractBasePtr AbstractClass::GetAttribute(const std::string &name) {
  auto it = std::find_if(attributes_.begin(), attributes_.end(),
                         [name](const AbstractAttribute &pair) -> bool { return pair.first == name; });
  if (it != attributes_.end()) {
    return it->second;
  }
  return nullptr;
}

ValuePtr AbstractClass::GetMethod(const std::string &name) {
  auto method_pair = methods_.find(name);
  if (method_pair != methods_.end()) {
    return method_pair->second;
  }
  return kAnyValue;
}

AbstractBasePtr AbstractClass::Clone() const {
  std::vector<AbstractAttribute> attributes_clone;
  for (auto attr : attributes_) {
    MS_EXCEPTION_IF_NULL(attr.second);
    AbstractBasePtr clone = attr.second->Clone();
    AbstractAttribute elem(attr.first, clone);
    attributes_clone.push_back(elem);
  }
  return std::make_shared<AbstractClass>(tag_, attributes_clone, methods_);
}

AbstractBasePtr AbstractClass::Broaden(uint8_t config) const {
  std::vector<AbstractAttribute> attributes_clone;
  for (auto attr : attributes_) {
    MS_EXCEPTION_IF_NULL(attr.second);
    AbstractBasePtr clone = attr.second->Broaden(config);
    AbstractAttribute elem(attr.first, clone);
    attributes_clone.push_back(elem);
  }
  return std::make_shared<AbstractClass>(tag_, attributes_clone, methods_);
}

std::string AbstractClass::ToString() const {
  std::ostringstream buffer;
  buffer << type_name() << "(tag: " << tag_ << ") attrs:(";
  bool append_comma = false;
  for (const auto &attr : attributes_) {
    if (append_comma) {
      buffer << ", ";
    } else {
      append_comma = true;
    }
    MS_EXCEPTION_IF_NULL(attr.second);
    buffer << attr.first << ":" << attr.second->ToString();
  }
  buffer << ") method:(";
  append_comma = false;
  for (const auto &iter : methods_) {
    if (append_comma) {
      buffer << ", ";
    } else {
      append_comma = true;
    }
    MS_EXCEPTION_IF_NULL(iter.second);
    buffer << iter.first << ":" << iter.second->ToString();
  }
  buffer << ")";
  return buffer.str();
}

std::size_t AbstractClass::hash() const {
  std::size_t hash_sum = std::accumulate(attributes_.begin(), attributes_.end(), hash_combine(tid(), tag_.hash()),
                                         [](std::size_t hash_sum, const AbstractAttribute &item) {
                                           MS_EXCEPTION_IF_NULL(item.second);
                                           return hash_combine(hash_sum, item.second->hash());
                                         });

  return hash_sum;
}

ValuePtr AbstractClass::RealBuildValue() const {
  auto cls = BuildType()->cast<ClassPtr>();
  std::unordered_map<std::string, ValuePtr> attributes_value_map;
  for (const auto &attr : attributes_) {
    MS_EXCEPTION_IF_NULL(attr.second);
    ValuePtr _value = attr.second->BuildValue();
    if (_value->isa<AnyValue>()) {
      return kAnyValue;
    }
    attributes_value_map[attr.first] = _value;
  }
  cls->set_value(attributes_value_map);
  return cls;
}

TypePtr AbstractJTagged::BuildType() const {
  MS_EXCEPTION_IF_NULL(element_);
  TypePtr subtype = element_->BuildType();
  return std::make_shared<JTagged>(subtype);
}

AbstractBasePtr AbstractJTagged::Join(const AbstractBasePtr &other) {
  auto other_jtagged = dyn_cast<AbstractJTagged>(other);
  if (other_jtagged == nullptr) {
    MS_LOG(EXCEPTION) << "Join failed as type mismatch, this: " << ToString() << ", other: " << other->ToString();
  }
  auto joined_elem = element_->Join(other_jtagged->element_);
  return std::make_shared<AbstractJTagged>(joined_elem);
}

bool AbstractJTagged::operator==(const AbstractJTagged &other) const {
  MS_EXCEPTION_IF_NULL(element_);
  MS_EXCEPTION_IF_NULL(other.element_);
  return (*element_ == *other.element_);
}

bool AbstractJTagged::operator==(const AbstractBase &other) const {
  if (other.isa<AbstractJTagged>()) {
    auto other_jtagged = static_cast<const AbstractJTagged *>(&other);
    return *this == *other_jtagged;
  }
  return false;
}

std::string AbstractJTagged::ToString() const {
  std::ostringstream buffer;
  MS_EXCEPTION_IF_NULL(element_);
  buffer << type_name() << "("
         << "element: " << element_->ToString() << ")";
  return buffer.str();
}

AbstractRef::AbstractRef(const AbstractBasePtr &ref_key, const AbstractTensorPtr &ref_value)
    : AbstractTensor(*ref_value), ref_key_(ref_key), ref_key_value_(nullptr) {
  set_type(std::make_shared<RefType>());
  if (ref_key && ref_key->isa<AbstractRefKey>()) {
    ref_key_value_ = ref_key->cast<AbstractRefKeyPtr>()->ref_key_value();
  }
}

TypePtr AbstractRef::BuildType() const {
  auto subtype = AbstractTensor::BuildType()->cast<TensorTypePtr>();
  return std::make_shared<RefType>(subtype);
}

bool AbstractRef::operator==(const AbstractRef &other) const {
  return AbstractTensor::equal_to(other) && (*ref_key_ == *other.ref_key_);
}

bool AbstractRef::operator==(const AbstractBase &other) const {
  if (other.isa<AbstractRef>()) {
    auto other_conf = static_cast<const AbstractRef *>(&other);
    return *this == *other_conf;
  }
  return false;
}

AbstractBasePtr AbstractRefKey::Join(const AbstractBasePtr &other) {
  MS_EXCEPTION_IF_NULL(other);
  if (*this == *other) {
    auto ret = shared_from_base<AbstractBase>();
    return ret;
  }
  auto value_self = GetValueTrack();
  MS_EXCEPTION_IF_NULL(value_self);
  ValuePtr res_value = ValueJoin(value_self, other->GetValueTrack());
  if (res_value == value_self) {
    auto ret = shared_from_base<AbstractBase>();
    return ret;
  }
  auto ret = std::make_shared<AbstractRefKey>();
  ret->set_value(res_value);
  return ret;
}

AbstractBasePtr AbstractRef::Join(const AbstractBasePtr &other) {
  auto other_ref = other->cast<AbstractRefPtr>();
  if (other_ref == nullptr) {
    return AbstractTensor::Join(other)->cast<AbstractTensorPtr>();
  }
  if ((*this == *other) && (*ref_key_ == *other_ref->ref_key_)) {
    return shared_from_base<AbstractBase>();
  }
  auto ref_key = ref_key_->Join(other_ref->ref_key_);
  auto ref = AbstractTensor::Join(other_ref->ref())->cast<AbstractTensorPtr>();
  return std::make_shared<AbstractRef>(ref_key, ref);
}

std::string AbstractRef::ToString() const {
  std::ostringstream buffer;
  buffer << type_name() << "("
         << "key: " << ref_key_->ToString() << " ref_value: " << AbstractTensor::ToString();
  auto value = GetValueTrack();
  if (value) {
    buffer << ", value: " << value->ToString();
  }
  buffer << ")";
  return buffer.str();
}

bool AbstractNone::operator==(const AbstractNone &) const { return true; }

bool AbstractNone::operator==(const AbstractBase &other) const {
  if (other.isa<AbstractNone>()) {
    auto other_none = static_cast<const AbstractNone *>(&other);
    return *this == *other_none;
  }
  return false;
}

std::string AbstractNone::ToString() const {
  std::ostringstream buffer;
  buffer << type_name() << "(Value: None)";
  return buffer.str();
}

ValuePtr AbstractNone::RealBuildValue() const { return kNone; }

AbstractBasePtr AbstractRefKey::Broaden(uint8_t config) const {
  auto refkey = std::make_shared<AbstractRefKey>();
  auto not_broaden = config & (kBroadenTensorOnly | kBroadenParameterOnly);
  if (not_broaden == 0) {
    refkey->set_value(kAnyValue);
  }
  return refkey;
}

bool AbstractRefKey::operator==(const AbstractRefKey &other) const {
  ValuePtr value_self = GetValueTrack();
  ValuePtr value_other = other.GetValueTrack();
  if (value_self != nullptr && value_other != nullptr) {
    if (value_self->isa<AnyValue>() && value_other->isa<AnyValue>()) {
      return true;
    }
    if (!value_self->isa<RefKey>() || !value_other->isa<RefKey>()) {
      return false;
    }
    RefKeyPtr type_self = value_self->cast<RefKeyPtr>();
    RefKeyPtr type_other = value_other->cast<RefKeyPtr>();
    return *type_self == *type_other;
  } else if (value_self != nullptr || value_other != nullptr) {
    return false;
  }
  return true;
}

bool AbstractRefKey::operator==(const AbstractBase &other) const {
  if (other.isa<AbstractRefKey>()) {
    auto other_confkey = static_cast<const AbstractRefKey *>(&other);
    return *this == *other_confkey;
  } else {
    return false;
  }
}

std::string AbstractRefKey::ToString() const {
  std::ostringstream buffer;
  buffer << type_name();
  auto value = GetValueTrack();
  if (value) {
    buffer << "(value: " << value->ToString() << ")";
  }
  return buffer.str();
}

bool AbstractNull::operator==(const AbstractNull &) const { return true; }

bool AbstractNull::operator==(const AbstractBase &other) const {
  if (&other == this) {
    return true;
  }
  if (other.isa<AbstractNull>()) {
    auto other_none = static_cast<const AbstractNull *>(&other);
    return *this == *other_none;
  } else {
    return false;
  }
}

std::string AbstractNull::ToString() const {
  std::ostringstream buffer;
  buffer << type_name() << "(Value: Null)";
  return buffer.str();
}

bool AbstractEllipsis::operator==(const AbstractEllipsis &) const { return true; }

bool AbstractEllipsis::operator==(const AbstractBase &other) const {
  if (&other == this) {
    return true;
  }
  if (other.isa<AbstractEllipsis>()) {
    auto other_none = static_cast<const AbstractEllipsis *>(&other);
    return *this == *other_none;
  } else {
    return false;
  }
}

std::string AbstractEllipsis::ToString() const {
  std::ostringstream buffer;
  buffer << type_name() << "(Value: Ellipsis)";
  return buffer.str();
}

TypePtr AbstractKeywordArg::BuildType() const {
  MS_EXCEPTION_IF_NULL(arg_value_);
  TypePtr type = arg_value_->BuildType();
  return std::make_shared<Keyword>(arg_name_, type);
}

AbstractBasePtr AbstractKeywordArg::Clone() const {
  MS_EXCEPTION_IF_NULL(arg_value_);
  return std::make_shared<AbstractKeywordArg>(arg_name_, arg_value_->Clone());
}

AbstractBasePtr AbstractKeywordArg::Broaden(uint8_t config) const {
  MS_EXCEPTION_IF_NULL(arg_value_);
  return std::make_shared<AbstractKeywordArg>(arg_name_, arg_value_->Broaden(config));
}

std::size_t AbstractKeywordArg::hash() const {
  MS_EXCEPTION_IF_NULL(arg_value_);
  return hash_combine({tid(), std::hash<std::string>{}(arg_name_), arg_value_->hash()});
}

std::string AbstractKeywordArg::ToString() const {
  std::ostringstream buffer;
  MS_EXCEPTION_IF_NULL(arg_value_);
  buffer << type_name() << "(";
  buffer << "key : " << arg_name_;
  buffer << "value : " << arg_value_->ToString();
  buffer << ")";
  return buffer.str();
}

bool AbstractKeywordArg::operator==(const AbstractBase &other) const {
  if (&other == this) {
    return true;
  }

  if (other.isa<AbstractKeywordArg>()) {
    auto other_tuple = static_cast<const AbstractKeywordArg *>(&other);
    return *this == *other_tuple;
  }
  return false;
}

bool AbstractKeywordArg::operator==(const AbstractKeywordArg &other) const {
  if (&other == this) {
    return true;
  }
  MS_EXCEPTION_IF_NULL(arg_value_);
  MS_EXCEPTION_IF_NULL(other.arg_value_);
  return other.arg_name_ == arg_name_ && *other.arg_value_ == *arg_value_;
}

ValuePtr AbstractKeywordArg::RealBuildValue() const {
  MS_EXCEPTION_IF_NULL(arg_value_);
  ValuePtr value = arg_value_->BuildValue();
  MS_EXCEPTION_IF_NULL(value);
  if (value->isa<AnyValue>()) {
    return kAnyValue;
  }
  return std::make_shared<KeywordArg>(arg_name_, value);
}

std::size_t AbstractBasePtrListHash(const AbstractBasePtrList &args_spec_list) {
  std::size_t hash_value = 0;
  // Hashing all elements is costly, so only take at most 4 elements into account based on
  // some experiments.
  for (size_t i = 0; (i < args_spec_list.size()) && (i < 4); i++) {
    MS_EXCEPTION_IF_NULL(args_spec_list[i]);
    hash_value = hash_combine(hash_value, args_spec_list[i]->hash());
  }
  return hash_value;
}

bool AbstractBasePtrListDeepEqual(const AbstractBasePtrList &lhs, const AbstractBasePtrList &rhs) {
  if (lhs.size() != rhs.size()) {
    return false;
  }
  std::size_t size = lhs.size();
  for (std::size_t i = 0; i < size; i++) {
    MS_EXCEPTION_IF_NULL(lhs[i]);
    MS_EXCEPTION_IF_NULL(rhs[i]);
    if (lhs[i] == rhs[i]) {
      continue;
    }
    if (!(*lhs[i] == *rhs[i])) {
      return false;
    }
  }
  return true;
}

std::size_t AbstractBasePtrListHasher::operator()(const AbstractBasePtrList &args_spec_list) const {
  return AbstractBasePtrListHash(args_spec_list);
}

bool AbstractBasePtrListEqual::operator()(const AbstractBasePtrList &lhs, const AbstractBasePtrList &rhs) const {
  return AbstractBasePtrListDeepEqual(lhs, rhs);
}

// RowTensor
TypePtr AbstractRowTensor::BuildType() const {
  MS_EXCEPTION_IF_NULL(element());
  TypePtr element_type = element()->BuildType();
  return std::make_shared<RowTensorType>(element_type);
}

AbstractBasePtr AbstractRowTensor::Clone() const {
  MS_EXCEPTION_IF_NULL(element());
  auto clone = std::make_shared<AbstractRowTensor>(element()->Clone());
  ShapePtr shp = shape();
  clone->set_shape(shp->Clone());
  clone->set_value(GetValueTrack());
  clone->set_indices(indices_->Clone()->cast<AbstractTensorPtr>());
  clone->set_values(values_->Clone()->cast<AbstractTensorPtr>());
  clone->set_dense_shape(dense_shape_->Clone()->cast<AbstractTuplePtr>());
  return clone;
}

AbstractBasePtr AbstractRowTensor::Broaden(uint8_t config) const {
  MS_EXCEPTION_IF_NULL(element());
  auto broaden = std::make_shared<AbstractRowTensor>(element()->Broaden());
  auto shp = shape();
  broaden->set_shape(shp->Clone());
  broaden->set_value(kAnyValue);
  broaden->set_indices(indices_->Clone()->cast<AbstractTensorPtr>());
  broaden->set_values(values_->Clone()->cast<AbstractTensorPtr>());
  broaden->set_dense_shape(dense_shape_->Clone()->cast<AbstractTuplePtr>());
  return broaden;
}

AbstractBasePtr AbstractRowTensor::BroadenWithShape() const {
  MS_EXCEPTION_IF_NULL(element());
  auto broaden = std::make_shared<AbstractRowTensor>(element()->Broaden());
  auto shp = shape()->Clone();
  shp->Broaden();
  broaden->set_shape(shp);
  broaden->set_value(kAnyValue);
  broaden->set_indices(indices_->Clone()->cast<AbstractTensorPtr>());
  broaden->set_values(values_->Clone()->cast<AbstractTensorPtr>());
  broaden->set_dense_shape(dense_shape_->Clone()->cast<AbstractTuplePtr>());
  return broaden;
}

std::string AbstractRowTensor::ToString() const {
  std::ostringstream buffer;
  BaseShapePtr shape_track = GetShapeTrack();
  MS_EXCEPTION_IF_NULL(shape_track);
  MS_EXCEPTION_IF_NULL(element());
  auto value_track = GetValueTrack();
  MS_EXCEPTION_IF_NULL(value_track);
  buffer << type_name() << "("
         << "shape: " << shape_track->ToString() << ", element: " << element()->ToString()
         << ", value_ptr: " << value_track << ", value: " << value_track->ToString() << ")"
         << ", indices: " << indices_->ToString() << ", values" << values_->ToString()
         << ", dense_shape: " << dense_shape_->ToString();
  return buffer.str();
}

// SparseTensor
TypePtr AbstractSparseTensor::BuildType() const {
  MS_EXCEPTION_IF_NULL(element());
  TypePtr element_type = element()->BuildType();
  return std::make_shared<SparseTensorType>(element_type);
}

AbstractBasePtr AbstractSparseTensor::Clone() const {
  MS_EXCEPTION_IF_NULL(element());
  auto clone = std::make_shared<AbstractSparseTensor>(element()->Clone());
  ShapePtr shp = shape();
  clone->set_shape(shp->Clone());
  clone->set_value(GetValueTrack());
  clone->set_indices(indices_->Clone()->cast<AbstractTensorPtr>());
  clone->set_values(values_->Clone()->cast<AbstractTensorPtr>());
  clone->set_dense_shape(dense_shape_->Clone()->cast<AbstractTuplePtr>());
  return clone;
}

AbstractBasePtr AbstractSparseTensor::Broaden(uint8_t config) const {
  MS_EXCEPTION_IF_NULL(element());
  auto broaden = std::make_shared<AbstractSparseTensor>(element()->Broaden());
  auto shp = shape();
  broaden->set_shape(shp->Clone());
  broaden->set_value(kAnyValue);
  broaden->set_indices(indices_->Clone()->cast<AbstractTensorPtr>());
  broaden->set_values(values_->Clone()->cast<AbstractTensorPtr>());
  broaden->set_dense_shape(dense_shape_->Clone()->cast<AbstractTuplePtr>());
  return broaden;
}

AbstractBasePtr AbstractSparseTensor::BroadenWithShape() const {
  MS_EXCEPTION_IF_NULL(element());
  auto broaden = std::make_shared<AbstractSparseTensor>(element()->Broaden());
  auto shp = shape()->Clone();
  shp->Broaden();
  broaden->set_shape(shp);
  broaden->set_value(kAnyValue);
  broaden->set_indices(indices_->Clone()->cast<AbstractTensorPtr>());
  broaden->set_values(values_->Clone()->cast<AbstractTensorPtr>());
  broaden->set_dense_shape(dense_shape_->Clone()->cast<AbstractTuplePtr>());
  return broaden;
}

std::string AbstractSparseTensor::ToString() const {
  std::ostringstream buffer;
  BaseShapePtr shape_track = GetShapeTrack();
  MS_EXCEPTION_IF_NULL(shape_track);
  MS_EXCEPTION_IF_NULL(element());
  auto value_track = GetValueTrack();
  MS_EXCEPTION_IF_NULL(value_track);
  buffer << type_name() << "("
         << "shape: " << shape_track->ToString() << ", element: " << element()->ToString()
         << ", value_ptr: " << value_track << ", value: " << value_track->ToString() << ")"
         << ", indices: " << indices_->ToString() << ", values" << values_->ToString()
         << ", dense_shape: " << dense_shape_->ToString();
  return buffer.str();
}

AbstractBasePtr AbstractUMonad::Join(const AbstractBasePtr &other) {
  MS_EXCEPTION_IF_NULL(other);
  if (other->isa<AbstractUMonad>()) {
    return shared_from_base<AbstractBase>();
  }
  MS_EXCEPTION(TypeError) << "Type join failed, type1 = " << GetTypeTrack()->ToString()
                          << ", type2 = " << other->GetTypeTrack()->ToString();
}

bool AbstractUMonad::operator==(const AbstractUMonad &) const { return true; }

bool AbstractUMonad::operator==(const AbstractBase &other) const {
  if (&other == this) {
    return true;
  }
  return other.isa<AbstractUMonad>();
}

AbstractBasePtr AbstractIOMonad::Join(const AbstractBasePtr &other) {
  MS_EXCEPTION_IF_NULL(other);
  if (other->isa<AbstractIOMonad>()) {
    return shared_from_base<AbstractBase>();
  }
  MS_EXCEPTION(TypeError) << "Type join failed, type1 = " << GetTypeTrack()->ToString()
                          << ", type2 = " << other->GetTypeTrack()->ToString();
}

bool AbstractIOMonad::operator==(const AbstractIOMonad &) const { return true; }

bool AbstractIOMonad::operator==(const AbstractBase &other) const {
  if (&other == this) {
    return true;
  }
  return other.isa<AbstractIOMonad>();
}

}  // namespace abstract
}  // namespace mindspore
