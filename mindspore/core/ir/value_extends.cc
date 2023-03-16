/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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

#include "ir/value.h"
#include <algorithm>
#include <memory>

#include "abstract/abstract_value.h"

namespace mindspore {
using ContextPtr = abstract::AnalysisContextPtr;

abstract::AbstractBasePtr Scalar::ToAbstract() {
  return std::make_shared<abstract::AbstractScalar>(shared_from_base<Value>());
}

abstract::AbstractBasePtr StringImm::ToAbstract() {
  return std::make_shared<abstract::AbstractScalar>(shared_from_base<Value>(), std::make_shared<String>());
}

abstract::AbstractBasePtr ValueAny::ToAbstract() { return std::make_shared<abstract::AbstractScalar>(); }

abstract::AbstractBasePtr ValueTuple::ToAbstract() {
  abstract::AbstractBasePtrList a_list;
  (void)std::transform(elements_.begin(), elements_.end(), std::back_inserter(a_list), [](const ValuePtr &ele) {
    MS_EXCEPTION_IF_NULL(ele);
    return ele->ToAbstract();
  });
  return std::make_shared<abstract::AbstractTuple>(a_list);
}

abstract::AbstractBasePtr ValueList::ToAbstract() {
  abstract::AbstractBasePtrList a_list;
  (void)std::transform(elements_.begin(), elements_.end(), std::back_inserter(a_list), [](const ValuePtr &ele) {
    MS_EXCEPTION_IF_NULL(ele);
    return ele->ToAbstract();
  });
  return std::make_shared<abstract::AbstractList>(a_list);
}

abstract::AbstractBasePtr ValueSlice::ToAbstract() {
  MS_EXCEPTION_IF_NULL(start_);
  MS_EXCEPTION_IF_NULL(stop_);
  MS_EXCEPTION_IF_NULL(step_);
  abstract::AbstractBasePtr start = start_->ToAbstract();
  abstract::AbstractBasePtr end = stop_->ToAbstract();
  abstract::AbstractBasePtr step = step_->ToAbstract();
  return std::make_shared<abstract::AbstractSlice>(start, end, step);
}

abstract::AbstractBasePtr KeywordArg::ToAbstract() {
  MS_EXCEPTION_IF_NULL(value_);
  abstract::AbstractBasePtr argument = value_->ToAbstract();
  return std::make_shared<abstract::AbstractKeywordArg>(key_, argument);
}

abstract::AbstractBasePtr ValueDictionary::ToAbstract() {
  std::vector<std::pair<abstract::AbstractBasePtr, abstract::AbstractBasePtr>> kv;
  (void)std::transform(key_values_.cbegin(), key_values_.cend(), std::back_inserter(kv),
                       [](const std::pair<ValuePtr, ValuePtr> &item) {
                         return std::make_pair(item.first->ToAbstract(), item.second->ToAbstract());
                       });
  return std::make_shared<abstract::AbstractDictionary>(kv);
}

abstract::AbstractBasePtr UMonad::ToAbstract() { return std::make_shared<abstract::AbstractUMonad>(); }

abstract::AbstractBasePtr IOMonad::ToAbstract() { return std::make_shared<abstract::AbstractIOMonad>(); }
}  // namespace mindspore
