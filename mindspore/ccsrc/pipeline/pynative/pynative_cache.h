/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_PIPELINE_PYNATIVE_PYNATIVE_ABS_CACHE_H
#define MINDSPORE_CCSRC_PIPELINE_PYNATIVE_PYNATIVE_ABS_CACHE_H
#include <string>
#include <utility>
#include <vector>
#include <memory>
#include <unordered_map>
#include "ir/anf.h"

namespace mindspore::pynative {
struct AbsCacheKey {
  std::string prim_name_;
  size_t prim_hash_value_;
  std::unordered_map<std::string, ValuePtr> prim_attrs_;
};

struct AbsCacheKeyHasher {
  size_t operator()(const AbsCacheKey &key) const { return key.prim_hash_value_; }
};

struct AbsCacheKeyEqual {
  bool operator()(const AbsCacheKey &lk, const AbsCacheKey &rk) const {
    if (lk.prim_attrs_.size() != rk.prim_attrs_.size()) {
      return false;
    }
    if (lk.prim_name_ != rk.prim_name_) {
      return false;
    }

    auto all = std::all_of(lk.prim_attrs_.begin(), lk.prim_attrs_.end(),
                           [&rk](const std::pair<std::string, ValuePtr> &item) -> bool {
                             auto iter = rk.prim_attrs_.find(item.first);
                             if (iter == rk.prim_attrs_.end()) {
                               return false;
                             }
                             if (item.second == iter->second) {
                               return true;
                             }
                             MS_EXCEPTION_IF_NULL(item.second);
                             MS_EXCEPTION_IF_NULL(iter->second);
                             return *item.second == *iter->second;
                           });
    return all;
  }
};

struct PrimAbsInfo {
  abstract::AbstractBasePtr abs;
  bool is_dynamic_shape = false;
  std::unordered_map<std::string, ValuePtr> attrs;
};
using AbstractListMap = std::unordered_map<abstract::AbstractBasePtrList, PrimAbsInfo,
                                           abstract::AbstractBasePtrListHasher, abstract::AbstractBasePtrListEqual>;
using PrimAbsCache = std::unordered_map<AbsCacheKey, AbstractListMap, AbsCacheKeyHasher, AbsCacheKeyEqual>;

// Used for id
struct PyObjectHasher {
  size_t operator()(const py::handle &key) const { return py::hash(key); }
};

struct PyObjectEqual {
  bool operator()(const py::handle &p1, const py::handle &p2) const { return p1 == p2; }
};
using PyObjectIdCache = std::unordered_map<py::handle, std::string, PyObjectHasher, PyObjectEqual>;

struct PrimSignature {
  bool has_dtype_sig;
  std::vector<SignatureEnumDType> dtypes;
  std::unordered_map<SignatureEnumDType, std::vector<size_t>> type_indexes;
};
using ImplicitCastCache = std::unordered_map<std::string, PrimSignature>;
}  // namespace mindspore::pynative
#endif  // MINDSPORE_CCSRC_PIPELINE_PYNATIVE_PYNATIVE_ABS_CACHE_H
