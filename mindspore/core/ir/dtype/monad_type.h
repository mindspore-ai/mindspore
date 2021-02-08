/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CORE_IR_DTYPE_MONAD_H_
#define MINDSPORE_CORE_IR_DTYPE_MONAD_H_

#include <memory>
#include <string>

#include "base/base.h"
#include "ir/dtype/type.h"

namespace mindspore {
class MonadType : public Object {
 public:
  ~MonadType() override = default;
  MS_DECLARE_PARENT(MonadType, Object)

  TypeId generic_type_id() const override { return kObjectTypeMonad; }
  TypePtr DeepCopy() const override = 0;

 protected:
  explicit MonadType(const TypeId type_id) : Object(type_id) {}
};
using MonadTypePtr = std::shared_ptr<MonadType>;

// UniversalMonad type
class UMonadType : public MonadType {
 public:
  UMonadType() : MonadType(kObjectTypeUMonad) {}
  ~UMonadType() override {}
  MS_DECLARE_PARENT(UMonadType, MonadType)

  TypeId generic_type_id() const override { return kObjectTypeUMonad; }
  TypePtr DeepCopy() const override { return std::make_shared<UMonadType>(); }
  std::string ToString() const override { return "UMonad"; }
};
using UMonadTypePtr = std::shared_ptr<UMonadType>;

// IOMonad type
class IOMonadType : public MonadType {
 public:
  IOMonadType() : MonadType(kObjectTypeIOMonad) {}
  ~IOMonadType() override {}
  MS_DECLARE_PARENT(IOMonadType, MonadType)

  TypeId generic_type_id() const override { return kObjectTypeIOMonad; }
  TypePtr DeepCopy() const override { return std::make_shared<IOMonadType>(); }
  std::string ToString() const override { return "IOMonad"; }
};
using IOMonadTypePtr = std::shared_ptr<IOMonadType>;

extern const TypePtr kIOMonadType;
extern const TypePtr kUMonadType;
}  // namespace mindspore

#endif  // MINDSPORE_CORE_IR_DTYPE_MONDA_H_
