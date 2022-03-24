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

#ifndef MINDSPORE_CORE_MINDAPI_IR_COMMON_H_
#define MINDSPORE_CORE_MINDAPI_IR_COMMON_H_

#include <vector>
#include "mindapi/base/shared_ptr.h"

namespace mindspore::api {
class AnfNode;
using AnfNodePtr = SharedPtr<AnfNode>;
using AnfNodePtrList = std::vector<AnfNodePtr>;

class Value;
using ValuePtr = SharedPtr<Value>;
using ValuePtrList = std::vector<ValuePtr>;

class Primitive;
using PrimitivePtr = SharedPtr<Primitive>;

class Type;
using TypePtr = SharedPtr<Type>;

class AbstractBase;
using AbstractBasePtr = SharedPtr<AbstractBase>;
using AbstractBasePtrList = std::vector<AbstractBasePtr>;

class Shape;
using ShapePtr = SharedPtr<Shape>;

class FuncGraph;
using FuncGraphPtr = SharedPtr<FuncGraph>;

class FuncGraphManager;
using FuncGraphManagerPtr = SharedPtr<FuncGraphManager>;

class CNode;
using CNodePtr = SharedPtr<CNode>;
using CNodePtrList = std::vector<CNodePtr>;
}  // namespace mindspore::api
#endif  // MINDSPORE_CORE_MINDAPI_IR_COMMON_H_
