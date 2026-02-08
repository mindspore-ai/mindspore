/**
 * This is the C++ adaptation and derivative work of Myia (https://github.com/mila-iqia/myia/).
 *
 * Copyright 2019-2025 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CORE_ABSTRACT_JOIN_H_
#define MINDSPORE_CORE_ABSTRACT_JOIN_H_

#include <utility>
#include <memory>
#include <functional>
#include "utils/shape_utils.h"
#include "abstract/abstract_value.h"

namespace mindspore {
namespace abstract {
ShapePtr ShapeJoin(const ShapePtr &shape1, const ShapePtr &shape2);
ValuePtr ValueJoin(const ValuePtr &value1, const ValuePtr &value2);
}  // namespace abstract
}  // namespace mindspore
#endif  // MINDSPORE_CORE_ABSTRACT_JOIN_H_
