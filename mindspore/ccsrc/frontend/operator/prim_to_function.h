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

#ifndef MINDSPORE_CCSRC_OPERATOR_PRIM_TO_FUNCTION_H_
#define MINDSPORE_CCSRC_OPERATOR_PRIM_TO_FUNCTION_H_

#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <typeindex>
#include <unordered_map>
#include <vector>

#include "ir/anf.h"
#include "ir/primitive.h"
#include "ir/dtype.h"

namespace mindspore {
/* namespace to support prim related definition */
namespace prim {
// Supported meta type
enum PrimType { kPrimTypeUnknown, kPrimTypeOneArg, kPrimTypeTwoArgs };

class PrimToFunction;

// Get the args, return value and function handle for a primitive instance.
class PrimToFunction {
 public:
  // Return a thread-safe singleton instance
  static PrimToFunction &GetInstance() {
    static PrimToFunction instance;
    return instance;
  }
  PrimToFunction(const PrimToFunction &) = delete;
  PrimToFunction &operator=(const PrimToFunction &) = delete;
  ~PrimToFunction() = default;

  // Get the args and return value for a primitive instance.
  bool GetFunction(const PrimitivePtr &prim, FunctionPtr *func) const;

 private:
  PrimToFunction();
  // Get the number of primitive arguments
  int GetPrimType(const PrimitivePtr &prim) const;
  const std::unordered_map<std::string, int> prim_func_type_map_;
};
}  // namespace prim
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_OPERATOR_PRIM_TO_FUNCTION_H_
