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

#include "operator/prim_to_function.h"
#include <exception>
#include <iostream>
#include <string>

namespace mindspore {
// namespace to support prim related definition
namespace prim {

PrimToFunction::PrimToFunction()
    : prim_func_type_map_({// ONE_ARG prim
                           {"bool_not", kPrimTypeOneArg},
                           {"scalar_cos", kPrimTypeOneArg},
                           {"scalar_exp", kPrimTypeOneArg},
                           {"scalar_floor", kPrimTypeOneArg},
                           {"scalar_log", kPrimTypeOneArg},
                           {"scalar_sin", kPrimTypeOneArg},
                           {"scalar_tan", kPrimTypeOneArg},
                           {"scalar_trunc", kPrimTypeOneArg},
                           {"typeof", kPrimTypeOneArg},
                           {"scalar_uadd", kPrimTypeOneArg},
                           {"scalar_usub", kPrimTypeOneArg},
                           // TWO_ARGS prim
                           {"scalar_add", kPrimTypeTwoArgs},
                           {"bool_and", kPrimTypeTwoArgs},
                           {"bool_eq", kPrimTypeTwoArgs},
                           {"bool_or", kPrimTypeTwoArgs},
                           {"scalar_div", kPrimTypeTwoArgs},
                           {"scalar_eq", kPrimTypeTwoArgs},
                           {"scalar_ge", kPrimTypeTwoArgs},
                           {"scalar_gt", kPrimTypeTwoArgs},
                           {"scalar_le", kPrimTypeTwoArgs},
                           {"scalar_lt", kPrimTypeTwoArgs},
                           {"scalar_ne", kPrimTypeTwoArgs},
                           {"scalar_mod", kPrimTypeTwoArgs},
                           {"scalar_mul", kPrimTypeTwoArgs},
                           {"scalar_pow", kPrimTypeTwoArgs},
                           {"scalar_sub", kPrimTypeTwoArgs},
                           {"scalar_floordiv", kPrimTypeTwoArgs}}) {}

bool PrimToFunction::GetFunction(const PrimitivePtr &prim, FunctionPtr *const func) const {
  bool result = false;

  if (func != nullptr) {
    int args_num = GetPrimType(prim);
    std::vector<TypePtr> one_arg{std::make_shared<Number>()};
    std::vector<TypePtr> two_args{std::make_shared<Number>(), std::make_shared<Number>()};
    TypePtr retval = std::make_shared<Number>();
    result = true;
    switch (args_num) {
      case kPrimTypeOneArg:
        *func = Function(one_arg, retval).DeepCopy()->cast<FunctionPtr>();
        break;
      case kPrimTypeTwoArgs:
        *func = Function(two_args, retval).DeepCopy()->cast<FunctionPtr>();
        break;
      default:
        result = false;
        break;
    }
  }

  return result;
}

int PrimToFunction::GetPrimType(const PrimitivePtr &prim) const {
  MS_EXCEPTION_IF_NULL(prim);
  int prim_type = static_cast<int>(kPrimTypeUnknown);

  auto value = prim_func_type_map_.find(prim->name());
  if (value != prim_func_type_map_.end()) {
    prim_type = value->second;
  }
  return prim_type;
}
}  // namespace prim
}  // namespace mindspore
