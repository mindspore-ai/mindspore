/**
 * This is the C++ adaptation and derivative work of Myia (https://github.com/mila-iqia/myia/).
 *
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#include "pipeline/pynative/grad/function/function_utils.h"

namespace mindspore::pynative::autograd {

bool ExistTensor(const ValuePtr &value) {
  if (value->isa<tensor::Tensor>()) {
    return true;
  } else if (value->isa<ValueSequence>()) {
    bool is_exist = false;
    auto seq = value->cast<ValueSequencePtr>();
    for (const auto &val : seq->value()) {
      is_exist = is_exist || ExistTensor(val);
      if (is_exist) {
        return true;
      }
    }
    return is_exist;
  }
  return false;
}
void FlattenArgs(const ValuePtr &arg, ValuePtrList *flatten_args) {
  if (!arg->isa<ValueSequence>()) {
    (void)flatten_args->emplace_back(arg);
    return;
  } else {
    if (ExistTensor(arg)) {
      auto seq = arg->cast<ValueSequencePtr>();
      for (const auto &val : seq->value()) {
        FlattenArgs(val, flatten_args);
      }
    } else {
      (void)flatten_args->emplace_back(arg);
    }
  }
}

ValuePtrList FlattenArgs(const ValuePtrList &args) {
  ValuePtrList flatten_args;
  flatten_args.reserve(args.size());
  for (const auto &arg : args) {
    FlattenArgs(arg, &flatten_args);
  }
  return flatten_args;
}
}  // namespace mindspore::pynative::autograd