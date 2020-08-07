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

#include "minddata/dataset/api/python/pybind_register.h"

namespace mindspore {
namespace dataset {

PybindDefinedFunctionRegister &PybindDefinedFunctionRegister::GetSingleton() {
  static PybindDefinedFunctionRegister instance;
  return instance;
}

// This is where we externalize the C logic as python modules
PYBIND11_MODULE(_c_dataengine, m) {
  m.doc() = "pybind11 for _c_dataengine";

  auto all_fns = mindspore::dataset::PybindDefinedFunctionRegister::AllFunctions();

  for (auto &item : all_fns) {
    for (auto &func : item.second) {
      func.second(&m);
    }
  }
}
}  // namespace dataset
}  // namespace mindspore
