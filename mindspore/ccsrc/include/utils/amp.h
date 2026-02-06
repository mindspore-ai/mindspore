/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_INCLUDE_COMMON_AMP_AMP_H_
#define MINDSPORE_CCSRC_INCLUDE_COMMON_AMP_AMP_H_

#include <vector>
#include <string>
#include <memory>
#include <stack>
#include <utility>
#include <map>
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "mindspore/core/include/base/base.h"
#include "mindspore/core/include/ir/dtype/amp.h"
#include "include/utils/visible.h"

namespace py = pybind11;

namespace mindspore {
namespace amp {
AmpStrategyPtr COMMON_EXPORT CreateAmpStrategy(const AmpLevel amp_level, const TypePtr amp_dtype,
                                               const PrimArgList white_list, const PrimArgList black_list);
void COMMON_EXPORT PushAmpStratrgy(const AmpLevel amp_level, const TypePtr amp_dtype, const PrimArgList white_list,
                                   const PrimArgList black_list);
void COMMON_EXPORT PopAmpStrategy();
AmpStrategyPtr COMMON_EXPORT GetCurrentAmpStrategy();
PrimCastStrategyInfo COMMON_EXPORT GetPrimCastStrategyInfo(const AmpStrategyPtr &amp_strategy,
                                                           const std::string &op_name);
}  // namespace amp
void COMMON_EXPORT RegAmpModule(py::module *m);
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_INCLUDE_COMMON_AMP_AMP_H_
