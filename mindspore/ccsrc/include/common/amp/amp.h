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
#include "mindspore/core/base/base.h"
#include "mindspore/core/ir/dtype/type.h"
#include "include/common/visible.h"

namespace py = pybind11;

namespace mindspore {
namespace amp {
// prim_name and arg indexes which need to be casted, e.g. (prim0, [arg0, arg1, ...])
using PrimArg = std::pair<std::string, std::vector<uint8_t>>;
using PrimArgList = std::vector<PrimArg>;

typedef enum AmpLevel_ {
  O0 = 0,
  O1 = 1,
  O2 = 2,
  O3 = 3,
  Auto = 4,
} AmpLevel;

typedef enum PrimCastStrategy_ {
  Ignore = 0,       // Do not insert cast for inputs
  DoCast = 1,       // Insert cast for inputs with specific float dtype in PrimCastInfo
  SetDtype = 2,     // Set prim dtype to specific float dtype in PrimCastInfo
  SetDtypeOpt = 3,  // Set prim dtype to specific float dtype in PrimCastInfo if dtype is not set by user
  AutoPromote = 4,  // Insert cast for inputs with widest float type
} PrimCastStrategy;

typedef struct PrimCastStrategyInfo_ {
  PrimCastStrategy strategy;     // cast strategy
  TypePtr dtype;                 // dtype that inputs to be casted to
  std::vector<uint8_t> arg_pos;  // position of args that need to be casted, cast all float args when empty
} PrimCastStrategyInfo;

class COMMON_EXPORT AmpStrategy {
 public:
  AmpStrategy(const AmpLevel amp_level, const TypePtr amp_dtype, const PrimArgList white_list,
              const PrimArgList black_list)
      : amp_level_(amp_level), amp_dtype_(amp_dtype), white_list_(white_list), black_list_(black_list) {}
  ~AmpStrategy() = default;

  AmpLevel GetAmpLevel() { return amp_level_; }
  TypePtr GetAmpDtype() { return amp_dtype_; }
  PrimArgList GetWhiteList() { return white_list_; }
  PrimArgList GetBlackList() { return black_list_; }
  PrimCastStrategyInfo GetPrimCastStrategyInfo(const std::string &op_name);

 private:
  AmpLevel amp_level_ = AmpLevel::Auto;
  TypePtr amp_dtype_ = nullptr;
  PrimArgList white_list_;
  PrimArgList black_list_;
  PrimArgList set_dtype_list_;
  PrimArgList set_dtype_opt_list_;
  PrimArgList auto_promote_list_;
  std::map<std::string, PrimCastStrategyInfo> strategy_info_cache;
};
using AmpStrategyPtr = std::shared_ptr<AmpStrategy>;

AmpStrategyPtr COMMON_EXPORT CreateAmpStrategy(const AmpLevel amp_level, const TypePtr amp_dtype,
                                               const PrimArgList white_list, const PrimArgList black_list);
void COMMON_EXPORT PushAmpStratrgy(const AmpLevel amp_level, const TypePtr amp_dtype, const PrimArgList white_list,
                                   const PrimArgList black_list);
void COMMON_EXPORT PopAmpStrategy();
AmpStrategyPtr COMMON_EXPORT GetCurrentAmpStrategy();
}  // namespace amp
void COMMON_EXPORT RegAmpModule(py::module *m);
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_INCLUDE_COMMON_AMP_AMP_H_
