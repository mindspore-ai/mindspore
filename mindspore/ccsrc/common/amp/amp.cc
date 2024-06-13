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
#include "include/common/amp/amp.h"
#include "mindspore/core/ir/dtype/number.h"
#include "utils/log_adapter.h"

namespace mindspore {
namespace amp {
static std::stack<AmpStrategyPtr> g_AmpStrategyStack;
PrimArgList g_SetDtypeOptList = {PrimArg("ProdExt", {}), PrimArg("SumExt", {})};
PrimArgList g_SetDtypeList = {};
PrimArgList g_AutoPromoteList = {PrimArg("Addcdiv", {}),  PrimArg("Addcmul", {}),       PrimArg("Cross", {}),
                                 PrimArg("Dot", {}),      PrimArg("GridSampler2D", {}), PrimArg("GridSampler3D", {}),
                                 PrimArg("IndexPut", {}), PrimArg("BiasAdd", {})};

PrimCastStrategyInfo AmpStrategy::GetPrimCastStrategyInfo(std::string op_name) {
  PrimCastStrategyInfo strategy_info = {PrimCastStrategy::Ignore, kFloat32, {}};
  // Check cache to improve performance
  if (strategy_info_cache.find(op_name) != strategy_info_cache.end()) {
    strategy_info = strategy_info_cache[op_name];
    MS_LOG(DEBUG) << "Cache hits, prim " << op_name << " amp strategy: " << strategy_info.strategy
                  << ", dtype: " << strategy_info.dtype << ", arg_pos: " << strategy_info.arg_pos;
    return strategy_info;
  }
  if (GetAmpLevel() == AmpLevel::O0) {
    strategy_info.strategy = PrimCastStrategy::Ignore;
  } else if (GetAmpLevel() == AmpLevel::O1) {
    auto iter = std::find_if(white_list_.begin(), white_list_.end(),
                             [op_name](const PrimArg &prim_arg) { return op_name == prim_arg.first; });
    if (iter != white_list_.end()) {
      strategy_info.strategy = PrimCastStrategy::DoCast;
      strategy_info.dtype = amp_dtype_;
      strategy_info.arg_pos = iter->second;
    } else {
      strategy_info.strategy = PrimCastStrategy::Ignore;
    }
  } else if (GetAmpLevel() == AmpLevel::O2) {
    strategy_info.strategy = PrimCastStrategy::DoCast;
    auto iter = std::find_if(black_list_.begin(), black_list_.end(),
                             [op_name](const PrimArg &prim_arg) { return op_name == prim_arg.first; });
    if (iter != black_list_.end()) {
      strategy_info.dtype = kFloat32;
      strategy_info.arg_pos = iter->second;
    } else {
      strategy_info.dtype = amp_dtype_;
    }
  } else if (GetAmpLevel() == AmpLevel::O3) {
    strategy_info.strategy = PrimCastStrategy::DoCast;
    strategy_info.dtype = amp_dtype_;
  } else if (GetAmpLevel() == AmpLevel::Auto) {
    do {
      auto iter = std::find_if(white_list_.begin(), white_list_.end(),
                               [op_name](const PrimArg &prim_arg) { return op_name == prim_arg.first; });
      if (iter != white_list_.end()) {
        strategy_info.strategy = PrimCastStrategy::DoCast;
        strategy_info.dtype = amp_dtype_;
        strategy_info.arg_pos = iter->second;
        break;
      }
      iter = std::find_if(black_list_.begin(), black_list_.end(),
                          [op_name](const PrimArg &prim_arg) { return op_name == prim_arg.first; });
      if (iter != black_list_.end()) {
        strategy_info.strategy = PrimCastStrategy::DoCast;
        strategy_info.dtype = kFloat32;
        strategy_info.arg_pos = iter->second;
        break;
      }
      iter = std::find_if(g_SetDtypeList.begin(), g_SetDtypeList.end(),
                          [op_name](const PrimArg &prim_arg) { return op_name == prim_arg.first; });
      if (iter != g_SetDtypeList.end()) {
        strategy_info.strategy = PrimCastStrategy::SetDtype;
        strategy_info.dtype = kFloat32;
        strategy_info.arg_pos = iter->second;
        break;
      }
      iter = std::find_if(g_SetDtypeOptList.begin(), g_SetDtypeOptList.end(),
                          [op_name](const PrimArg &prim_arg) { return op_name == prim_arg.first; });
      if (iter != g_SetDtypeOptList.end()) {
        strategy_info.strategy = PrimCastStrategy::SetDtypeOpt;
        strategy_info.dtype = kFloat32;
        strategy_info.arg_pos = iter->second;
        break;
      }
      iter = std::find_if(g_AutoPromoteList.begin(), g_AutoPromoteList.end(),
                          [op_name](const PrimArg &prim_arg) { return op_name == prim_arg.first; });
      if (iter != g_AutoPromoteList.end()) {
        strategy_info.strategy = PrimCastStrategy::AutoPromote;
        strategy_info.arg_pos = iter->second;
        break;
      }
      strategy_info.strategy = PrimCastStrategy::Ignore;
    } while (0);
  } else {
    MS_LOG(WARNING) << "Invalid amp level: " << GetAmpLevel() << ", ignore amp for op: " << op_name;
  }
  MS_LOG(DEBUG) << "Prim " << op_name << " amp strategy: " << strategy_info.strategy
                << ", dtype: " << strategy_info.dtype << ", arg_pos: " << strategy_info.arg_pos;
  // Cache strategy info to improve performance.
  strategy_info_cache[op_name] = strategy_info;
  return strategy_info;
}

AmpStrategyPtr CreateAmpStrategy(const AmpLevel amp_level, const TypePtr amp_dtype, const PrimArgList white_list,
                                 const PrimArgList black_list) {
  AmpStrategyPtr amp_strategy = std::make_shared<AmpStrategy>(amp_level, amp_dtype, white_list, black_list);
  return amp_strategy;
}

void PushAmpStratrgy(const AmpLevel amp_level, const TypePtr amp_dtype, const PrimArgList white_list,
                     const PrimArgList black_list) {
  AmpStrategyPtr amp_strategy = std::make_shared<AmpStrategy>(amp_level, amp_dtype, white_list, black_list);
  g_AmpStrategyStack.push(amp_strategy);
  MS_LOG(DEBUG) << "g_AmpStrategyStack size:" << g_AmpStrategyStack.size();
}

void PopAmpStrategy() {
  if (g_AmpStrategyStack.empty()) {
    MS_LOG(WARNING) << "g_AmpStrategyStack is empty when trying to pop the amp strategy.";
    return;
  }
  g_AmpStrategyStack.pop();
  MS_LOG(DEBUG) << "g_AmpStrategyStack size:" << g_AmpStrategyStack.size();
}

AmpStrategyPtr GetCurrentAmpStrategy() {
  if (g_AmpStrategyStack.empty()) {
    MS_LOG(INFO) << "amp strategy stack is empty";
    return nullptr;
  }
  return g_AmpStrategyStack.top();
}
}  // namespace amp

void RegAmpModule(py::module *m) {
  auto m_sub = m->def_submodule("amp", "auto mixed precision module");
  (void)m_sub.def("push_amp_strategy", &amp::PushAmpStratrgy,
                  "Push an auto mixed precision strategy into amp strategy stack");
  (void)m_sub.def("pop_amp_strategy", &amp::PopAmpStrategy,
                  "Pop an auto mixed precision strategy from amp strategy stack");
  (void)m_sub.def("get_curr_amp_strategy", &amp::GetCurrentAmpStrategy, "Get current amp strategy");
  (void)m_sub.def("create_amp_strategy", &amp::CreateAmpStrategy, "Create an auto mixed precision strategy");
  (void)py::enum_<amp::AmpLevel_>(*m_sub, "AmpLevel", py::arithmetic())
    .value("AmpO0", amp::AmpLevel_::O0)
    .value("AmpO1", amp::AmpLevel_::O1)
    .value("AmpO2", amp::AmpLevel_::O2)
    .value("AmpO3", amp::AmpLevel_::O3)
    .value("AmpAuto", amp::AmpLevel_::Auto);
  (void)py::class_<amp::AmpStrategy, std::shared_ptr<amp::AmpStrategy>>(*m_sub, "AmpStrategy")
    .def("get_amp_level", &amp::AmpStrategy::GetAmpLevel)
    .def("get_amp_dtype", &amp::AmpStrategy::GetAmpDtype)
    .def("get_white_list", &amp::AmpStrategy::GetWhiteList)
    .def("get_black_list", &amp::AmpStrategy::GetBlackList)
    .def("get_prim_cast_strategy_info", &amp::AmpStrategy::GetPrimCastStrategyInfo);
  (void)py::enum_<amp::PrimCastStrategy_>(*m_sub, "PrimCastStrategy", py::arithmetic())
    .value("AmpIgnore", amp::PrimCastStrategy_::Ignore)
    .value("AmpDoCast", amp::PrimCastStrategy_::DoCast)
    .value("AmpSetDtype", amp::PrimCastStrategy_::SetDtype)
    .value("AmpSetDtypeOpt", amp::PrimCastStrategy_::SetDtypeOpt)
    .value("AmpAutoPromote", amp::PrimCastStrategy_::AutoPromote);
  (void)py::class_<amp::PrimCastStrategyInfo_, std::shared_ptr<amp::PrimCastStrategyInfo_>>(*m_sub,
                                                                                            "PrimCastStrategyInfo")
    .def_readwrite("strategy", &amp::PrimCastStrategyInfo_::strategy)
    .def_readwrite("dtype", &amp::PrimCastStrategyInfo_::dtype)
    .def_readwrite("arg_pos", &amp::PrimCastStrategyInfo_::arg_pos);
  m_sub.attr("SetDtypeList") = &amp::g_SetDtypeList;
  m_sub.attr("SetDtypeOptList") = &amp::g_SetDtypeOptList;
  m_sub.attr("AutoPromoteList") = &amp::g_AutoPromoteList;
}
}  // namespace mindspore
