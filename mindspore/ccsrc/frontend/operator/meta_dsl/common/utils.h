/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_FRONTEND_OPERATOR_META_DSL_COMMON_UTILS_H_
#define MINDSPORE_CCSRC_FRONTEND_OPERATOR_META_DSL_COMMON_UTILS_H_

#include <string>
#include <map>
#include <set>
#include <memory>
#include "utils/ms_utils.h"
#include "include/utils/utils.h"
#include "ir/anf.h"
#include "ir/meta_func_graph.h"
#include "ir/core_ops_primitive.h"

namespace mindspore::prim {
#define EXPAND_PARAMS(...) __VA_ARGS__

#define DECLARE_PARAM(param, index) NodePtr param = NewParam(#param, index);

#define DECLARE_PARAMS_0()

#define DECLARE_PARAMS_1(p1) DECLARE_PARAM(p1, 1)

#define DECLARE_PARAMS_2(p1, p2) \
  DECLARE_PARAM(p1, 1);          \
  DECLARE_PARAM(p2, 2)

#define DECLARE_PARAMS_3(p1, p2, p3) \
  DECLARE_PARAMS_2(p1, p2);          \
  DECLARE_PARAM(p3, 3)

#define DECLARE_PARAMS_4(p1, p2, p3, p4) \
  DECLARE_PARAMS_3(p1, p2, p3);          \
  DECLARE_PARAM(p4, 4)

#define DECLARE_PARAMS_5(p1, p2, p3, p4, p5) \
  DECLARE_PARAMS_4(p1, p2, p3, p4);          \
  DECLARE_PARAM(p5, 5)

#define DECLARE_PARAMS_6(p1, p2, p3, p4, p5, p6) \
  DECLARE_PARAMS_5(p1, p2, p3, p4, p5);          \
  DECLARE_PARAM(p6, 6)

#define DECLARE_PARAMS_7(p1, p2, p3, p4, p5, p6, p7) \
  DECLARE_PARAMS_6(p1, p2, p3, p4, p5, p6);          \
  DECLARE_PARAM(p7, 7)

#define DECLARE_PARAMS_8(p1, p2, p3, p4, p5, p6, p7, p8) \
  DECLARE_PARAMS_7(p1, p2, p3, p4, p5, p6, p7);          \
  DECLARE_PARAM(p8, 8)

#define DECLARE_PARAMS_9(p1, p2, p3, p4, p5, p6, p7, p8, p9) \
  DECLARE_PARAMS_8(p1, p2, p3, p4, p5, p6, p7, p8);          \
  DECLARE_PARAM(p9, 9)

#define DECLARE_PARAMS_10(p1, p2, p3, p4, p5, p6, p7, p8, p9, p10) \
  DECLARE_PARAMS_9(p1, p2, p3, p4, p5, p6, p7, p8, p9);            \
  DECLARE_PARAM(p10, 10)

#define DECLARE_PARAMS_11(p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11) \
  DECLARE_PARAMS_10(p1, p2, p3, p4, p5, p6, p7, p8, p9, p10);           \
  DECLARE_PARAM(p11, 11)

#define DECLARE_PARAMS_12(p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12) \
  DECLARE_PARAMS_11(p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11);           \
  DECLARE_PARAM(p12, 12)

#define DECLARE_PARAMS_13(p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13) \
  DECLARE_PARAMS_12(p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12);           \
  DECLARE_PARAM(p13, 13)

#define DECLARE_PARAMS_14(p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14) \
  DECLARE_PARAMS_13(p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13);           \
  DECLARE_PARAM(p14, 14)

#define DECLARE_PARAMS_15(p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15) \
  DECLARE_PARAMS_14(p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14);           \
  DECLARE_PARAM(p15, 15)

#define DECLARE_PARAMS_16(p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15, p16) \
  DECLARE_PARAMS_15(p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15);           \
  DECLARE_PARAM(p16, 16)

#define DECLARE_PARAMS_17(p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15, p16, p17) \
  DECLARE_PARAMS_16(p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15, p16);           \
  DECLARE_PARAM(p17, 17)

#define DECLARE_PARAMS_18(p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15, p16, p17, p18) \
  DECLARE_PARAMS_17(p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15, p16, p17);           \
  DECLARE_PARAM(p18, 18)

#define DECLARE_PARAMS_19(p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15, p16, p17, p18, p19) \
  DECLARE_PARAMS_18(p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15, p16, p17, p18);           \
  DECLARE_PARAM(p19, 19)

#define DECLARE_PARAMS_20(p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15, p16, p17, p18, p19, p20) \
  DECLARE_PARAMS_19(p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15, p16, p17, p18, p19);           \
  DECLARE_PARAM(p20, 20)

#define GET_DECLARE_PARAMS_MACRO(_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17, _18, _19, \
                                 _20, NAME, ...)                                                                       \
  NAME

// DECLARE_PARAMS support 1 to 20 parameters.
#define DECLARE_PARAMS(...)                                                                                           \
  _EXPAND(GET_DECLARE_PARAMS_MACRO(                                                                                   \
    __VA_ARGS__, DECLARE_PARAMS_20, DECLARE_PARAMS_19, DECLARE_PARAMS_18, DECLARE_PARAMS_17, DECLARE_PARAMS_16,       \
    DECLARE_PARAMS_15, DECLARE_PARAMS_14, DECLARE_PARAMS_13, DECLARE_PARAMS_12, DECLARE_PARAMS_11, DECLARE_PARAMS_10, \
    DECLARE_PARAMS_9, DECLARE_PARAMS_8, DECLARE_PARAMS_7, DECLARE_PARAMS_6, DECLARE_PARAMS_5, DECLARE_PARAMS_4,       \
    DECLARE_PARAMS_3, DECLARE_PARAMS_2, DECLARE_PARAMS_1, DECLARE_PARAMS_0)(__VA_ARGS__))

#define _REGISTER_META_IMPL(name)      \
  const auto meta_impl_helper_##name = \
    RegMetaImplFactory::RegHelper(#name, []() { return std::make_shared<name##MetaImpl>(); })

#define _REGISTER_META_IMPL_WITH_CHECK(name, check_func) \
  const auto meta_impl_helper_##name =                   \
    RegMetaImplFactory::RegHelper(#name, []() { return std::make_shared<name##MetaImpl>(); }, check_func)

// Definition of MetaImpl subclass.
#define _DEFINE_META_IMPL(name)                 \
  class name##MetaImpl : public MetaImpl {      \
   public:                                      \
    name##MetaImpl() : MetaImpl(#name) {}       \
    ~name##MetaImpl() override = default;       \
    MS_DECLARE_PARENT(name##MetaImpl, MetaImpl) \
    void GenerateFunction() override;           \
  }

// DEFINE_FUNCTION_OP(op_name) -> _DEFINE_FUNCTION_OP
#define _DEFINE_FUNCTION_OP(name) \
  _DEFINE_META_IMPL(name);        \
  _REGISTER_META_IMPL(name)

// DEFINE_FUNCTION_OP(op_name, check_func) -> _DEFINE_FUNCTION_OP_WITH_CHECK
#define _DEFINE_FUNCTION_OP_WITH_CHECK(name, check_func) \
  _DEFINE_META_IMPL(name);                               \
  _REGISTER_META_IMPL_WITH_CHECK(name, check_func)

#define _EXPAND(x) x

#define _GET_FUNCTION_OP_MACRO(_1, _2, NAME, ...) NAME

// Define REGISTER_META_IMPL api.
#define REGISTER_META_IMPL(...) \
  _EXPAND(_GET_FUNCTION_OP_MACRO(__VA_ARGS__, _REGISTER_META_IMPL_WITH_CHECK, _REGISTER_META_IMPL)(__VA_ARGS__))

// Define REGISTER_FUNCTION_OP api.
#define REGISTER_FUNCTION_OP(...) \
  _EXPAND(_GET_FUNCTION_OP_MACRO(__VA_ARGS__, _DEFINE_FUNCTION_OP_WITH_CHECK, _DEFINE_FUNCTION_OP)(__VA_ARGS__))

// Define PRIMITIVE_BPROP_REG api.
#define PRIMITIVE_BPROP_REG(name, bprop)     \
  const auto bprop_meta_impl_helper_##name = \
    RegMetaImplFactory::RegBpropHelper(kPrim##name, []() { return std::make_shared<bprop##MetaImpl>(); });

#define BeginFunction(name, ...)            \
  void name##MetaImpl::GenerateFunction() { \
    do {                                    \
    _EXPAND(DECLARE_PARAMS(__VA_ARGS__))

#define EndFunction(name)       \
  /* Used with BeginFunction */ \
  }                             \
  while (0);                    \
  }
}  // namespace mindspore::prim
#endif  // MINDSPORE_CCSRC_FRONTEND_OPERATOR_META_DSL_COMMON_UTILS_H_
