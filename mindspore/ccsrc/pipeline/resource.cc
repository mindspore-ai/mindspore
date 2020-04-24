/**
 * This is the C++ adaptation and derivative work of Myia (https://github.com/mila-iqia/myia/).
 *
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

#include "pipeline/resource.h"
#include "pipeline/pipeline.h"
#include "pipeline/static_analysis/static_analysis.h"
#include "debug/draw.h"
#include "debug/trace.h"
#include "ir/dtype.h"
#include "pipeline/parse/data_converter.h"
#include "operator/ops.h"
#include "utils/graph_utils.h"
#include "optimizer/ad/dfunctor.h"
#include "vm/segment_runner.h"

namespace mindspore {
// namespace to support opmap definition
namespace pipeline {

MethodMap &GetMethodMap() {
  static MethodMap method_map = {{kObjectTypeString,
                                  {
                                    {"__bool__", std::string("str_bool")}  // C.str_bool
                                  }},
                                 {kMetaTypeNone,
                                  {
                                    {"__bool__", std::string("none_bool")}  // C.none_bool
                                  }},
                                 {kNumberTypeBool,
                                  {
                                    {"__and__", prim::kPrimBoolAnd},     // P.bool_and
                                    {"__or__", prim::kPrimBoolOr},       // P.bool_or
                                    {"__eq__", prim::kPrimBoolEq},       // P.bool_eq
                                    {"__ne__", std::string("bool_ne")},  // C.bool_ne
                                    {"__bool__", prim::kPrimIdentity}    // P.identity
                                  }},
                                 {kNumberTypeInt,
                                  {
                                    {"__add__", prim::kPrimScalarAdd},              // P.scalar_add
                                    {"__sub__", prim::kPrimScalarSub},              // P.scalar_sub
                                    {"__mul__", prim::kPrimScalarMul},              // P.scalar_mul
                                    {"__floordiv__", std::string("int_floordiv")},  // C.int_floordiv
                                    {"__truediv__", std::string("int_truediv")},    // C.int_truediv
                                    {"__mod__", prim::kPrimScalarMod},              // P.scalar_mod
                                    {"__pow__", prim::kPrimScalarPow},              // P.scalar_pow
                                    {"__floor__", prim::kPrimIdentity},             // P.identity
                                    {"__trunc__", prim::kPrimIdentity},             // P.identity
                                    {"__pos__", prim::kPrimScalarUadd},             // P.scalar_uadd
                                    {"__neg__", prim::kPrimScalarUsub},             // P.scalar_usub
                                    {"__eq__", prim::kPrimScalarEq},                // P.scalar_eq
                                    {"__ne__", prim::kPrimScalarNe},                // P.scalar_ne
                                    {"__lt__", prim::kPrimScalarLt},                // P.scalar_lt
                                    {"__gt__", prim::kPrimScalarGt},                // P.scalar_gt
                                    {"__le__", prim::kPrimScalarLe},                // P.scalar_le
                                    {"__ge__", prim::kPrimScalarGe},                // P.scalar_ge
                                    {"__bool__", std::string("int_bool")},          // C.int_bool
                                    {"__ms_to_array__", prim::kPrimScalarToArray},  // P.scalar_to_array
                                  }},
                                 {kNumberTypeUInt,
                                  {
                                    {"__add__", prim::kPrimScalarAdd},              // P.scalar_add,
                                    {"__sub__", prim::kPrimScalarSub},              // P.scalar_sub,
                                    {"__mul__", prim::kPrimScalarMul},              // P.scalar_mul,
                                    {"__floordiv__", prim::kPrimScalarDiv},         // P.scalar_div,
                                    {"__truediv__", std::string("int_truediv")},    // C.int_truediv
                                    {"__mod__", prim::kPrimScalarMod},              // P.scalar_mod,
                                    {"__pow__", prim::kPrimScalarPow},              // P.scalar_pow,
                                    {"__floor__", prim::kPrimIdentity},             // P.identity,
                                    {"__trunc__", prim::kPrimIdentity},             // P.identity,
                                    {"__pos__", prim::kPrimScalarUadd},             // P.scalar_uadd,
                                    {"__neg__", prim::kPrimScalarUsub},             // P.scalar_usub,
                                    {"__eq__", prim::kPrimScalarEq},                // P.scalar_eq,
                                    {"__ne__", prim::kPrimScalarNe},                // P.scalar_ne,
                                    {"__lt__", prim::kPrimScalarLt},                // P.scalar_lt,
                                    {"__gt__", prim::kPrimScalarGt},                // P.scalar_gt,
                                    {"__le__", prim::kPrimScalarLe},                // P.scalar_le,
                                    {"__ge__", prim::kPrimScalarGe},                // P.scalar_ge,
                                    {"__bool__", std::string("int_bool")},          // C.int_bool
                                    {"__ms_to_array__", prim::kPrimScalarToArray},  // P.scalar_to_array,
                                  }},
                                 {kNumberTypeFloat,
                                  {
                                    {"__add__", prim::kPrimScalarAdd},                // P.scalar_add,
                                    {"__sub__", prim::kPrimScalarSub},                // P.scalar_sub,
                                    {"__mul__", prim::kPrimScalarMul},                // P.scalar_mul,
                                    {"__floordiv__", std::string("float_floordiv")},  // C.float_floordiv
                                    {"__truediv__", prim::kPrimScalarDiv},            // P.scalar_div,
                                    {"__mod__", prim::kPrimScalarMod},                // P.scalar_mod,
                                    {"__pow__", prim::kPrimScalarPow},                // P.scalar_pow,
                                    {"__floor__", prim::kPrimScalarFloor},            // P.scalar_floor,
                                    {"__trunc__", prim::kPrimScalarTrunc},            // P.scalar_trunc,
                                    {"__pos__", prim::kPrimScalarUadd},               // P.scalar_uadd,
                                    {"__neg__", prim::kPrimScalarUsub},               // P.scalar_usub,
                                    {"__eq__", prim::kPrimScalarEq},                  // P.scalar_eq,
                                    {"__ne__", prim::kPrimScalarNe},                  // P.scalar_ne,
                                    {"__lt__", prim::kPrimScalarLt},                  // P.scalar_lt,
                                    {"__gt__", prim::kPrimScalarGt},                  // P.scalar_gt,
                                    {"__le__", prim::kPrimScalarLe},                  // P.scalar_le,
                                    {"__ge__", prim::kPrimScalarGe},                  // P.scalar_ge,
                                    {"__bool__", std::string("float_bool")},          // C.float_bool
                                    {"__ms_to_array__", prim::kPrimScalarToArray},    // P.scalar_to_array,
                                  }},
                                 {kObjectTypeTuple,
                                  {
                                    {"__len__", prim::kPrimTupleLen},                  // P.tuple_len,
                                    {"__getitem__", prim::kPrimTupleGetItem},          // P.tuple_getitem,
                                    {"__setitem__", prim::kPrimTupleSetItem},          // P.tuple_setitem,
                                    {"__ms_iter__", prim::kPrimIdentity},              // P.identity,
                                    {"__ms_next__", std::string("tuple_next")},        // C.tuple_next,
                                    {"__ms_hasnext__", std::string("tuple_hasnext")},  // C.tuple_hasnext
                                    {"__bool__", std::string("tuple_bool")}            // C.tuple_bool
                                  }},
                                 {kObjectTypeList,
                                  {
                                    {"__len__", prim::kPrimListLen},            // P.list_len,
                                    {"__getitem__", prim::kPrimListGetItem},    // P.list_getitem,
                                    {"__setitem__", prim::kPrimListSetItem},    // P.list_setitem,
                                    {"__ms_iter__", prim::kPrimIdentity},       // P.identity
                                    {"__ms_next__", std::string("list_next")},  // C.list_next
                                    {"append", std::string("list_append")},     // C.list_next
                                    {"__bool__", std::string("list_bool")},     // C.list_bool
                                    {"__ms_hasnext__", std::string("list_hasnext")},
                                  }},
                                 {kObjectTypeDictionary,
                                  {
                                    {"__len__", prim::kPrimDictLen},          // P.dict_len
                                    {"__getitem__", prim::kPrimDictGetItem},  // P.dict_getitem
                                    {"__setitem__", prim::kPrimDictSetItem},  // P.dict_setitem,
                                    {"__bool__", std::string("dict_bool")}    // C.dict_bool
                                  }},
                                 {kObjectTypeTensorType,
                                  {
                                    {"__add__", std::string("add")},             // C.add
                                    {"__sub__", std::string("sub")},             // C.sub
                                    {"__mul__", std::string("mul")},             // C.mul
                                    {"__truediv__", std::string("truediv")},     // C.truediv
                                    {"__floordiv__", std::string("floordiv")},   // C.floordiv
                                    {"__mod__", std::string("mod")},             // C.mod
                                    {"__pow__", std::string("pow_")},            // C.pow
                                    {"__floor__", std::string("array_floor")},   // C.array_floor
                                    {"__trunc__", std::string("array_trunc")},   // C.array_trunc
                                    {"__pos__", std::string("array_uadd")},      // C.array_uadd
                                    {"__neg__", std::string("array_usub")},      // C.array_usub
                                    {"__eq__", std::string("eq")},               // C.eq
                                    {"__ne__", std::string("ne")},               // C.ne
                                    {"__lt__", std::string("lt")},               // C.lt
                                    {"__gt__", std::string("gt")},               // C.gt
                                    {"__le__", std::string("le")},               // C.le
                                    {"__ge__", std::string("ge")},               // C.ge
                                    {"__matmul__", prim::kPrimDot},              // P.dot,
                                    {"__len__", prim::kPrimArrayLen},            // P.array_len,
                                    {"__getitem__", prim::kPrimArrayGetItem},    // P.array_getitem,
                                    {"__setitem__", prim::kPrimArraySetItem},    // P.array_setitem,
                                    {"__ms_iter__", std::string("array_iter")},  // C.array_iter
                                    {"__ms_to_array__", prim::kPrimIdentity},    // P.identity,
                                    {"item", prim::kPrimArrayToScalar},          // P.array_to_scalar,
                                    {"transpose", std::string("transpose")},     // P.transpose
                                    {"__bool__", std::string("tensor_bool")},    // C.tensor_bool
                                  }},
                                 {kObjectTypeJTagged, {}},
                                 {kObjectTypeSymbolicKeyType, {}},
                                 {kObjectTypeEnvType, {}}};
  return method_map;
}

Resource::Resource(const py::object &obj)
    : engine_(std::make_shared<abstract::AnalysisEngine>(abstract::GetPrimEvaluatorConstructors(), manager_)),
      input_(obj),
      is_cleaned_(false) {}

Resource::~Resource() {
  MS_LOG(DEBUG) << "Resource clear";

  // If exit normally, these global variables will be cleaned
  // in Resource::Clean call by MsPipeline::Compile, but if exit with MS_LOGEXCEPTION,
  // these global variables may not being cleaned, it may
  // cause segmentfault when free python object inside these global variables
  // after python interpreter got freed, so these global variables
  // are cleaned here.
  // So if exit normally, these global variable will be cleaned twice,
  // care be taken to prevent double free in the following functions.
  if (!is_cleaned_) {
    try {
      Clean();
    } catch (const std::exception &e) {
      MS_LOG(ERROR) << "Exception when cleaning resource. Error info " << e.what();
    } catch (...) {
      MS_LOG(ERROR) << "Exception when cleaning resource.";
    }
  }
}

bool Resource::IsTypeInMethodMap(const TypeId &type) {
  TypeId type_id = NormalizeTypeId(type);
  const MethodMap &method_map = GetMethodMap();
  auto iter = method_map.find(static_cast<int>(type_id));
  if (iter != method_map.end()) {
    return true;
  }
  return false;
}

Any Resource::GetMethodPtr(const TypeId &type, const std::string &name) {
  TypeId type_id = NormalizeTypeId(type);
  const MethodMap &method_map = GetMethodMap();
  auto iter = method_map.find(static_cast<int>(type_id));
  if (iter == method_map.end()) {
    MS_LOG(WARNING) << "Object type: " << type_id << " not in the method_map";
    return Any();
  }

  auto iter_map = iter->second.find(name);
  if (iter_map == iter->second.end()) {
    MS_LOG(WARNING) << "Object type: " << type_id << " have no method: " << name;
    return Any();
  }
  return iter_map->second;
}

void Resource::Clean() {
  // AbstractTensor->elements() will be saved in AbstractBasePtrList
  args_spec_.clear();
  input_ = py::none();
  // Context with AbstractBasePtrList may be saved in GraphEvaluator
  // some Evaluator like ResolveEvaluator may save Python object in cache,
  // it should be cleaned before Python Interpreter destructed.
  MS_EXCEPTION_IF_NULL(engine_);
  engine_->ClearEvaluatorCache();
  // clean static variable to prevent from crash. As static variable is released after
  // Python threads is released.
  parse::data_converter::ClearObjectCache();
  parse::Parser::CleanParserResource();
  parse::CleanDataClassToClassMap();
  trace::ClearTraceStack();
  is_cleaned_ = true;
}
}  // namespace pipeline
}  // namespace mindspore
