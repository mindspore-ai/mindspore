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

#ifndef MINDSPORE_CCSRC_PIPELINE_JIT_PARSE_PARSE_BASE_H_
#define MINDSPORE_CCSRC_PIPELINE_JIT_PARSE_PARSE_BASE_H_
#include <string>
#include <memory>
#include "pybind11/pybind11.h"
#include "ir/anf.h"
#include "ir/func_graph.h"
#include "ir/manager.h"
#include "pybind_api/export_flags.h"

namespace py = pybind11;
namespace mindspore {
namespace parse {
// define the node type
enum AstMainType : int64_t {
  AST_MAIN_TYPE_STMT = 0,       // ast.Stmt
  AST_MAIN_TYPE_EXPR = 1,       // ast.Expr
  AST_MAIN_TYPE_SLICE = 2,      // ast.Slice
  AST_MAIN_TYPE_UNKNOWN = 0xFF  // Error
};

enum AstSubType : int64_t {
  AST_SUB_TYPE_AND = 3,        // ast.And
  AST_SUB_TYPE_OR = 4,         // ast.Or
  AST_SUB_TYPE_NAME = 5,       // ast.Name
  AST_SUB_TYPE_TUPLE = 6,      // ast.Tuple
  AST_SUB_TYPE_SUBSCRIPT = 7,  // ast.Subscript
  AST_SUB_TYPE_STARRED = 8,    // ast.Starred
  AST_SUB_TYPE_ATTRIBUTE = 9,  // ast.Attribute
  AST_SUB_TYPE_UNKNOWN = 0xFF  // Error
};

// define the parse target type
enum ParseTargetTypeDef {
  PARSE_TARGET_FUNCTION = 0,         // function
  PARSE_TARGET_METHOD = 1,           // method
  PARSE_TARGET_OBJECT_INSTANCE = 2,  // object instance
  PARSE_TARGET_UNKNOW = 0xFF         // ERROR TYPE
};

// define python module name
const char PYTHON_MOD_PARSE_MODULE[] = "mindspore._extends.parse";
const char PYTHON_MOD_PARSE_OBJECT_FUNCTION[] = "parse_cb";
const char PYTHON_MOD_RESOLVE_FUNCTION[] = "resolve_symbol";
const char PYTHON_MOD_RESOLVE_GET_OBJ_KEY[] = "get_object_key";
const char PYTHON_MOD_PARSE_CHECK_IS_CLASS_MEMBER[] = "is_class_member";
const char PYTHON_MOD_RESOLVE_GET_OBJ_TYPE[] = "get_obj_type";
const char PYTHON_MOD_GET_OBJ_ID[] = "get_obj_id";
const char PYTHON_MOD_GET_CLASS_INSTANCE_TYPE[] = "get_class_instance_type";
const char PYTHON_MOD_CREATE_OBJ_INSTANCE[] = "create_obj_instance";
const char PYTHON_MOD_GET_DATACLASS_ATTRS[] = "get_dataclass_attributes";
const char PYTHON_MOD_GET_DATACLASS_METHODS[] = "get_dataclass_methods";
const char PYTHON_MOD_GET_MODULE_NAMESPACE[] = "get_module_namespace";
const char PYTHON_MOD_GET_MEMBER_NAMESPACE_SYMBOL[] = "get_class_member_namespace_symbol";
const char PYTHON_MOD_GET_PARSE_METHOD[] = "get_parse_method_of_class";
const char PYTHON_MOD_GET_BPROP_METHOD[] = "get_bprop_method_of_class";
const char PYTHON_MOD_GET_OBJECT_DESCRIPTION[] = "get_object_description";
const char PYTHON_MOD_CONVERT_TO_MS_TENSOR[] = "convert_to_ms_tensor";

const char PYTHON_PARSE_GET_ARGS[] = "get_args";
const char PYTHON_PARSE_GET_ARGS_DEFAULT_VALUES[] = "get_args_default_values";
const char PYTHON_PARSE_GET_NODE_TYPE[] = "get_node_type";
const char PYTHON_PARSE_GET_AST_TYPE[] = "get_ast_type";
const char PYTHON_PARSE_GET_NAMESPACE_SYMBOL[] = "get_namespace_symbol";
const char PYTHON_PARSE_GET_AST_NAMESPACE_SYMBOL[] = "get_ast_namespace_symbol";
const char PYTHON_PARSE_GET_OPERATION_NAMESPACE_SYMBOL[] = "get_operation_namespace_symbol";
const char PYTHON_PARSE_GET_LOCATION[] = "get_location";
const char PYTHON_PARSE_EXPAND_EXPR_STATEMENT[] = "expand_expr_statement";
const char PYTHON_PARSE_GENERATE_SCOPE[] = "generate_scope";
const char PYTHON_PARSE_GET_SCOPE_NAME[] = "get_scope_name";
const char PYTHON_PARSE_ANALYZE_SUPER[] = "analyze_super";

const char PYTHON_PARSE_CLASS_SLICE[] = "create_slice_obj";
const char PYTHON_PARSE_CLASS_ELLIPSIS[] = "create_ellipsis_obj";

// define the common name
const char NAMED_PRIMITIVE_LEN[] = "len";
const char NAMED_PRIMITIVE_BODY[] = "body";
const char NAMED_PRIMITIVE_ASSIGN[] = "Assign";
const char NAMED_PRIMITIVE_AUGASSIGN[] = "AugAssign";
const char NAMED_PRIMITIVE_FOR[] = "For";
const char NAMED_PRIMITIVE_IF[] = "If";
const char NAMED_PRIMITIVE_WHILE[] = "While";
const char NAMED_PRIMITIVE_VALUE[] = "value";
const char NAMED_PRIMITIVE_FUNC[] = "func";
const char NAMED_PRIMITIVE_TEST[] = "test";
const char NAMED_PRIMITIVE_LEFT[] = "left";
const char NAMED_PRIMITIVE_CALL[] = "Call";
const char NAMED_PRIMITIVE_SUBSCRIPT[] = "Subscript";
const char NAMED_PRIMITIVE_ATTRIBUTE[] = "Attribute";
const char NAMED_PRIMITIVE_COMPARE[] = "Compare";
const char NAMED_PRIMITIVE_NAMECONSTANT[] = "NameConstant";
const char NAMED_PRIMITIVE_COMPARATORS[] = "comparators";
const char NAMED_PRIMITIVE_TARGET[] = "target";
const char NAMED_PRIMITIVE_SLICE[] = "slice";
const char NAMED_PRIMITIVE_NAME[] = "Name";
const char NAMED_PRIMITIVE_NUM[] = "Num";
const char NAMED_PRIMITIVE_STR[] = "Str";
const char NAMED_PRIMITIVE_ITER[] = "iter";
const char NAMED_PRIMITIVE_NEXT[] = "next";
const char NAMED_PRIMITIVE_GETITEM[] = "getitem";
const char NAMED_PRIMITIVE_SETITEM[] = "setitem";
const char NAMED_PRIMITIVE_HASNEXT[] = "hasnext";
const char NAMED_PRIMITIVE_BOOL[] = "bool_";  // bool: P.identity
const char NAMED_PRIMITIVE_MAKETUPLE[] = "MakeTuple";
const char NAMED_PRIMITIVE_MAKELIST[] = "make_list";
const char NAMED_PRIMITIVE_MAKESLICE[] = "make_slice";
const char NAMED_PRIMITIVE_MAKEDICT[] = "make_dict";
const char NAMED_METAGRAPH_UNPACKCALL[] = "unpack_call";

// define NAMED_PRIMITIVE_GETATTR "getattr"
// define python inline attr
const char PYTHON_GET_METHOD_LEN[] = "__len__";
const char PYTHON_GET_METHOD_SELF_CLASS[] = "__self__";
const char PYTHON_GET_OBJ_DESC[] = "__str__";

const char PYTHON_EXTERN_PARSE_METHOD[] = "__parse_method__";
const char PYTHON_EXTERN_MINDSPORE_FLAG[] = "_mindspore_flags";

// define the parse constant
const int64_t MAX_COMPARISON_OPS_SUPPORTED = 1;
const char CUSTOM_BPROP_NAME[] = "bprop";
const char STAGE_NAME[] = "pipeline_stage";

// define the Namespace name
const char RESOLVE_NAMESPACE_NAME_AST[] = "Ast";                   // for ast type namespace
const char RESOLVE_NAMESPACE_NAME_CLASS_MEMBER[] = "ClassMember";  // for class member namespace
const char RESOLVE_NAMESPACE_NAME_SYMBOL_STR[] = "SymbolStr";      // for symbol str namespace
const char RESOLVE_NAMESPACE_NAME_COMMON_OPS[] = "CommonOPS";      // for common ops, eg: hasnext, next
const char RESOLVE_NAMESPACE_NAME_MODULE[] = "Module";             // fro Module namespace

// define Resolve type
enum ResolveTypeDef : int64_t {
  RESOLVE_TYPE_NONE = 0,            // resolve None
  RESOLVE_TYPE_FUNCTION = 1,        // resolve function
  RESOLVE_TYPE_METHOD = 2,          // resolve class method
  RESOLVE_TYPE_CLASS_TYPE = 3,      // resolve class type
  RESOLVE_TYPE_CLASS_INSTANCE = 4,  // resolve the class instance of common class
  RESOLVE_TYPE_INVALID = 0xFF       // resolve invalid
};

// define the class instance detail type When the type is RESOLVE_TYPE_CLASS_INSTANCE
enum ClassInstanceTypeDef {
  CLASS_INSTANCE_TYPE_CELL = 0,       // class instance type is Cell
  CLASS_INSTANCE_TYPE_PRIMITIVE = 1,  // class instance type is Primitive
  CLASS_INSTANCE_TYPE_INVALID = 0xFF
};

// Convert python object to ValuePtr
bool ConvertData(const py::object &obj, ValuePtr *data, bool use_signature = false, TypePtr dtype = nullptr);

// Convert python obj to graph
FuncGraphPtr ConvertToFuncGraph(const py::object &obj,
                                const std::string &python_mod_get_parse_method = PYTHON_MOD_GET_PARSE_METHOD);

// Parse the python object to graph
FuncGraphPtr ParsePythonCode(const py::object &obj,
                             const std::string &python_mod_get_parse_method = PYTHON_MOD_GET_PARSE_METHOD);
// add wrap for cell top graph.
FuncGraphPtr MakeTopGraph(const py::object &cell, const ValuePtr &cell_ptr);
}  // namespace parse
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PIPELINE_JIT_PARSE_PARSE_BASE_H_
