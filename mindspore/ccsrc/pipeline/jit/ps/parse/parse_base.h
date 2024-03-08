/**
 * Copyright 2019-2024 Huawei Technologies Co., Ltd
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
// Define the node type.
enum AstMainType : int64_t {
  AST_MAIN_TYPE_STMT = 0,       // ast.Stmt
  AST_MAIN_TYPE_EXPR = 1,       // ast.Expr
  AST_MAIN_TYPE_SLICE = 2,      // ast.Slice
  AST_MAIN_TYPE_UNKNOWN = 0xFF  // Unknown type
};

enum AstSubType : int64_t {
  AST_SUB_TYPE_AND = 3,         // ast.And
  AST_SUB_TYPE_OR = 4,          // ast.Or
  AST_SUB_TYPE_NAME = 5,        // ast.Name
  AST_SUB_TYPE_TUPLE = 6,       // ast.Tuple
  AST_SUB_TYPE_LIST = 7,        // ast.List
  AST_SUB_TYPE_SUBSCRIPT = 8,   // ast.Subscript
  AST_SUB_TYPE_STARRED = 9,     // ast.Starred
  AST_SUB_TYPE_ATTRIBUTE = 10,  // ast.Attribute
  AST_SUB_TYPE_DICT = 11,       // ast.Dict
  AST_SUB_TYPE_UNKNOWN = 0xFF   // Unknown type
};

// Define the parse target type.
enum ParseTargetType {
  PARSE_TARGET_FUNCTION = 0,         // Function
  PARSE_TARGET_METHOD = 1,           // Method
  PARSE_TARGET_OBJECT_INSTANCE = 2,  // Object instance
  PARSE_TARGET_UNKNOW = 0xFF         // Unknown type
};

// Define python module name.
const char PYTHON_MOD_MODULE[] = "mindspore";
const char PYTHON_MOD_PARSE_MODULE[] = "mindspore._extends.parse";
const char PYTHON_MOD_PRIMITIVE_ARG_HANDLER_MODULE[] = "mindspore.ops.auto_generate.gen_arg_handler";
const char PYTHON_MOD_PRIMITIVE_ARG_DTYPE_CAST_MODULE[] = "mindspore.ops.auto_generate.gen_arg_dtype_cast";
const char PYTHON_MOD_PRIMITIVE_OP_CREATE_INSTANCE_HELPER_MODULE[] =
  "mindspore.ops.auto_generate.cpp_create_prim_instance_helper";
const char PYTHON_MOD_PRIMITIVE_OP_TYPE_CAST[] = "do_type_cast";
const char PYTHON_MOD_PRIMITIVE_OP_LABELS_DICT[] = "op_labels";
const char PYTHON_MOD_PRIMITIVE_OP_DEFAULT_VALUE_DICT[] = "op_args_default_value";
const char PYTHON_MOD_PARSE_OBJECT_FUNCTION[] = "parse_cb";
const char PYTHON_MOD_RESOLVE_FUNCTION[] = "resolve_symbol";
const char PYTHON_MOD_RESOLVE_GET_OBJ_KEY[] = "get_object_key";
const char PYTHON_MOD_PARSE_CHECK_IS_CLASS_MEMBER_OF_SELF[] = "is_class_member_of_self";
const char PYTHON_MOD_PARSE_CHECK_IS_CLASS_MEMBER_RECURSIVE[] = "is_class_member_recursive";
const char PYTHON_MOD_RESOLVE_GET_OBJ_TYPE[] = "get_obj_type";
const char PYTHON_MOD_GET_OBJ_ID[] = "get_obj_id";
const char PYTHON_MOD_GET_CLASS_INSTANCE_TYPE[] = "get_class_instance_type";
const char PYTHON_MOD_CREATE_INSTANCE[] = "create_instance";
const char PYTHON_MOD_IS_SUPPORTED_CREATE_INSTANCE_TYPE[] = "is_supported_create_instance_type";
const char PYTHON_MOD_IS_CLASS_TYPE[] = "is_class_type";
const char PYTHON_MOD_GET_ADAPTER_TENSOR_ATTR[] = "get_adapter_tensor_attr";
const char PYTHON_MOD_IS_ADAPTER_TENSOR_CLASS[] = "is_adapter_tensor_class";
const char PYTHON_MOD_IS_ADAPTER_PARAMETER_CLASS[] = "is_adapter_parameter_class";
const char PYTHON_MOD_GET_MS_CLASS_NAME[] = "get_ms_class_name";
const char PYTHON_MOD_GET_MODULE_NAMESPACE[] = "get_module_namespace";
const char PYTHON_MOD_GET_MEMBER_NAMESPACE_SYMBOL[] = "get_class_member_namespace_symbol";
const char PYTHON_MOD_GET_OBJ_DEFINED[] = "get_obj_defined_from_obj_type";
const char PYTHON_MOD_GET_ATTR_FROM_OBJ[] = "get_attr_from_object";
const char PYTHON_MOD_GET_PARSE_METHOD[] = "get_parse_method_of_class";
const char PYTHON_MOD_GET_BPROP_METHOD[] = "get_bprop_method_of_class";
const char PYTHON_MOD_GET_OBJECT_DESCRIPTION[] = "get_object_description";
const char PYTHON_MOD_IS_CELL_LIST[] = "is_cell_list";
const char PYTHON_MOD_CONVERT_CELL_LIST_TO_SEQUENCE[] = "convert_cell_list_to_sequence";
const char PYTHON_MOD_GET_ITEM_FROM_SEQUENCE[] = "get_obj_from_sequence";
const char PYTHON_MOD_CONVERT_TO_MS_TENSOR[] = "convert_to_ms_tensor";
const char PYTHON_MOD_CONVERT_TO_MS_CSRTENSOR[] = "convert_to_ms_csrtensor";
const char PYTHON_MOD_CONVERT_TO_MS_COOTENSOR[] = "convert_to_ms_cootensor";
const char PYTHON_MOD_CONVERT_TO_NAMEDTUPLE[] = "convert_to_namedtuple";
const char PYTHON_MOD_EVAL_PY_SCRIPT[] = "eval_script";
const char PYTHON_MOD_GET_SCRIPT_ID_ATTRS[] = "get_script_id_attrs";
const char PYTHON_MOD_PYTHON_ISINSTANCE[] = "python_isinstance";
const char PYTHON_MOD_MS_ISINSTANCE[] = "ms_isinstance";
const char PYTHON_MOD_CONVERT_CLASS_TO_FUNCTION[] = "convert_class_to_function";
const char PYTHON_MOD_GET_CONST_ABS[] = "get_const_abs";
const char PYTHON_MOD_GET_CONST_ROUND[] = "get_const_round";
const char PYTHON_MOD_GET_CONST_LEN[] = "get_const_len";
const char PYTHON_MOD_CHECK_ATTRS[] = "check_attrs";
const char PYTHON_MOD_CHECK_IS_SUBCLASS[] = "check_is_subclass";
const char PYTHON_MOD_GET_METHOD_INFO[] = "get_method_info";
const char PYTHON_MOD_IS_MS_TENSOR_METHOD[] = "is_ms_tensor_method";
const char PYTHON_MOD_CAN_CONSTANT_FOLD[] = "can_constant_fold";

const char PYTHON_PARSE_GET_ARGS[] = "get_args";
const char PYTHON_PARSE_GET_ARGS_DEFAULT_VALUES[] = "get_args_default_values";
const char PYTHON_PARSE_GET_NODE_TYPE[] = "get_node_type";
const char PYTHON_PARSE_GET_AST_TYPE[] = "get_ast_type";
const char PYTHON_PARSE_GET_AST_NAMESPACE_SYMBOL[] = "get_ast_namespace_symbol";
const char PYTHON_PARSE_GET_OPERATION_SYMBOL[] = "get_operation_symbol";
const char PYTHON_PARSE_GET_OPERATION_NAMESPACE_SYMBOL[] = "get_operation_namespace_symbol";
const char PYTHON_PARSE_GET_CLASS_TENSOR_TYPE[] = "get_tensor_class_type";
const char PYTHON_PARSE_GET_NAMESPACE_SYMBOL[] = "get_namespace_symbol";
const char PYTHON_PARSE_IS_BUILTIN_FUNCTION_NAME[] = "is_builtin_function_name";
const char PYTHON_PARSE_GET_LOCATION[] = "get_location";
const char PYTHON_PARSE_EXPAND_EXPR_STATEMENT[] = "expand_expr_statement";
const char PYTHON_PARSE_GENERATE_SCOPE[] = "generate_scope";
const char PYTHON_PARSE_GET_SCOPE_NAME[] = "get_scope_name";
const char PYTHON_PARSE_GET_TYPE[] = "get_type";
const char PYTHON_PARSE_ANALYZE_SUPER[] = "analyze_super";
const char PYTHON_PARSE_CHECK_THIRD_PARTY_LIBRARY_SIDE_EFFECT[] = "check_third_party_library_side_effect";
const char PYTHON_PARSE_CHECK_ATTR_IS_PROPERTY[] = "check_attr_is_property";

const char PYTHON_PARSE_CLASS_SLICE[] = "create_slice_obj";
const char PYTHON_PARSE_CLASS_ELLIPSIS[] = "create_ellipsis_obj";

const char PYTHON_MOD_GET_MODULE_AND_NAME_INFO[] = "get_obj_module_and_name_info";
const char PYTHON_MOD_IS_JIT_FORBIDDEN_MODULE[] = "is_jit_forbidden_module";
const char PYTHON_MOD_IS_INVALID_METHOD[] = "is_invalid_or_jit_forbidden_method";
const char PYTHON_MOD_IS_FROM_THIRD_PARTY_LIBRARY[] = "is_from_third_party_library";

// Define the common name.
const char NAMED_PRIMITIVE_LEN[] = "len";
const char NAMED_PRIMITIVE_BODY[] = "body";
const char NAMED_PRIMITIVE_ASSIGN[] = "Assign";
const char NAMED_PRIMITIVE_AUGASSIGN[] = "AugAssign";
const char NAMED_PRIMITIVE_FOR[] = "For";
const char NAMED_PRIMITIVE_IF[] = "If";
const char NAMED_PRIMITIVE_ORELSE[] = "orelse";
const char NAMED_PRIMITIVE_WHILE[] = "While";
const char NAMED_PRIMITIVE_VALUE[] = "value";
const char NAMED_PRIMITIVE_VALUES[] = "values";
const char NAMED_PRIMITIVE_FUNC[] = "func";
const char NAMED_PRIMITIVE_TEST[] = "test";
const char NAMED_PRIMITIVE_LEFT[] = "left";
const char NAMED_PRIMITIVE_ARGS[] = "args";
const char NAMED_PRIMITIVE_CALL[] = "Call";
const char NAMED_PRIMITIVE_SUBSCRIPT[] = "Subscript";
const char NAMED_PRIMITIVE_ATTRIBUTE[] = "Attribute";
const char NAMED_PRIMITIVE_COMPARE[] = "Compare";
const char NAMED_PRIMITIVE_BOOLOP[] = "BoolOp";
const char NAMED_PRIMITIVE_NAMECONSTANT[] = "NameConstant";
const char NAMED_PRIMITIVE_CONSTANT[] = "Constant";
const char NAMED_PRIMITIVE_COMPARATORS[] = "comparators";
const char NAMED_PRIMITIVE_TARGET[] = "target";
const char NAMED_PRIMITIVE_TARGETS[] = "targets";
const char NAMED_PRIMITIVE_SLICE[] = "slice";
const char NAMED_PRIMITIVE_NAME[] = "Name";
const char NAMED_PRIMITIVE_NUM[] = "Num";
const char NAMED_PRIMITIVE_STR[] = "Str";
const char NAMED_PRIMITIVE_ITER[] = "iter";
const char NAMED_PRIMITIVE_NEXT[] = "next";
const char NAMED_PRIMITIVE_GETITEM[] = "getitem";
const char NAMED_PRIMITIVE_SETITEM[] = "setitem";
const char NAMED_PRIMITIVE_HASNEXT[] = "hasnext";
const char NAMED_PRIMITIVE_BOOL[] = "bool_";
const char NAMED_PRIMITIVE_CHECK_LEN[] = "check_len_";
const char NAMED_PRIMITIVE_REAL_BOOL[] = "real_bool_";
const char NAMED_PRIMITIVE_MAKETUPLE[] = "MakeTuple";
const char NAMED_PRIMITIVE_MAKELIST[] = "make_list";
const char NAMED_PRIMITIVE_MAKESLICE[] = "make_slice";
const char NAMED_PRIMITIVE_MAKEDICT[] = "make_dict";
const char NAMED_METAGRAPH_UNPACKCALL[] = "unpack_call";
const char NAMED_METAGRAPH_STARRED_UNPACK[] = "starred_unpack";
const char NAMED_METAGRAPH_STARRED_GET_ITEM[] = "starred_get_item";
const char NAMED_METAGRAPH_STARRED_UNPACK_MERGE[] = "starred_unpack_merge";

// Define NAMED_PRIMITIVE_GETATTR "getattr".
// Define python inline attr.
const char PYTHON_GET_METHOD_LEN[] = "__len__";
const char PYTHON_GET_METHOD_SELF_CLASS[] = "__self__";
const char PYTHON_GET_OBJ_DESC[] = "__str__";

const char PYTHON_PARSE_METHOD[] = "__parse_method__";
const char PYTHON_FUNC_GRAPH_FLAGS[] = "_func_graph_flags";

// Define the parse constant.
const char CUSTOM_BPROP_NAME[] = "bprop";
const char STAGE_NAME[] = "_pipeline_stage";
const char SEGMENT_NAME[] = "_pipeline_segment";

// Define the Namespace name.
const char RESOLVE_NAMESPACE_NAME_AST[] = "Ast";                   // For ast type namespace.
const char RESOLVE_NAMESPACE_NAME_CLASS_OBJECT[] = "ClassObject";  // For class object itself namespace.
const char RESOLVE_NAMESPACE_NAME_CLASS_MEMBER[] = "ClassMember";  // For class member namespace.
const char RESOLVE_NAMESPACE_NAME_SYMBOL_STR[] = "SymbolStr";      // For symbol str namespace.
const char RESOLVE_NAMESPACE_NAME_COMMON_OPS[] = "CommonOPS";      // For common ops, eg: hasnext, next.
const char RESOLVE_NAMESPACE_NAME_MODULE[] = "Module";             // For Module namespace.

// Define Resolve type.
enum ResolveType : int64_t {
  RESOLVE_TYPE_NONE = 0,                // Resolve None.
  RESOLVE_TYPE_FUNCTION = 1,            // Resolve function.
  RESOLVE_TYPE_METHOD = 2,              // Resolve class method.
  RESOLVE_TYPE_CLASS_TYPE = 3,          // Resolve class type.
  RESOLVE_TYPE_CLASS_INSTANCE = 4,      // Resolve the class instance of common class.
  RESOLVE_TYPE_NAMESPACE_INSTANCE = 5,  // Resolve the namespace instance.
  RESOLVE_TYPE_NUMPY_INT_NUMBER = 6,    // Resolve numpy number int type.
  RESOLVE_TYPE_NUMPY_FLOAT_NUMBER = 7,  // Resolve numpy number float type.
  RESOLVE_TYPE_NUMPY_BOOL_NUMBER = 8,   // Resolve numpy bool number.
  RESOLVE_TYPE_TUPLE = 9,               // Resolve builtin tuple type.
  RESOLVE_TYPE_LIST = 10,               // Resolve builtin list type.
  RESOLVE_TYPE_INVALID = 0xFF           // Resolve invalid.
};

// Define the class instance detail type When the type is RESOLVE_TYPE_CLASS_INSTANCE.
enum ClassInstanceType {
  CLASS_INSTANCE_TYPE_CELL = 0,            // Class instance type is Cell.
  CLASS_INSTANCE_TYPE_PRIMITIVE = 1,       // Class instance type is Primitive.
  CLASS_INSTANCE_TYPE_NUMPY_ARRAY = 2,     // Class instance type is Numpy Array.
  CLASS_INSTANCE_TYPE_TENSOR = 3,          // Class instance type is Tensor
  CLASS_INSTANCE_TYPE_ADAPTER_TENSOR = 4,  // Class instance type is Adapter Tensor
  CLASS_INSTANCE_TYPE_INVALID = 0xFF
};

// Define syntax support type.
enum SyntaxSupportType : int {
  SYNTAX_SUPPORTED = 0,                  // Supported syntax
  SYNTAX_UNSUPPORTED_INTERNAL_TYPE = 1,  // Unsupported internal type
  SYNTAX_UNSUPPORTED_EXTERNAL_TYPE = 2,  // Unsupported external type
  SYNTAX_HYBRID_TYPE = 3,                // Hybrid type
  SYNTAX_UNSUPPORTED_NAMESPACE = 4       // Unsupported namespace
};

// Convert python object to ValuePtr.
bool ConvertData(const py::object &obj, ValuePtr *data, bool use_signature = false, const TypePtr &dtype = nullptr,
                 bool forbid_reuse = false);

bool ConvertStubData(const py::object &obj, ValuePtr *data, bool use_signature = false, const TypePtr &dtype = nullptr,
                     bool forbid_reuse = false);

// Convert python obj to graph.
FuncGraphPtr ConvertToFuncGraph(const py::object &obj, const ValuePtrList &args_value_list = {},
                                const std::string &python_mod_get_parse_method = PYTHON_MOD_GET_PARSE_METHOD,
                                bool forbid_reuse = false);

// Parse the python object to graph.
FuncGraphPtr ParsePythonCode(const py::object &obj,
                             const std::string &python_mod_get_parse_method = PYTHON_MOD_GET_PARSE_METHOD,
                             const ValuePtrList &args_value_list = {});
ValuePtr GetArgDefaultValue(const std::string &prim_name, const std::string &arg_name);
AnfNodePtr TransPropertyToFunc(const FuncGraphPtr &fg, const AnfNodePtr &node, const py::object &property_net_obj,
                               std::string attr_str);
}  // namespace parse
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PIPELINE_JIT_PARSE_PARSE_BASE_H_
