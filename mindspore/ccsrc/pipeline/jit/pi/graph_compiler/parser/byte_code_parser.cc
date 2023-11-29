/**
 * Copyright 2019-2023 Huawei Technologies Co., Ltd
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

#include "pipeline/jit/pi/graph_compiler/parser/byte_code_parser.h"
#include <memory>
#include <string>
#include <utility>
#include "pipeline/jit/pi/graph_compiler/pi_ir/debug_info.h"
#include "utils/log_adapter.h"

namespace mindspore {
namespace jit {
namespace graph {
namespace {
template <typename T>
T var_init(PyObject *obj) {
  if (obj == nullptr) {
    return T();
  }
  if (py::isinstance<py::module>(obj)) {
    obj = PyObject_GetAttrString(obj, "__dict__");
  }
  return py::cast<T>(obj);
}

const PyFunctionObject &GetFunctionObject(const py::object &func) {
  if (PyMethod_Check(func.ptr())) {
    return *reinterpret_cast<PyFunctionObject *>(PyMethod_GET_FUNCTION(func.ptr()));
  }
  return *reinterpret_cast<PyFunctionObject *>(func.ptr());
}
}  // namespace

ByteCodeParser::ByteCodeParser(const py::object &func) : ByteCodeParser(GetFunctionObject(func)) {}

ByteCodeParser::ByteCodeParser(const PyFunctionObject &func)
    : func_(std::make_shared<ir::FunctionNode>(
        py::cast<std::string>((reinterpret_cast<PyCodeObject *>(func.func_code))->co_name))),
      code_(*reinterpret_cast<PyCodeObject *>(func.func_code)),
      globals_(var_init<py::dict>(func.func_globals)),
      builtins_(var_init<py::dict>(PyDict_GetItemString(func.func_globals, "__builtins__"))),
      clousre_(var_init<py::tuple>(func.func_closure)),
      kwdefaults_(var_init<py::dict>(func.func_kwdefaults)),
      defaults_(var_init<py::tuple>(func.func_defaults)) {
  func_->AddFileName(py::cast<std::string>((reinterpret_cast<PyCodeObject *>(func.func_code))->co_filename));
  BuildMethodMap();
  Register(&func_->GetNodes());
}

void ByteCodeParser::BuildMethodMap() {
  BuildStackMethodMap();
  BuildLoadStoreMethodMap();
  BuildMathMethodMap();
  BuildBitwiseMethodMap();
  BuildContainerMethodMap();
  BuildContrlFlowMethodMap();
  BuildOtherMethodMap();
}

void ByteCodeParser::BuildStackMethodMap() {
  // op : 1, opname : POP_TOP
  instr_method_map_[POP_TOP] = &ByteCodeParser::ParsePopTop;
  // op : 2, opname : ROT_TWO
  instr_method_map_[ROT_TWO] = &ByteCodeParser::ParseRotTwo;
  // op : 3, opname : ROT_THREE
  instr_method_map_[ROT_THREE] = &ByteCodeParser::ParseRotThree;
  // op : 4, opname : DUP_TOP
  instr_method_map_[DUP_TOP] = &ByteCodeParser::ParseDupTop;
  // op : 5, opname : DUP_TOP_TWO
  instr_method_map_[DUP_TOP_TWO] = &ByteCodeParser::ParseDupTwo;
  // op : 6, opname : ROT_FOUR
  instr_method_map_[ROT_FOUR] = &ByteCodeParser::ParseRotFour;
  // op : 9, opname : NOP
  instr_method_map_[NOP] = &ByteCodeParser::ParseNop;
  // op : 61, opname : DELETE_SUBSCR
  instr_method_map_[DELETE_SUBSCR] = &ByteCodeParser::ParseDeleteSubscr;
  // op : 91, opname : DELETE_NAME
  instr_method_map_[DELETE_NAME] = &ByteCodeParser::ParseDeleteName;
  // op : 96, opname : DELETE_ATTR
  instr_method_map_[DELETE_ATTR] = &ByteCodeParser::ParseDeleteAttr;
  // op : 98, opname : DELETE_GLOBAL
  instr_method_map_[DELETE_GLOBAL] = &ByteCodeParser::ParseDeleteName;
  // op : 126, opname : DELETE_FAST
  instr_method_map_[DELETE_FAST] = &ByteCodeParser::ParseDeleteName;
  // op : 138, opname : DELETE_DEREF
  instr_method_map_[DELETE_DEREF] = &ByteCodeParser::ParseDeleteName;
}

void ByteCodeParser::BuildLoadStoreMethodMap() {
  // op : 60, opname : STORE_SUBSCR
  instr_method_map_[STORE_SUBSCR] = &ByteCodeParser::ParseStoreSubscr;
  // op : 71, opname : LOAD_BUILD_CLASS
  instr_method_map_[LOAD_BUILD_CLASS] = &ByteCodeParser::ParseUnaryOpertion;
  // op : 74, opname : LOAD_ASSERTION_ERROR
  instr_method_map_[LOAD_ASSERTION_ERROR] = &ByteCodeParser::ParseLoadAssertError;
  // op : 84, opname : IMPORT_STAR
  instr_method_map_[IMPORT_STAR] = &ByteCodeParser::ParseImport;
  // op : 90, opname : STORE_NAME
  instr_method_map_[STORE_NAME] = &ByteCodeParser::ParseStoreName;
  // op : 95, opname : STORE_ATTR
  instr_method_map_[STORE_ATTR] = &ByteCodeParser::ParseStoreAttr;
  // op : 97, opname : STORE_GLOBAL
  instr_method_map_[STORE_GLOBAL] = &ByteCodeParser::ParseStoreName;
  // op : 100, opname : LOAD_CONST
  instr_method_map_[LOAD_CONST] = &ByteCodeParser::ParseLoadConst;
  // op : 101, opname : LOAD_NAME
  instr_method_map_[LOAD_NAME] = &ByteCodeParser::ParseLoadName;
  // op : 106, opname : LOAD_ATTR
  instr_method_map_[LOAD_ATTR] = &ByteCodeParser::ParseLoadAttr;
  // op : 108, opname : IMPORT_NAME
  instr_method_map_[IMPORT_NAME] = &ByteCodeParser::ParseImport;
  // op : 109, opname : IMPORT_FROM
  instr_method_map_[IMPORT_FROM] = &ByteCodeParser::ParseImport;
  // op : 116, opname : LOAD_GLOBAL
  instr_method_map_[LOAD_GLOBAL] = &ByteCodeParser::ParseLoadGlobal;
  // op : 124, opname : LOAD_FAST
  instr_method_map_[LOAD_FAST] = &ByteCodeParser::ParseLoadFast;
  // op : 125, opname : STORE_FAST
  instr_method_map_[STORE_FAST] = &ByteCodeParser::ParseStoreName;
  // op : 135, opname : LOAD_CLOSURE
  instr_method_map_[LOAD_CLOSURE] = &ByteCodeParser::ParseLoadClosure;
  // op : 136, opname : LOAD_DEREF
  instr_method_map_[LOAD_DEREF] = &ByteCodeParser::ParseLoadClosure;
  // op : 137, opname : STORE_DEREF
  instr_method_map_[STORE_DEREF] = &ByteCodeParser::ParseStoreName;
  // op : 148, opname : LOAD_CLASSDEREF
  instr_method_map_[LOAD_CLASSDEREF] = &ByteCodeParser::ParseLoadClosure;
  // op : 160, opname : LOAD_METHOD
  instr_method_map_[LOAD_METHOD] = &ByteCodeParser::ParseLoadMethod;
}

void ByteCodeParser::BuildMathMethodMap() {
  // op : 10, opname : UNARY_POSITIVE
  instr_method_map_[UNARY_POSITIVE] = &ByteCodeParser::ParseNop;
  // op : 11, opname : UNARY_NEGATIVE
  instr_method_map_[UNARY_NEGATIVE] = &ByteCodeParser::ParseUnaryNegative;
  // op : 16, opname : BINARY_MATRIX_MULTIPLY
  instr_method_map_[BINARY_MATRIX_MULTIPLY] = &ByteCodeParser::ParseMul;
  // op : 17, opname : INPLACE_MATRIX_MULTIPLY
  instr_method_map_[INPLACE_MATRIX_MULTIPLY] = &ByteCodeParser::ParseMul;
  // op : 19, opname : BINARY_POWER
  instr_method_map_[BINARY_POWER] = &ByteCodeParser::ParseBinaryOpertion;
  // op : 20, opname : BINARY_MULTIPLY
  instr_method_map_[BINARY_MULTIPLY] = &ByteCodeParser::ParseMul;
  // op : 22, opname : BINARY_MODULO
  instr_method_map_[BINARY_MODULO] = &ByteCodeParser::ParseBinaryOpertion;
  // op : 23, opname : BINARY_ADD
  instr_method_map_[BINARY_ADD] = &ByteCodeParser::ParseAdd;
  // op : 24, opname : BINARY_SUBTRACT
  instr_method_map_[BINARY_SUBTRACT] = &ByteCodeParser::ParseSub;
  // op : 26, opname : BINARY_FLOOR_DIVIDE
  instr_method_map_[BINARY_FLOOR_DIVIDE] = &ByteCodeParser::ParseDiv;
  // op : 27, opname : BINARY_TRUE_DIVIDE
  instr_method_map_[BINARY_TRUE_DIVIDE] = &ByteCodeParser::ParseDiv;
  // op : 28, opname : INPLACE_FLOOR_DIVIDE
  instr_method_map_[INPLACE_FLOOR_DIVIDE] = &ByteCodeParser::ParseDiv;
  // op : 29, opname : INPLACE_TRUE_DIVIDE
  instr_method_map_[INPLACE_TRUE_DIVIDE] = &ByteCodeParser::ParseDiv;
  // op : 55, opname : INPLACE_ADD
  instr_method_map_[INPLACE_ADD] = &ByteCodeParser::ParseAdd;
  // op : 56, opname : INPLACE_SUBTRACT
  instr_method_map_[INPLACE_SUBTRACT] = &ByteCodeParser::ParseSub;
  // op : 57, opname : INPLACE_MULTIPLY
  instr_method_map_[INPLACE_MULTIPLY] = &ByteCodeParser::ParseMul;
  // op : 59, opname : INPLACE_MODULO
  instr_method_map_[INPLACE_MODULO] = &ByteCodeParser::ParseBinaryOpertion;
  // op : 67, opname : INPLACE_POWER
  instr_method_map_[INPLACE_POWER] = &ByteCodeParser::ParseBinaryOpertion;
}

void ByteCodeParser::BuildBitwiseMethodMap() {
  // op : 12, opname : UNARY_NOT
  instr_method_map_[UNARY_NOT] = &ByteCodeParser::ParseUnaryNot;
  // op : 15, opname : UNARY_INVERT
  instr_method_map_[UNARY_INVERT] = &ByteCodeParser::ParseUnaryInvert;
  // op : 62, opname : BINARY_LSHIFT
  instr_method_map_[BINARY_LSHIFT] = &ByteCodeParser::ParseBitwise;
  // op : 63, opname : BINARY_RSHIFT
  instr_method_map_[BINARY_RSHIFT] = &ByteCodeParser::ParseBitwise;
  // op : 64, opname : BINARY_AND
  instr_method_map_[BINARY_AND] = &ByteCodeParser::ParseBitwise;
  // op : 65, opname : BINARY_XOR
  instr_method_map_[BINARY_XOR] = &ByteCodeParser::ParseBitwise;
  // op : 66, opname : BINARY_OR
  instr_method_map_[BINARY_OR] = &ByteCodeParser::ParseBitwise;
  // op : 75, opname : INPLACE_LSHIFT
  instr_method_map_[INPLACE_LSHIFT] = &ByteCodeParser::ParseBitwise;
  // op : 76, opname : INPLACE_RSHIFT
  instr_method_map_[INPLACE_RSHIFT] = &ByteCodeParser::ParseBitwise;
  // op : 77, opname : INPLACE_AND
  instr_method_map_[INPLACE_AND] = &ByteCodeParser::ParseBitwise;
  // op : 78, opname : INPLACE_XOR
  instr_method_map_[INPLACE_XOR] = &ByteCodeParser::ParseBitwise;
  // op : 79, opname : INPLACE_OR
  instr_method_map_[INPLACE_OR] = &ByteCodeParser::ParseBitwise;
}

void ByteCodeParser::BuildContainerMethodMap() {
  // op : 82, opname : LIST_TO_TUPLE
  instr_method_map_[LIST_TO_TUPLE] = &ByteCodeParser::ParseListToTuple;
  // op : 102, opname : BUILD_TUPLE
  instr_method_map_[BUILD_TUPLE] = &ByteCodeParser::ParseBuild;
  // op : 103, opname : BUILD_LIST
  instr_method_map_[BUILD_LIST] = &ByteCodeParser::ParseBuild;
  // op : 104, opname : BUILD_SET
  instr_method_map_[BUILD_SET] = &ByteCodeParser::ParseBuild;
  // op : 105, opname : BUILD_MAP
  instr_method_map_[BUILD_MAP] = &ByteCodeParser::ParseBuild;
  // op : 133, opname : BUILD_SLICE
  instr_method_map_[BUILD_SLICE] = &ByteCodeParser::ParseBuild;
  // op : 145, opname : LIST_APPEND
  instr_method_map_[LIST_APPEND] = &ByteCodeParser::ParseContainerUpdate;
  // op : 146, opname : SET_ADD
  instr_method_map_[SET_ADD] = &ByteCodeParser::ParseContainerUpdate;
  // op : 147, opname : MAP_ADD
  instr_method_map_[MAP_ADD] = &ByteCodeParser::ParseContainerUpdate;
  // op : 156, opname : BUILD_CONST_KEY_MAP
  instr_method_map_[BUILD_CONST_KEY_MAP] = &ByteCodeParser::ParseBuild;
  // op : 157, opname : BUILD_STRING
  instr_method_map_[BUILD_STRING] = &ByteCodeParser::ParseBuild;
  // op : 162, opname : LIST_EXTEND
  instr_method_map_[LIST_EXTEND] = &ByteCodeParser::ParseContainerUpdate;
  // op : 163, opname : SET_UPDATE
  instr_method_map_[SET_UPDATE] = &ByteCodeParser::ParseContainerUpdate;
  // op : 164, opname : DICT_MERGE
  instr_method_map_[DICT_MERGE] = &ByteCodeParser::ParseContainerUpdate;
  // op : 165, opname : DICT_UPDATE
  instr_method_map_[DICT_UPDATE] = &ByteCodeParser::ParseContainerUpdate;
}

void ByteCodeParser::BuildContrlFlowMethodMap() {
  // op : 48, opname : RERAISE
  instr_method_map_[RERAISE] = &ByteCodeParser::ParseUnaryOpertion;
  // op : 49, opname : WITH_EXCEPT_START
  instr_method_map_[WITH_EXCEPT_START] = &ByteCodeParser::ParseWithExceptStart;
  // op : 87, opname : POP_BLOCK
  instr_method_map_[POP_BLOCK] = &ByteCodeParser::ParseNoArgOperation;
  // op : 89, opname : POP_EXCEPT
  instr_method_map_[POP_EXCEPT] = &ByteCodeParser::ParseNoArgOperation;
  // op : 93, opname : FOR_ITER
  instr_method_map_[FOR_ITER] = &ByteCodeParser::ParseJump;
  // op : 110, opname : JUMP_FORWARD
  instr_method_map_[JUMP_FORWARD] = &ByteCodeParser::ParseJump;
  // op : 111, opname : JUMP_IF_FALSE_OR_POP
  instr_method_map_[JUMP_IF_FALSE_OR_POP] = &ByteCodeParser::ParseJump;
  // op : 112, opname : JUMP_IF_TRUE_OR_POP
  instr_method_map_[JUMP_IF_TRUE_OR_POP] = &ByteCodeParser::ParseJump;
  // op : 113, opname : JUMP_ABSOLUTE
  instr_method_map_[JUMP_ABSOLUTE] = &ByteCodeParser::ParseJump;
  // op : 114, opname : POP_JUMP_IF_FALSE
  instr_method_map_[POP_JUMP_IF_FALSE] = &ByteCodeParser::ParseJump;
  // op : 115, opname : POP_JUMP_IF_TRUE
  instr_method_map_[POP_JUMP_IF_TRUE] = &ByteCodeParser::ParseJump;
  // op : 121, opname : JUMP_IF_NOT_EXC_MATCH
  instr_method_map_[JUMP_IF_NOT_EXC_MATCH] = &ByteCodeParser::ParseJump;
  // op : 122, opname : SETUP_FINALLY
  instr_method_map_[SETUP_FINALLY] = &ByteCodeParser::ParseNoArgOperation;
  // op : 130, opname : RAISE_VARARGS
  instr_method_map_[RAISE_VARARGS] = &ByteCodeParser::ParseRaiseVarargs;
}

void ByteCodeParser::BuildOtherMethodMap() {
  // op : 25, opname : BINARY_SUBSCR
  instr_method_map_[BINARY_SUBSCR] = &ByteCodeParser::ParseBinarySubscr;
  // op : 50, opname : GET_AITER
  instr_method_map_[GET_AITER] = &ByteCodeParser::ParseGet;
  // op : 51, opname : GET_ANEXT
  instr_method_map_[GET_ANEXT] = &ByteCodeParser::ParseGet;
  // op : 68, opname : GET_ITER
  instr_method_map_[GET_ITER] = &ByteCodeParser::ParseGet;
  // op : 69, opname : GET_YIELD_FROM_ITER
  instr_method_map_[GET_YIELD_FROM_ITER] = &ByteCodeParser::ParseGet;
  // op : 70, opname : PRINT_EXPR
  instr_method_map_[PRINT_EXPR] = &ByteCodeParser::ParseUnaryOpertion;
  // op : 72, opname : YIELD_FROM
  instr_method_map_[YIELD_FROM] = &ByteCodeParser::ParseUnaryOpertion;
  // op : 73, opname : GET_AWAITABLE
  instr_method_map_[GET_AWAITABLE] = &ByteCodeParser::ParseGet;
  // op : 83, opname : RETURN_VALUE
  instr_method_map_[RETURN_VALUE] = &ByteCodeParser::ParseReturnValue;
  // op : 85, opname : SETUP_ANNOTATIONS
  instr_method_map_[SETUP_ANNOTATIONS] = &ByteCodeParser::ParseNoArgOperation;
  // op : 86, opname : YIELD_VALUE
  instr_method_map_[YIELD_VALUE] = &ByteCodeParser::ParseUnaryOpertion;
  // op : 92, opname : UNPACK_SEQUENCE
  instr_method_map_[UNPACK_SEQUENCE] = &ByteCodeParser::ParseUnpack;
  // op : 94, opname : UNPACK_EX
  instr_method_map_[UNPACK_EX] = &ByteCodeParser::ParseUnpack;
  // op : 107, opname : COMPARE_OP
  instr_method_map_[COMPARE_OP] = &ByteCodeParser::ParseCompareOp;
  // op : 117, opname : IS_OP
  instr_method_map_[IS_OP] = &ByteCodeParser::ParseIsOp;
  // op : 118, opname : CONTAINS_OP
  instr_method_map_[CONTAINS_OP] = &ByteCodeParser::ParseContainsOp;
  // op : 131, opname : CALL_FUNCTION
  instr_method_map_[CALL_FUNCTION] = &ByteCodeParser::ParseCallFunction;
  // op : 132, opname : MAKE_FUNCTION
  instr_method_map_[MAKE_FUNCTION] = &ByteCodeParser::ParseMakeFunction;
  // op : 141, opname : CALL_FUNCTION_KW
  instr_method_map_[CALL_FUNCTION_KW] = &ByteCodeParser::ParseCallFunction;
  // op : 142, opname : CALL_FUNCTION_EX
  instr_method_map_[CALL_FUNCTION_EX] = &ByteCodeParser::ParseCallFunction;
  // op : 143, opname : SETUP_WITH
  instr_method_map_[SETUP_WITH] = &ByteCodeParser::ParseSetupWith;
  // op : 155, opname : FORMAT_VALUE
  instr_method_map_[FORMAT_VALUE] = &ByteCodeParser::ParseFormatValue;
  // op : 161, opname : CALL_METHOD
  instr_method_map_[CALL_METHOD] = &ByteCodeParser::ParseCallFunction;
}

void ByteCodeParser::SaveNode(const ir::NodePtr &node) {
  latest_gen_node_ = node;
  nodes_.back()->push_back(node);
}

ir::NodePtr ByteCodeParser::PopStack() {
  ir::NodePtr node = stack_.back();
  stack_.pop_back();
  return node;
}

void ByteCodeParser::PushStack(const ir::NodePtr &node) {
  latest_gen_node_ = node;
  stack_.emplace_back(node);
}

ir::FunctionNodePtr ByteCodeParser::Parse() {
  GenerateFunctionParameters();
  py::object code = py::cast<py::object>(reinterpret_cast<PyObject *>(const_cast<PyCodeObject *>(&code_)));
  const py::object instructions = py::module::import("dis").attr("get_instructions")(code);
  ParseInstructions(instructions);
  for (auto &[key, value] : jump_nodes_map_) {
    MS_EXCEPTION_IF_CHECK_FAIL(targets_map_.find(key) != targets_map_.end(), "Jump no target.");
    for (auto &node : value) {
      node->SetTarget(targets_map_.at(key));
    }
  }
  return func_;
}

bool ByteCodeParser::IsConditionJump(ir::OpCode op) {
  return op == POP_JUMP_IF_TRUE || op == POP_JUMP_IF_FALSE || op == JUMP_IF_NOT_EXC_MATCH ||
         op == JUMP_IF_TRUE_OR_POP || op == JUMP_IF_FALSE_OR_POP;
}

void ByteCodeParser::CallInstrMethod(const InstrPtr &instr) {
  bool need_ext_instr = (instr->GetArg() > 255);
  (this->*instr_method_map_[instr->GetOpCode()])(instr);
  latest_gen_node_->SetNeedExtInstr(need_ext_instr);
  if (instr->IsJumpTarget()) {
    MS_EXCEPTION_IF_CHECK_FAIL(latest_gen_node_ != nullptr, "Jump Target must not be nullptr.");
    targets_map_[instr->GetOffset()] = latest_gen_node_;
  }
}

void ByteCodeParser::ParseInstructions(const py::list &instrs) {
  const int bits_per_byte = 8;
  int extended_arg = 0;
  for (size_t index = 0; index < instrs.size(); index++) {
    InstrPtr instr = std::make_unique<Instr>(instrs[index]);
    MS_LOG(DEBUG) << "Start parse instruction : " << instr->ToString();
    int op_code = instr->GetOpCode();
    if (op_code == EXTENDED_ARG) {
      extended_arg = (instr->GetArg());
      continue;
    }
    if (extended_arg != 0) {
      instr->SetArg(extended_arg << bits_per_byte | instr->GetArg());
      extended_arg = 0;
    }
    if (instr_method_map_.find(op_code) == instr_method_map_.end()) {
      MS_LOG_EXCEPTION << "OpCode : " << instr->GetOpName() << " Not Implemented.";
    }
    if (IsConditionJump(op_code)) {
      index++;
      size_t size = (instr->GetArg() - instr->GetOffset()) / 2;
      py::list then_ = instrs[py::slice(index, index + size - 1, 1)];
      index += then_.size();
      InstrPtr may_jump = std::make_unique<Instr>(then_[then_.size() - 1]);
      py::list else_;
      if (may_jump->GetOpCode() == JUMP_ABSOLUTE) {
        if (may_jump->GetArg() < may_jump->GetOffset()) {
          ParseWhile(instr, then_);
        } else {
          else_ = instrs[py::slice(index, instrs.size(), 1)];
          index += else_.size();
          ParseIf(instr, then_, else_);
        }
      } else {
        if (may_jump->GetOpCode() == JUMP_FORWARD) {
          else_ = instrs[py::slice(index, index + (may_jump->GetArg() / 2), 1)];
          index += else_.size();
        }
        ParseIf(instr, then_, else_);
      }
    }
    if (op_code == FOR_ITER) {
      index++;
      size_t size = instr->GetArg() / 2;
      py::list body = instrs[py::slice(index, index + size, 1)];
      index += size;
      ParseWhile(instr, body);
    }
    if (index >= instrs.size()) {
      break;
    }
    instr = std::make_unique<Instr>(instrs[index]);
    CallInstrMethod(instr);
    MS_LOG(DEBUG) << "Instruction " << instr->GetOpName() << " parse finished.";
  }
}

void ByteCodeParser::ParseIf(const InstrPtr &cond, const py::list &then_, const py::list &else_) {
  CallInstrMethod(cond);
  ir::IfNodePtr node = std::make_shared<ir::IfNode>(PopStack());
  Register(&node->GetThen());
  ParseInstructions(then_);
  Restore();
  Register(&node->GetElse());
  ir::OpCode op_code = cond->GetOpCode();
  if (op_code == JUMP_IF_TRUE_OR_POP || op_code == JUMP_IF_FALSE_OR_POP) {
    stack_.emplace_back(node->GetCondition());
  }
  ParseInstructions(else_);
  Restore();
  SaveNode(node);
}

void ByteCodeParser::ParseWhile(const InstrPtr &cond, const py::list &body) {
  CallInstrMethod(cond);
  // TOS is an iterator. Call its __next__() method.
  // If this yields a new value, push it on the stack (leaving the iterator below it).
  // If the iterator indicates it is exhausted, TOS is popped
  ir::WhileNodePtr node = std::make_shared<ir::WhileNode>(PopStack());
  PushStack(std::make_shared<ir::RefNode>(node->GetCondition()));
  Register(&node->GetBody());
  ParseInstructions(body);
  Restore();
  SaveNode(node);
}

void ByteCodeParser::GeneratePostionalParameters() {
  MS_LOG(DEBUG) << "Generate function postional parameters ...";
  for (size_t index = 0; index < (size_t)code_.co_argcount; index++) {
    std::string name = py::cast<std::string>(PyTuple_GET_ITEM(code_.co_varnames, index));
    ir::ParameterPtr param = std::make_shared<ir::Parameter>(index, name);
    // Set arg as positional parameter
    param->SetCategory(ir::Parameter::POSITIONAL);
    // Parameter with default values ​​are placed back position in the positional parameters list
    if (index >= (size_t)code_.co_argcount - defaults_.size()) {
      size_t defalut_value_index = index + defaults_.size() - (size_t)code_.co_argcount;
      ir::ValuePtr defalut_value = std::make_shared<ir::Value>(defaults_[defalut_value_index]);
      param->SetDefaultValue(defalut_value);
    }
    func_->AddParameter(param);
  }
}

void ByteCodeParser::GenerateVariableParameter() {
  if ((code_.co_flags & CO_VARARGS) == 0x0) {
    return;
  }
  MS_LOG(DEBUG) << "Generate function variable parameter ...";
  // Arguments order : { postional args, keyword only args, variable arg, keyword arg }
  auto index = code_.co_argcount + code_.co_kwonlyargcount;
  std::string name = py::cast<std::string>(PyTuple_GET_ITEM(code_.co_varnames, index));
  // Parameters order : { postional parameters, keyword only parameters, variable parameter, keyword parameter }
  ir::ParameterPtr param = std::make_shared<ir::Parameter>(code_.co_argcount, name);
  // set arg as var arg
  param->SetCategory(ir::Parameter::VARIABLE);
  func_->AddParameter(param);
}

void ByteCodeParser::GenerateKeywordOnlyParameters() {
  if (code_.co_kwonlyargcount == 0) {
    return;
  }
  MS_LOG(DEBUG) << "Generate function keyword only parameters ...";
  // Parameters order : { postional parameters, keyword only parameters, variable parameter, keyword parameter }
  int index = code_.co_argcount + (func_->HasVarArg() ? 1 : 0);
  for (auto &[key, value] : kwdefaults_) {
    auto name = py::cast<std::string>(key);
    ir::ParameterPtr param = std::make_shared<ir::Parameter>(index, name);
    index++;
    // set arg as kwonly arg
    param->SetCategory(ir::Parameter::KEYWORD_ONLY);
    ir::ValuePtr defalut_value = std::make_shared<ir::Value>(py::cast<py::object>(value));
    param->SetDefaultValue(defalut_value);
    func_->AddParameter(param);
  }
}

void ByteCodeParser::GenerateKeywordParameter() {
  if ((code_.co_flags & CO_VARKEYWORDS) == 0x0) {
    return;
  }
  MS_LOG(DEBUG) << "Generate function keyword parameter ...";
  auto index = code_.co_argcount + code_.co_kwonlyargcount;
  index += (func_->HasVarArg() ? 1 : 0);
  std::string name = py::cast<std::string>(PyTuple_GET_ITEM(code_.co_varnames, index));
  ir::ParameterPtr param = std::make_shared<ir::Parameter>(index, name);
  // set arg as kw arg
  param->SetCategory(ir::Parameter::KEYWORD);
  func_->AddParameter(param);
}

void ByteCodeParser::GenerateFunctionParameters() {
  func_->SetPosArgsCnt(code_.co_argcount);
  func_->SetKwOnlyArgsCnt(code_.co_kwonlyargcount);
  func_->SetFlags(code_.co_flags);
  func_->SetFirstLineNo(code_.co_firstlineno);
  func_->SetStackSize(code_.co_stacksize);
  func_->AddFileName(py::cast<std::string>(code_.co_filename));
  GeneratePostionalParameters();
  GenerateVariableParameter();
  GenerateKeywordOnlyParameters();
  GenerateKeywordParameter();
}

void ByteCodeParser::ParsePopTop(const InstrPtr &instr) {
  MS_EXCEPTION_IF_CHECK_FAIL(!stack_.empty(), "There is no item arg to pop at the top of stack.");
  ir::NodePtr node = std::make_shared<ir::UnaryOperation>(instr->GetOpCode(), PopStack());
  node->SetDebugInfo(GetNodeDebugInfo(instr));
  SaveNode(node);
}

void ByteCodeParser::DoRot(const int &cnt) {
  MS_EXCEPTION_IF_CHECK_FAIL((stack_.size() >= (size_t)cnt), "There is no item arg to pop at the top of stack.");
  auto top = PopStack();
  ir::NodePtrList stack;
  for (int idx = 0; idx < (cnt - 1); idx++) {
    stack.push_back(PopStack());
  }
  stack_.push_back(top);
  while (!stack.empty()) {
    stack_.push_back(stack.back());
    stack.pop_back();
  }
}

void ByteCodeParser::ParseRotTwo(const InstrPtr &instr) { DoRot(2); }

void ByteCodeParser::ParseRotThree(const InstrPtr &instr) { DoRot(3); }

void ByteCodeParser::ParseRotFour(const InstrPtr &instr) { DoRot(4); }

void ByteCodeParser::ParseNop(const InstrPtr &instr) {}

void ByteCodeParser::ParseDupTop(const InstrPtr &instr) { PushStack(stack_.back()); }

void ByteCodeParser::ParseDupTwo(const InstrPtr &instr) {
  auto iter = stack_.end();
  auto first = *(--iter);
  auto second = *(--iter);
  PushStack(second);
  PushStack(first);
}

// Parse unary operators, eg: `not a`, `-a`, etc.
void ByteCodeParser::ParseUnaryOpertion(const InstrPtr &instr) {
  auto top = PopStack();
  ir::NodePtr node = std::make_shared<ir::UnaryOperation>(instr->GetOpCode(), top);
  node->SetDebugInfo(GetNodeDebugInfo(instr));
  if (instr->GetOpCode() == RERAISE) {
    SaveNode(node);
  } else {
    PushStack(node);
  }
}

// Parse unary operators, eg: `-a`.
void ByteCodeParser::ParseUnaryNegative(const InstrPtr &instr) {
  auto top = PopStack();
  ir::NodePtr node = std::make_shared<ir::NegativeNode>(top);
  node->SetDebugInfo(GetNodeDebugInfo(instr));
  PushStack(node);
}

// Parse unary operators, eg: `not a`.
void ByteCodeParser::ParseUnaryNot(const InstrPtr &instr) {
  auto top = PopStack();
  ir::NodePtr node = std::make_shared<ir::NotNode>(top);
  node->SetDebugInfo(GetNodeDebugInfo(instr));
  PushStack(node);
}

// Parse unary operators, eg: `~a`.
void ByteCodeParser::ParseUnaryInvert(const InstrPtr &instr) {
  auto top = PopStack();
  ir::NodePtr node = std::make_shared<ir::InvertNode>(top);
  node->SetDebugInfo(GetNodeDebugInfo(instr));
  PushStack(node);
}

// Process binary operators, eg: `a + b`, `a | b`, etc.
void ByteCodeParser::ParseBinaryOpertion(const InstrPtr &instr) {
  auto right = PopStack();
  auto left = PopStack();
  ir::NodePtr node = std::make_shared<ir::BinaryOperation>(instr->GetOpCode(), left, right);
  node->SetDebugInfo(GetNodeDebugInfo(instr));
  PushStack(node);
}

// Process binary operators, eg: `a + b`, `a | b`, etc.
void ByteCodeParser::ParseBitwise(const InstrPtr &instr) {
  auto right = PopStack();
  auto left = PopStack();
  ir::NodePtr node = std::make_shared<ir::BitwiseNode>(instr->GetOpCode(), left, right);
  node->SetDebugInfo(GetNodeDebugInfo(instr));
  PushStack(node);
}

void ByteCodeParser::ParseAdd(const InstrPtr &instr) {
  auto right = PopStack();
  auto left = PopStack();
  ir::NodePtr node = std::make_shared<ir::AddNode>(instr->GetOpCode(), left, right);
  node->SetDebugInfo(GetNodeDebugInfo(instr));
  PushStack(node);
}

void ByteCodeParser::ParseSub(const InstrPtr &instr) {
  auto right = PopStack();
  auto left = PopStack();
  ir::NodePtr node = std::make_shared<ir::SubNode>(instr->GetOpCode(), left, right);
  node->SetDebugInfo(GetNodeDebugInfo(instr));
  PushStack(node);
}

void ByteCodeParser::ParseMul(const InstrPtr &instr) {
  auto right = PopStack();
  auto left = PopStack();
  ir::NodePtr node = std::make_shared<ir::MulNode>(instr->GetOpCode(), left, right);
  node->SetDebugInfo(GetNodeDebugInfo(instr));
  PushStack(node);
}

void ByteCodeParser::ParseDiv(const InstrPtr &instr) {
  auto right = PopStack();
  auto left = PopStack();
  ir::NodePtr node = std::make_shared<ir::DivNode>(instr->GetOpCode(), left, right);
  node->SetDebugInfo(GetNodeDebugInfo(instr));
  PushStack(node);
}

// Process binary operators, eg: `TOS = TOS1[TOS]`.
void ByteCodeParser::ParseBinarySubscr(const InstrPtr &instr) {
  auto subscr = PopStack();
  auto base = PopStack();
  ir::NodePtr node = std::make_shared<ir::BinaryOperation>(instr->GetOpCode(), base, subscr);
  node->SetDebugInfo(GetNodeDebugInfo(instr));
  PushStack(node);
}

// Process binary operators, eg: `a > b`, `a < b`, etc.
void ByteCodeParser::ParseCompareOp(const InstrPtr &instr) {
  auto right = PopStack();
  auto left = PopStack();
  ir::NodePtr node = std::make_shared<ir::CompareNode>(instr->GetArg(), left, right);
  node->SetDebugInfo(GetNodeDebugInfo(instr));
  PushStack(node);
}

void ByteCodeParser::ParseJump(const InstrPtr &instr) {
  ir::OpCode op_code = instr->GetOpCode();
  ir::NodePtr condition = nullptr;
  if (op_code != JUMP_FORWARD && op_code != JUMP_ABSOLUTE) {
    condition = PopStack();
  }
  if (op_code == JUMP_IF_NOT_EXC_MATCH) {
    condition = std::make_shared<ir::PairNode>(condition, PopStack());
  }
  ir::JumpNodePtr node = std::make_shared<ir::JumpNode>(op_code, condition, nullptr);
  node->SetDebugInfo(GetNodeDebugInfo(instr));
  int target_offset = instr->GetArg();
  if (op_code == JUMP_FORWARD || op_code == FOR_ITER) {
    target_offset += instr->GetOffset() + 2;
  }
  jump_nodes_map_[target_offset].push_back(node);
  if (op_code == JUMP_FORWARD || op_code == JUMP_ABSOLUTE) {
    SaveNode(node);
  } else {
    PushStack(node);
  }
}

void ByteCodeParser::ParseListToTuple(const InstrPtr &instr) {
  auto top = PopStack();
  ir::NodePtr node = std::make_shared<ir::CastNode>(top);
  node->SetDebugInfo(GetNodeDebugInfo(instr));
  PushStack(node);
}

void ByteCodeParser::ParseReturnValue(const InstrPtr &instr) {
  auto top = PopStack();
  ir::NodePtr node = std::make_shared<ir::ReturnNode>(top);
  node->SetDebugInfo(GetNodeDebugInfo(instr));
  PushStack(node);
  SaveNode(node);
}

void ByteCodeParser::ParseLoadConst(const InstrPtr &instr) {
  py::object opnd = py::cast<py::object>(PyTuple_GET_ITEM(code_.co_consts, instr->GetArg()));
  ir::ValuePtr arg = std::make_shared<ir::Value>(opnd, py::str(opnd), ir::kScopeConst);
  ir::NodePtr node = std::make_shared<ir::LoadValueNode>(instr->GetOpCode(), arg);
  node->SetDebugInfo(GetNodeDebugInfo(instr));
  PushStack(node);
}

void ByteCodeParser::ParseLoadName(const InstrPtr &instr) {
  py::object name = py::cast<py::object>(PyTuple_GET_ITEM(code_.co_names, instr->GetArg()));
  ir::ValuePtr arg = std::make_shared<ir::Value>(name, name.cast<std::string>(), ir::kScopeName);
  ir::NodePtr node = std::make_shared<ir::LoadValueNode>(instr->GetOpCode(), arg);
  node->SetDebugInfo(GetNodeDebugInfo(instr));
  PushStack(node);
}

void ByteCodeParser::ParseMakeFunction(const InstrPtr &instr) {
  ir::NodePtrList args;
  args.push_back(PopStack());
  args.push_back(PopStack());
  int flag = instr->GetArg();
  // closure
  if (flag & 0x08) {
    args.push_back(PopStack());
  }
  // annotations
  if (flag & 0x04) {
    args.push_back(PopStack());
  }
  // kwdefaults
  if (flag & 0x02) {
    args.push_back(PopStack());
  }
  // defaults
  if (flag & 0x01) {
    args.push_back(PopStack());
  }
  std::reverse(args.begin(), args.end());
  ir::NodePtr node = std::make_shared<ir::NaryWithFlagNode>(MAKE_FUNCTION, args, flag);
  node->SetDebugInfo(GetNodeDebugInfo(instr));
  PushStack(node);
}

// Process operators, eg:
// Tuple (..., TOS3, TOS2, TOS1, TOS)
// List {..., TOS3, TOS2, TOS1, TOS}
// Dict {..., TOS3: TOS2, TOS1: TOS}
// Dict {Value {..., TOS3, TOS2, TOS1}, Key (TOS)}
// Slice(TOS2, TOS1, TOS)
// Slice(TOS1, TOS)
void ByteCodeParser::ParseBuild(const InstrPtr &instr) {
  int size = instr->GetArg();
  if (instr->GetOpCode() == BUILD_CONST_KEY_MAP) {
    size++;
  }
  ir::NodePtrList opnds((instr->GetOpCode() == BUILD_MAP) ? (size + size) : size);
  for (int i = size - 1; i >= 0; i--) {
    if (instr->GetOpCode() != BUILD_MAP) {
      opnds[i] = PopStack();
    } else {
      opnds[i + i] = PopStack();
      opnds[i + i + 1] = PopStack();
    }
  }
  ir::NodePtr node = std::make_shared<ir::BuildNode>(instr->GetOpCode(), opnds);
  node->SetDebugInfo(GetNodeDebugInfo(instr));
  PushStack(node);
}

void ByteCodeParser::ParseLoadAttr(const InstrPtr &instr) {
  auto attr_name = py::cast<py::object>(PyTuple_GET_ITEM(code_.co_names, instr->GetArg()));
  ir::ValuePtr attr = std::make_shared<ir::Value>(attr_name, attr_name.cast<std::string>(), ir::kScopeName);
  ir::NodePtr node = std::make_shared<ir::LoadFieldNode>(instr->GetOpCode(), PopStack(), attr);
  node->SetDebugInfo(GetNodeDebugInfo(instr));
  PushStack(node);
}

void ByteCodeParser::ParseImport(const InstrPtr &instr) {
  ir::NodePtrList opnds = {PopStack()};
  if (instr->GetOpCode() == IMPORT_NAME) {
    opnds.insert(opnds.begin(), PopStack());
  }
  if (instr->GetOpCode() != IMPORT_STAR) {
    auto attr_name = py::cast<py::object>(PyTuple_GET_ITEM(code_.co_names, instr->GetArg()));
    ir::ValuePtr attr = std::make_shared<ir::Value>(attr_name, attr_name.cast<std::string>(), ir::kScopeName);
    opnds.push_back(attr);
  }
  ir::NodePtr node = std::make_shared<ir::NaryOperation>(instr->GetOpCode(), opnds);
  node->SetDebugInfo(GetNodeDebugInfo(instr));
  PushStack(node);
}

void ByteCodeParser::ParseLoadGlobal(const InstrPtr &instr) {
  py::object name = py::cast<py::object>(PyTuple_GET_ITEM(code_.co_names, instr->GetArg()));
  bool is_global = globals_.contains(name);
  py::object global = is_global ? globals_[name] : builtins_[name];
  ir::Scope scope = is_global ? ir::kScopeGlobal : ir::kScopeBuiltIn;
  ir::ValuePtr value = std::make_shared<ir::Value>(global, name.cast<std::string>(), scope);
  ir::NodePtr node = std::make_shared<ir::LoadValueNode>(instr->GetOpCode(), value);
  node->SetDebugInfo(GetNodeDebugInfo(instr));
  PushStack(node);
}

void ByteCodeParser::ParseIsOp(const InstrPtr &instr) {
  auto right = PopStack();
  auto left = PopStack();
  ir::IsNodePtr node = std::make_shared<ir::IsNode>(left, right, instr->GetArg());
  node->SetDebugInfo(GetNodeDebugInfo(instr));
  PushStack(node);
}

void ByteCodeParser::ParseContainsOp(const InstrPtr &instr) {
  auto right = PopStack();
  auto left = PopStack();
  ir::ContainsNodePtr node = std::make_shared<ir::ContainsNode>(left, right, instr->GetArg());
  node->SetDebugInfo(GetNodeDebugInfo(instr));
  PushStack(node);
}

void ByteCodeParser::ParseLoadFast(const InstrPtr &instr) {
  auto name = py::cast<py::object>(PyTuple_GET_ITEM(code_.co_varnames, instr->GetArg()));
  ir::ValuePtr value = std::make_shared<ir::Value>(name, name.cast<std::string>(), ir::kScopeLocal);
  ir::NodePtr node = std::make_shared<ir::LoadValueNode>(instr->GetOpCode(), value);
  node->SetDebugInfo(GetNodeDebugInfo(instr));
  PushStack(node);
}

void ByteCodeParser::ParseStoreName(const InstrPtr &instr) {
  ir::OpCode op_code = instr->GetOpCode();
  bool is_free_var = (instr->GetArg() > PyTuple_GET_SIZE(code_.co_cellvars));
  auto names = (op_code == STORE_FAST)
                 ? code_.co_varnames
                 : (op_code == STORE_DEREF) ? (is_free_var ? code_.co_freevars : code_.co_cellvars) : code_.co_names;
  int index = is_free_var ? instr->GetArg() - PyTuple_GET_SIZE(code_.co_cellvars) : instr->GetArg();
  py::object name = py::cast<py::object>(PyTuple_GET_ITEM(names, index));
  auto scope = (op_code == STORE_FAST)
                 ? ir::kScopeLocal
                 : (op_code == STORE_DEREF) ? (is_free_var ? ir::kScopeFreeVar : ir::kScopeCellVar) : ir::kScopeName;
  ir::ValuePtr value = std::make_shared<ir::Value>(name, name.cast<std::string>(), scope);
  auto top = PopStack();
  ir::NodePtr node = std::make_shared<ir::StoreNode>(op_code, top, value);
  node->SetDebugInfo(GetNodeDebugInfo(instr));
  SaveNode(node);
}

void ByteCodeParser::ParseStoreSubscr(const InstrPtr &instr) {
  auto subscr = PopStack();
  auto base = PopStack();
  ir::NodePtr target = std::make_shared<ir::SubscrNode>(base, subscr);
  auto source = PopStack();
  ir::NodePtr node = std::make_shared<ir::StoreNode>(instr->GetOpCode(), source, target);
  node->SetDebugInfo(GetNodeDebugInfo(instr));
  SaveNode(node);
}

void ByteCodeParser::ParseStoreAttr(const InstrPtr &instr) {
  py::object name = py::cast<py::object>(PyTuple_GET_ITEM(code_.co_names, instr->GetArg()));
  ir::NodePtr target = std::make_shared<ir::Value>(name, ir::kScopeName);
  target = std::make_shared<ir::AttrNode>(PopStack(), target);
  auto source = PopStack();
  ir::NodePtr node = std::make_shared<ir::StoreNode>(instr->GetOpCode(), source, target);
  node->SetDebugInfo(GetNodeDebugInfo(instr));
  SaveNode(node);
}

// Process function call, eg : f1(x, y) ...
void ByteCodeParser::ParseCallFunction(const InstrPtr &instr) {
  // The number of args of the called function object, except CALL_FUNCTION_EX
  // CALL_FUNCTION_EX ：size is the flag bit of whether it has kwargs
  int size = instr->GetArg();
  if (instr->GetOpCode() == CALL_FUNCTION_EX) {
    // The Value of size must be 0 or 1
    // the number of parameters is 2 when flag is 1
    // Otherwise, the number of parameters is 1
    MS_EXCEPTION_IF_CHECK_FAIL((size == 0 || size == 1), "The flag of CALL_FUNCTION_EX must be 0 or 1.");
    size = (size & 0x1) ? 2 : 1;
  }
  // The tuple of keys occupies a position
  if (instr->GetOpCode() == CALL_FUNCTION_KW) {
    size++;
  }
  // The function object occupies a position
  size++;
  ir::NodePtrList args;
  for (int index = 0; index < size; index++) {
    args.push_back(PopStack());
  }
  std::reverse(args.begin(), args.end());
  ir::NodePtr node = std::make_shared<ir::CallNode>(instr->GetOpCode(), args);
  node->SetDebugInfo(GetNodeDebugInfo(instr));
  PushStack(node);
}

void ByteCodeParser::ParseLoadClosure(const InstrPtr &instr) {
  size_t cell_var_size = PyTuple_GET_SIZE(code_.co_cellvars);
  size_t index = instr->GetArg();
  ir::NodePtr opnd;
  // if index is less than the length of co_cellvars, then closure is co_cellvars[index]
  // else closure is co_freevars[index - len(co_cellvars)]
  if (cell_var_size > index) {
    auto name = py::cast<py::object>(PyTuple_GET_ITEM(code_.co_cellvars, index));
    opnd = std::make_shared<ir::Value>(name, name.cast<std::string>(), ir::kScopeCellVar);
  } else {
    index -= cell_var_size;
    auto name = py::cast<std::string>(PyTuple_GET_ITEM(code_.co_freevars, index));
    opnd = std::make_shared<ir::Value>(clousre_[index], name, ir::kScopeClousre);
  }
  ir::NodePtr node = std::make_shared<ir::LoadValueNode>(instr->GetOpCode(), opnd);
  node->SetDebugInfo(GetNodeDebugInfo(instr));
  PushStack(node);
}

// This opcode performs several operations before a with block starts.
// First, it loads __exit__() from the context manager and pushes it onto the stack for later use by WITH_EXCEPT_START.
// Then, __enter__() is called, and a finally block pointing to delta is pushed.
// Finally, the result of calling the __enter__() method is pushed onto the stack.
// The next opcode will either ignore it (POP_TOP), or store it in (a) variable(s) (STORE_FAST, STORE_NAME, or
// UNPACK_SEQUENCE).
void ByteCodeParser::ParseSetupWith(const InstrPtr &instr) {
  auto node = std::make_shared<ir::JumpNode>(instr->GetOpCode(), PopStack(), nullptr);
  int target_offset = instr->GetArg() + instr->GetOffset() + 2;
  jump_nodes_map_[target_offset].push_back(node);
  node->SetDebugInfo(GetNodeDebugInfo(instr));
  SaveNode(node);
  ir::NodePtr ph = std::make_shared<ir::PlaceHolder>("SETUP_WITH EXIT");
  PushStack(ph);
  ph = std::make_shared<ir::PlaceHolder>("SETUP_WITH FINALLY BLOCK");
  PushStack(ph);
  ph = std::make_shared<ir::PlaceHolder>("SETUP_WITH ENTER");
  PushStack(ph);
}

void ByteCodeParser::ParseFormatValue(const InstrPtr &instr) {
  ir::NodePtrList opnds;
  opnds.push_back(PopStack());
  if ((instr->GetArg() & 0x04) == 0x04) {
    opnds.push_back(PopStack());
  }
  ir::NodePtr node = std::make_shared<ir::FormatNode>(opnds, instr->GetArg());
  node->SetDebugInfo(GetNodeDebugInfo(instr));
  PushStack(node);
}

void ByteCodeParser::ParseLoadMethod(const InstrPtr &instr) {
  auto name = py::cast<py::object>(PyTuple_GET_ITEM(code_.co_names, instr->GetArg()));
  ir::ValuePtr method = std::make_shared<ir::Value>(name, name.cast<std::string>(), ir::kScopeName);
  ir::NodePtr node = std::make_shared<ir::LoadFieldNode>(instr->GetOpCode(), PopStack(), method);
  node->SetDebugInfo(GetNodeDebugInfo(instr));
  PushStack(node);
}

void ByteCodeParser::ParseContainerUpdate(const InstrPtr &instr) {
  auto right = PopStack();
  MS_EXCEPTION_IF_CHECK_FAIL((instr->GetArg() == 1), "Not Excepted Update.");
  auto left = PopStack();
  ir::NodePtr node = std::make_shared<ir::UpdateNode>(instr->GetOpCode(), left, right, instr->GetArg());
  node->SetDebugInfo(GetNodeDebugInfo(instr));
  PushStack(node);
}

void ByteCodeParser::ParseNoArgOperation(const InstrPtr &instr) {
  ir::NodePtr node = std::make_shared<ir::NaryOperation>(instr->GetOpCode());
  node->SetDebugInfo(GetNodeDebugInfo(instr));
  PushStack(node);
}

void ByteCodeParser::ParseWithExceptStart(const InstrPtr &instr) {
  auto top = PopStack();
  bool is_exit = top->isa<ir::PlaceHolder>() && top->cast<ir::PlaceHolderPtr>()->GetTag() == "SETUP_WITH FINALLY BLOCK";
  MS_EXCEPTION_IF_CHECK_FAIL(is_exit, "Not Excepted args for with except start.");
  for (size_t index = 0; index < 6; index++) {
    ir::NodePtr node = std::make_shared<ir::PlaceHolder>("WITH_EXCEPT_START " + std::to_string(index));
    PushStack(node);
  }
  ir::NodePtr node = std::make_shared<ir::NaryOperation>(instr->GetOpCode());
  node->SetDebugInfo(GetNodeDebugInfo(instr));
  SaveNode(node);
}

void ByteCodeParser::ParseGet(const InstrPtr &instr) {
  auto top = PopStack();
  ir::NodePtr node = std::make_shared<ir::GetNode>(instr->GetOpCode(), top);
  node->SetDebugInfo(GetNodeDebugInfo(instr));
  PushStack(node);
}

void ByteCodeParser::ParseLoadAssertError(const InstrPtr &instr) {
  ir::NodePtr node = std::make_shared<ir::NaryOperation>(instr->GetOpCode());
  node->SetDebugInfo(GetNodeDebugInfo(instr));
  PushStack(node);
}

void ByteCodeParser::ParseDeleteSubscr(const InstrPtr &instr) {
  auto subscr = PopStack();
  auto base = PopStack();
  ir::NodePtr opnd = std::make_shared<ir::SubscrNode>(base, subscr);
  ir::NodePtr node = std::make_shared<ir::DeleteNode>(instr->GetOpCode(), opnd);
  node->SetDebugInfo(GetNodeDebugInfo(instr));
  SaveNode(node);
}

void ByteCodeParser::ParseDeleteName(const InstrPtr &instr) {
  auto names = (instr->GetOpCode() == DELETE_FAST)
                 ? code_.co_varnames
                 : (instr->GetOpCode() == DELETE_DEREF) ? code_.co_cellvars : code_.co_names;
  py::object name = py::cast<py::object>(PyTuple_GET_ITEM(names, instr->GetArg()));
  auto scope = (instr->GetOpCode() == DELETE_FAST)
                 ? ir::kScopeLocal
                 : (instr->GetOpCode() == DELETE_DEREF) ? ir::kScopeCellVar : ir::kScopeName;
  ir::NodePtr opnd = std::make_shared<ir::Value>(name, scope);
  ir::NodePtr node = std::make_shared<ir::DeleteNode>(instr->GetOpCode(), opnd);
  node->SetDebugInfo(GetNodeDebugInfo(instr));
  SaveNode(node);
}

void ByteCodeParser::ParseDeleteAttr(const InstrPtr &instr) {
  py::object name = py::cast<py::object>(PyTuple_GET_ITEM(code_.co_names, instr->GetArg()));
  ir::NodePtr opnd = std::make_shared<ir::Value>(name, ir::kScopeName);
  opnd = std::make_shared<ir::AttrNode>(PopStack(), opnd);
  ir::NodePtr node = std::make_shared<ir::DeleteNode>(instr->GetOpCode(), opnd);
  node->SetDebugInfo(GetNodeDebugInfo(instr));
  SaveNode(node);
}

void ByteCodeParser::ParseUnpack(const InstrPtr &instr) {
  ir::NodePtrList args = {PopStack()};
  for (int index = instr->GetArg(); index > 0; index--) {
    auto subscr = std::make_shared<ir::Value>(py::int_(index - 1), ir::kScopeConst);
    ir::NodePtr top = std::make_shared<ir::SubscrNode>(args[0], subscr);
    top = std::make_shared<ir::RefNode>(top);
    PushStack(top);
    args.push_back(top);
  }
  ir::NodePtr unpack = std::make_shared<ir::NaryWithFlagNode>(instr->GetOpCode(), args, instr->GetArg());
  unpack->SetDebugInfo(GetNodeDebugInfo(instr));
  SaveNode(unpack);
}

void ByteCodeParser::ParseRaiseVarargs(const InstrPtr &instr) {
  int flag = instr->GetArg();
  ir::NodePtrList args;
  if (flag == 1 || flag == 2) {
    args.push_back(PopStack());
  }
  if (flag == 2) {
    args.push_back(PopStack());
  }
  ir::NodePtr node = std::make_shared<ir::NaryWithFlagNode>(RAISE_VARARGS, args, flag);
  node->SetDebugInfo(GetNodeDebugInfo(instr));
  SaveNode(node);
}

ir::DebugInfoPtr ByteCodeParser::GetNodeDebugInfo(const InstrPtr &instr) {
  return std::move(std::make_shared<ir::DebugInfo>(instr->GetArgRepr(), func_->GetFileName(), instr->GetStartsLine()));
}
}  // namespace graph
}  // namespace jit
}  // namespace mindspore
