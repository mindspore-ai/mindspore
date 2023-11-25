/**
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
#include "pipeline/jit/pi/graph_compiler/cg/byte_code_generator.h"
#include <memory>
#include <string>
#include <algorithm>
#include "utils/log_adapter.h"

namespace mindspore {
namespace jit {
namespace graph {
// the number of bits per byte
constexpr int bits_per_byte = 8;

#ifndef MAKE_BYTE_CODE_UNIT
#ifdef WORDS_BIGENDIAN
#define MAKE_BYTE_CODE_UNIT(op, arg) (((op) << bits_per_byte) | (arg))
#else
#define MAKE_BYTE_CODE_UNIT(op, arg) ((op) | ((arg) << bits_per_byte))
#endif
#endif

py::object ByteCodeGenerator::GenFunction(const ir::FunctionNodePtr &func) {
  ByteCodeGeneratorPtr generator = std::make_shared<ByteCodeGenerator>();
  return generator->Generate(func);
}

py::object ByteCodeGenerator::Generate(const ir::FunctionNodePtr &func) {
  cell_var_cnt_ = CellVarCounter::GetCount(func);
  co_consts_.append(py::none());
  first_line_no_ = func->GetFirstLineNo();
  func->Sort();
  Visit(func);
  auto byte_code = py::reinterpret_steal<py::object>(
    PyBytes_FromStringAndSize((const char *)co_code_.data(), co_code_.size() * sizeof(co_code_[0])));
  auto lnotab = py::reinterpret_steal<py::object>(
    PyBytes_FromStringAndSize(co_lnotab_.data(), co_lnotab_.size() * sizeof(co_lnotab_[0])));
  auto var_names = py::cast<py::tuple>(co_var_names_);
  auto consts = py::cast<py::tuple>(co_consts_);
  auto names = py::cast<py::tuple>(co_names_);
  auto free_vars = py::cast<py::tuple>(co_free_vars_);
  auto cell_vars = py::cast<py::tuple>(co_cell_vars_);
  PyCodeObject *code = PyCode_New(func->GetPosArgsCnt(), func->GetKwOnlyArgsCnt(), var_names.size(),
                                  func->GetStackSize(), func->GetFlags(), byte_code.ptr(), consts.ptr(), names.ptr(),
                                  var_names.ptr(), free_vars.ptr(), cell_vars.ptr(), py::str(func->GetName()).ptr(),
                                  py::str(names).ptr(), func->GetFirstLineNo(), lnotab.ptr());
  globals_[py::str("__builtins__")] = builtins_.ptr();
  auto function = py::reinterpret_steal<py::object>(PyFunction_New(reinterpret_cast<PyObject *>(code), globals_.ptr()));
  Py_DECREF(code);
  auto tuple = py::cast<py::tuple>(defaults_);
  (void)PyFunction_SetDefaults(function.ptr(), tuple.ptr());
  tuple = py::cast<py::tuple>(clousre_);
  (void)PyFunction_SetClosure(function.ptr(), tuple.ptr());
  return function;
}

void ByteCodeGenerator::Visit_(const ir::ParameterPtr &node) {
  const std::string name = node->GetName();
  MS_EXCEPTION_IF_CHECK_FAIL((co_var_names_map_.find(name) == co_var_names_map_.end()),
                             "Duplicate parameter name " + name + ".");
  co_var_names_map_[name] = co_var_names_.size();
  co_var_names_.append(py::str(name));
  ir::NodePtr default_value = node->GetDefaultValue();
  if (default_value != nullptr) {
    if (node->GetCategory() == ir::Parameter::KEYWORD_ONLY) {
      kwdefaults_[co_var_names_[co_var_names_map_[name]]] = default_value;
    } else {
      MS_EXCEPTION_IF_CHECK_FAIL(node->GetCategory() == 0, "Error category of parameter.");
      defaults_.append(default_value->cast<ir::ValuePtr>()->GetValue());
    }
  }
}

#define DEFINE_UN_NODE_VISIT_(OP)                  \
  void ByteCodeGenerator::Visit_(const OP &node) { \
    Visit(node->GetArg());                         \
    CheckInstrOffset(node);                        \
    GenerateInstr(node->GetOpCode());              \
    SetStartsLine(node);                           \
  }

DEFINE_UN_NODE_VISIT_(ir::UnaryOperationPtr)
DEFINE_UN_NODE_VISIT_(ir::NegativeNodePtr)
DEFINE_UN_NODE_VISIT_(ir::NotNodePtr)
DEFINE_UN_NODE_VISIT_(ir::InvertNodePtr)
DEFINE_UN_NODE_VISIT_(ir::ReturnNodePtr)
DEFINE_UN_NODE_VISIT_(ir::CastNodePtr)
DEFINE_UN_NODE_VISIT_(ir::GetNodePtr)

#define DEFINE_BIN_NODE_VISIT_(OP)                 \
  void ByteCodeGenerator::Visit_(const OP &node) { \
    Visit(node->GetLeftArg());                     \
    Visit(node->GetRightArg());                    \
    CheckInstrOffset(node);                        \
    GenerateInstr(node->GetOpCode());              \
    SetStartsLine(node);                           \
  }

DEFINE_BIN_NODE_VISIT_(ir::BinaryOperationPtr)
DEFINE_BIN_NODE_VISIT_(ir::AddNodePtr)
DEFINE_BIN_NODE_VISIT_(ir::SubNodePtr)
DEFINE_BIN_NODE_VISIT_(ir::MulNodePtr)
DEFINE_BIN_NODE_VISIT_(ir::DivNodePtr)
DEFINE_BIN_NODE_VISIT_(ir::BitwiseNodePtr)

void ByteCodeGenerator::Visit_(const ir::ValuePtr &node) { (void)GetValueIndex(node); }

void ByteCodeGenerator::Visit_(const ir::NaryOperationPtr &node) {
  VISIT_NODE_LIST(node->GetArgs())
  CheckInstrOffset(node);
  GenerateInstr(node->GetOpCode(), node->GetArgsCnt());
  SetStartsLine(node);
}

void ByteCodeGenerator::Visit_(const ir::DeleteNodePtr &node) {
  Visit(node->GetArg());
  int arg = 0;
  if (node->GetOpCode() != DELETE_SUBSCR) {
    MS_EXCEPTION_IF_CHECK_FAIL(node->GetArg()->isa<ir::Value>(), "Expect delete a value.");
    arg = GetValueIndex(node->GetArg()->cast<ir::ValuePtr>());
  }
  CheckInstrOffset(node);
  GenerateInstr(node->GetOpCode(), arg);
  SetStartsLine(node);
}

void ByteCodeGenerator::Visit_(const ir::FormatNodePtr &node) {
  VISIT_NODE_LIST(node->GetArgs())
  CheckInstrOffset(node);
  GenerateInstr(node->GetOpCode(), node->GetFormatType());
  SetStartsLine(node);
}

void ByteCodeGenerator::Visit_(const ir::IsNodePtr &node) {
  Visit(node->GetLeftArg());
  Visit(node->GetRightArg());
  CheckInstrOffset(node);
  GenerateInstr(node->GetOpCode(), node->IsInvert());
  SetStartsLine(node);
}

void ByteCodeGenerator::Visit_(const ir::ContainsNodePtr &node) {
  Visit(node->GetLeftArg());
  Visit(node->GetRightArg());
  CheckInstrOffset(node);
  GenerateInstr(node->GetOpCode(), node->IsInvert());
  SetStartsLine(node);
}

void ByteCodeGenerator::Visit_(const ir::StoreNodePtr &node) {
  Visit(node->GetLeftArg());
  Visit(node->GetRightArg());
  ir::NodePtr target = node->GetRightArg();
  if (target->isa<ir::AttrNode>()) {
    target = target->cast<ir::AttrNodePtr>()->GetAttr();
  }
  int arg = 0;
  if (!target->isa<ir::SubscrNode>()) {
    MS_EXCEPTION_IF_CHECK_FAIL(target->isa<ir::Value>(), "Expect store to a var.");
    arg = GetValueIndex(target->cast<ir::ValuePtr>());
  }
  CheckInstrOffset(node);
  GenerateInstr(node->GetOpCode(), arg);
  SetStartsLine(node);
}

void ByteCodeGenerator::Visit_(const ir::JumpNodePtr &node) {
  ir::IRVisitor::Visit_(node);
  ir::OpCode op = node->GetOpCode();
  int arg = node->GetRightArg()->GetOffset() * 2;
  if (op == JUMP_FORWARD || op == FOR_ITER) {
    arg -= (node->GetOffset() + 1) * 2;
  }
  CheckInstrOffset(node);
  GenerateInstr(node->GetOpCode(), arg);
  SetStartsLine(node);
}

void ByteCodeGenerator::Visit_(const ir::CompareNodePtr &node) {
  Visit(node->GetLeftArg());
  Visit(node->GetRightArg());
  CheckInstrOffset(node);
  GenerateInstr(node->GetOpCode(), node->GetInstrArg());
  SetStartsLine(node);
}

void ByteCodeGenerator::Visit_(const ir::UpdateNodePtr &node) {
  Visit(node->GetLeftArg());
  Visit(node->GetRightArg());
  CheckInstrOffset(node);
  GenerateInstr(node->GetOpCode(), node->GetInstrArg());
  SetStartsLine(node);
}

void ByteCodeGenerator::Visit_(const ir::LoadNodePtr &node) {
  VISIT_NODE_LIST(node->GetArgs())
  int arg = GetValueIndex(node->GetArg(1)->cast<ir::ValuePtr>());
  CheckInstrOffset(node);
  GenerateInstr(node->GetOpCode(), arg);
  SetStartsLine(node);
}

void ByteCodeGenerator::Visit_(const ir::BuildNodePtr &node) {
  VISIT_NODE_LIST(node->GetArgs())
  CheckInstrOffset(node);
  int arg = node->GetArgsCnt();
  if (node->GetOpCode() == BUILD_CONST_KEY_MAP) {
    arg--;
  }
  GenerateInstr(node->GetOpCode(), arg);
  SetStartsLine(node);
}

void ByteCodeGenerator::Visit_(const ir::CallNodePtr &node) {
  VISIT_NODE_LIST(node->GetArgs())
  int arg = node->GetArgsCnt() - 1;
  if (node->GetOpCode() == CALL_FUNCTION_KW || node->GetOpCode() == CALL_FUNCTION_EX) {
    arg--;
  }
  CheckInstrOffset(node);
  GenerateInstr(node->GetOpCode(), arg);
  SetStartsLine(node);
}

void ByteCodeGenerator::Visit_(const ir::NaryWithFlagNodePtr &node) {
  VISIT_NODE_LIST(node->GetArgs())
  CheckInstrOffset(node);
  GenerateInstr(node->GetOpCode(), node->GetFlag());
  SetStartsLine(node);
}

int ByteCodeGenerator::GetValueIndex(const ir::ValuePtr &node) {
  auto scope = node->GetScope();
  MS_EXCEPTION_IF_CHECK_FAIL(scope_inquire_map_.find(scope) != scope_inquire_map_.end(),
                             "Invalid scope in " + node->ToString());
  auto name_map = *scope_inquire_map_.at(scope);
  auto name = node->GetName();
  if (name_map.find(name) != name_map.end()) {
    return name_map.at(name);
  }
  auto values = scope_value_list_.at(scope);
  values.append(node->GetValue());
  if (scope_name_list_.find(scope) == scope_name_list_.end()) {
    name_map[name] = values.size() - 1;
  } else {
    name_map[name] = scope_name_list_.at(scope).size();
    scope_name_list_.at(scope).append(py::str(name));
  }
  return name_map.at(name);
}

void ByteCodeGenerator::CheckInstrOffset(const ir::NodePtr &node) {
  MS_EXCEPTION_IF_CHECK_FAIL(
    node->GetOffset() - (node->NeedExtInstr() ? 1 : 0) == co_code_.size(),
    "The offset of " + node->GetNodeName() + "(%" + std::to_string(node->GetNodeId()) + ") is not expected.");
}

bool IsExtendedArg(int arg) { return (arg >> bits_per_byte) > 0; }

void ByteCodeGenerator::GenerateInstr(ir::OpCode op, int arg) {
  if (IsExtendedArg(arg)) {
    int ext_arg = arg >> bits_per_byte;
    co_code_.push_back(MAKE_BYTE_CODE_UNIT(EXTENDED_ARG, ext_arg));
    arg &= 0xff;
  }
  co_code_.push_back(MAKE_BYTE_CODE_UNIT(op, arg));
}

// co_lnotab_ : A string encoding the mapping from bytecode offsets to line numbers.
// Elements are value pairs
// Value pair : the first one is offset of bytecode
//              the second one is the increment of the line number relative to the previous value pair.
// For example :
// co_firstlineno  (8)
// co_lnotab_ = {(0, 1), (6, 1), (9, 2)}
// (0, 1) ----> 0 : the first bytecode
//              1 : the line no. of first bytecode is (1 + 8) = 9
// (6, 1) ----> 6 : the seventh bytecode, means The line number of the second to sixth bytecodes is 9
//              1 : the line no. of seventh bytecode is (1 + 9) = 10
// (9, 2) ----> 9 : the tenth bytecode, means The line number of the eighth and ninth bytecodes is 10
//              1 : the line no. of tenth is bytecode (2 + 10) = 12
void ByteCodeGenerator::SetStartsLine(const ir::NodePtr &node) {
  int new_line_no = node->GetDebugInfo()->GetLineNo();
  if (new_line_no == 0) {
    return;
  }
  int dis = 0;
  int inc = new_line_no;
  if (co_lnotab_.empty()) {
    inc -= first_line_no_;
  } else {
    MS_EXCEPTION_IF_CHECK_FAIL(last_starts_instr_ != nullptr, "last_starts_instr_ should not be nullptr.");
    dis = sizeof(_Py_CODEUNIT) * (node->GetOffset() - last_starts_instr_->GetOffset());
    inc -= last_starts_instr_->GetDebugInfo()->GetLineNo();
  }
  last_starts_instr_ = node;
  co_lnotab_.push_back(dis);
  co_lnotab_.push_back(inc);
}
}  // namespace graph
}  // namespace jit
}  // namespace mindspore
