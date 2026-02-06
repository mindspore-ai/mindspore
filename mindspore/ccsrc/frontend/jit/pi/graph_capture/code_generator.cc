/**
 * Copyright 2024-2025 Huawei Technologies Co., Ltd
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
#include "frontend/jit/pi/graph_capture/code_generator.h"
#include <set>
#include <regex>
#include "frontend/jit/pi/graph_capture/local_liveness.h"
#include "frontend/jit/pi/graph_capture/graph.h"
#include "frontend/jit/pi/graph_capture/cfg.h"
#include "frontend/jit/pi/graph_capture/side_effect.h"
#include "frontend/jit/pi/utils/utils.h"
#include "frontend/jit/pi/runtime.h"
#include "frontend/jit/pi/external.h"
#include "frontend/jit/pi/graph_compiler/compiler.h"
#include "utils/convert_utils_base.h"

namespace mindspore {
namespace pijit {
constexpr const size_t MoveEightBits = 8;

// https://github.com/python/cpython/blob/main/Python/assemble.c#L158
class ExceptionTableEncoder {
 public:
  void WriteByte(uint32_t i) { bytes_ << static_cast<uint8_t>(i); }
  void WriteVariant(int value, int mask_bits);
  void WriteItem(const ExceptionTableItem &item);
  py::bytes Result() {
    std::string r = bytes_.str();  // here must be copy from stream ensure it's complete c str
    return py::bytes(r);
  }

 private:
  std::stringstream bytes_;
};

// https://github.com/python/cpython/blob/main/Python/assemble.c#L340
class LocationTableEncoder {
 public:
  // copy from _PyCodeLocationInfoKind
  enum Kind {
    kCodeLocationShort0 = 0,
    kCodeLocationOneLine0 = 10,
    kCodeLocationOneLine1 = 11,
    kCodeLocationOneLine2 = 12,
    kCodeLocationNoColumns = 13,
    kCodeLocationLong = 14,
    kCodeLocationNone = 15
  };
  explicit LocationTableEncoder(int first_line) : pre_line_(first_line) {}
  void WriteByte(uint32_t i) { bytes_ << static_cast<uint8_t>(i); }
  py::bytes Result() {
    std::string r = bytes_.str();  // here must be copy from stream ensure it's complete c str
    return py::bytes(r);
  }

  // |- flag:1 -|- code:4 -|- len:3 -|
  void WriteItemStart(int code, int len);
  void WriteVariant(uint32_t value);
  void WriteSignedVariant(int32_t value);
  void Add(int bci_delta, const CodeLocation &loc);
  void AddEntry(int offset, const CodeLocation &loc);
  void NoneEntry(int offset);
  void NoColumn(int offset, int line_delta);
  void OneLineEntry(int offset, int line_delta, int column, int end_column);
  void ShortEntry(int offset, int column, int end_column);
  void LongEntry(int offset, const CodeLocation &loc);

 private:
  std::stringstream bytes_;
  int pre_line_;
};

class GraphParameterBuilder {
 public:
  static std::string Key(int, ValueNode *n);

  void Init(const std::vector<ValueNode *> &args);
  void Build(const std::unordered_map<ValueNode *, int> &locals);

  std::vector<ValueNode *> args_;
  std::vector<std::unique_ptr<Instr>> load_;  // load parameters and store parameters to global, for caller
  std::vector<std::unique_ptr<Instr>> dele_;  // delete global parameters, for caller
  std::vector<std::unique_ptr<Instr>> sort_;  // load global parameter and store to locals, for callee
};

static bool FindBlock(int start_bci, const CFG *cfg, int *target_bci, int *stack_effect);
std::string PrintInstr(const std::vector<std::unique_ptr<Instr>> &list);
std::string PrintNodeSet(const NodeSet &);

bool FindBlock(int bci, const CFG *cfg) {
  int end_bci;
  int stack_effect;
  return FindBlock(bci, cfg, &end_bci, &stack_effect);
}

std::string GenerateObjectKey(const py::object &value) {
  PyTypeObject *tp = Py_TYPE(value.ptr());
  std::stringstream s;
  s << (tp->tp_name ? tp->tp_name : "<unnamed>");
  if (tp == &PyFunction_Type) {
    s << "[" << PyUnicode_AsUTF8(reinterpret_cast<PyFunctionObject *>(value.ptr())->func_qualname) << "]";
  }
  if (tp == &PyModule_Type) {
    s << "[" << PyModule_GetName(value.ptr()) << "]";
  }
  s << "<" << value.ptr() << ">";
  return s.str();
}

void MapAdd(const py::dict &dict, const std::string &key, const py::object &value, std::string *rename) {
  py::str key_object(key);
  PyObject *old = PyDict_GetItem(dict.ptr(), key_object.ptr());
  if (old == value.ptr()) {
    return;
  }
  if (old == nullptr) {
    PyDict_SetItem(dict.ptr(), key_object.ptr(), value.ptr());
    return;
  }
  if (rename != nullptr) {
    std::string new_key = GenerateObjectKey(value);
    if (new_key != key) {
      PyDict_SetItem(dict.ptr(), py::str(new_key).ptr(), value.ptr());
      *rename = new_key;
      return;
    }
  }
  MS_LOG(INTERNAL_EXCEPTION) << "duplicate dict value, key: " << key << ", old value at " << old << ": "
                             << std::string(py::str(old)) << " -> new value at " << value.ptr() << ": "
                             << std::string(py::str(value.ptr()));
}

static int GetOpcodeMaxStackEffect(int op, int arg, bool jump) {
#if IS_PYTHON_3_11_PLUS && !IS_PYTHON_3_12_PLUS
  if (op == PRECALL) {
    return 0;
  }
  if (op == CALL) {
    return -(arg + 1);
  }
  if (op == CALL_FUNCTION_EX) {
    return -2 - static_cast<bool>(arg & 1);
  }
#endif

  int off = PyCompile_OpcodeStackEffect(op, arg);
  if (op == NOP || op == EXTENDED_ARG) {
    return 0;
  }
  if (op == END_FINALLY) {
    return -1;
  }
  return off;
}

int CodeGenerator::ExceptionStackRequired(const Code &ccode) {
  int sp = 0;
  if (ccode.co_exceptiontable.empty()) {
    return sp;
  }
  // python3.11+
  for (const auto &item : ccode.co_exceptiontable) {
    sp = std::max(item.stack_ + item.lasti_, sp);
  }
  return sp;
}

int CodeGenerator::CalculateStackSize(const Code &ccode, int sp) {
  const InstructionList &list = ccode.co_code;
  std::unordered_map<Instr *, int> blocks;
  int max_depth = 0;
  int flag = 0;
  for (size_t i = 0; i < list.size(); ++i) {
    Instr *instr = list[i].get();
    int op = instr->op();
    int arg = instr->arg();
    Instr *jump = instr->extra_jump();
    auto iter = blocks.find(instr);
    if (iter != blocks.end()) {
      flag = 0;
      sp = iter->second;
    } else if (flag == 1) {
      continue;
    }
    if (op == RAISE_VARARGS || op == RETURN_VALUE || op == RERAISE) {
      flag = 1;
    }
    if (jump != nullptr) {
      iter = blocks.find(jump);
      int jump_sp = sp + GetOpcodeMaxStackEffect(op, arg, true);
      blocks[jump] = (iter == blocks.end()) ? jump_sp : std::max(iter->second, jump_sp);
    }
    sp += GetOpcodeMaxStackEffect(op, arg, false);
    max_depth = std::max(sp, max_depth);
  }
  return sp < 0 ? -1 : max_depth + ExceptionStackRequired(ccode);
}

// reset bci, reset jump offset
static void CalculateOffset(const std::vector<std::unique_ptr<Instr>> &list) {
  bool re_calc;
  do {
    re_calc = false;
    int bci = 0;
    for (const auto &i : list) {
      i->set_bci(bci);
      bci += i->InstrSize();
    }
    for (const auto &i : list) {
      int isize = i->InstrSize();
      Instr *tar = i->extra_jump();
      if (tar) {
        int oparg = Opcode(i->op()).JumpOffset(i->bci(), tar->bci());
        i->set_arg(oparg);
        re_calc |= isize != i->InstrSize();
      }
    }
  } while (re_calc);
}

std::string PrintCodeBytes(const std::string &code_bytes) {
  std::stringstream s;
  for (auto iter = code_bytes.begin(), end = code_bytes.end(); iter != end; iter += sizeof(uint16_t)) {
    Opcode op(static_cast<int>(static_cast<uint8_t>(*iter)));
    int arg = static_cast<int>(static_cast<uint8_t>(*(iter + 1)));
    s << "code: " << op << " " << op.name() << " " << arg << std::endl;
  }
  return s.str();
}

std::pair<py::bytes, py::bytes> CodeGenerator::ConvertToCodeBytes(const Code &ccode) {
  const auto &list = ccode.co_code;
  int first_line = ccode.co_firstlineno;
  std::vector<char> co_lnotab;
  std::stringstream co_code;

  CalculateOffset(list);

  MS_LOG(DEBUG) << "after calculate offset, bci is: " << std::endl << PrintInstr(list);

  int line = first_line > 0 ? first_line : 0;
  int bci = 0;
  bool start_flag = IS_PYTHON_3_10_PLUS;
  for (const auto &i : list) {
    int addr_off = sizeof(_Py_CODEUNIT) * (i->bci() - bci);
    int line_off = i->line() - line;
    if (i->line() != -1 && line_off > 0 && line_off < INT8_MAX && addr_off < INT8_MAX) {
      if (start_flag) {
        --line_off;
        start_flag = false;
      }
      co_lnotab.push_back(addr_off);
      co_lnotab.push_back(line_off);
      bci = i->bci();
      line = i->line();
    }
    Opcode opcode(i->op());
    unsigned int oparg = opcode.HasArg() ? i->arg() : 0;
    for (int c = (oparg > 0xffffff) + (oparg > 0xffff) + (oparg > 0xff); c > 0; --c) {
      co_code << static_cast<uint8_t>(EXTENDED_ARG) << static_cast<uint8_t>(oparg >> (MoveEightBits * c));
    }
    co_code << static_cast<uint8_t>(i->op()) << static_cast<uint8_t>(oparg & 0xffu);
    for (int c = opcode.InstrSize() - 1; c > 0; --c) {
      co_code << static_cast<uint8_t>(CACHE) << static_cast<uint8_t>(0);
    }
  }
#if IS_PYTHON_3_10_PLUS
  co_lnotab.push_back(sizeof(_Py_CODEUNIT) * (list.size() - static_cast<size_t>(bci)));
  co_lnotab.push_back(1);
#endif
  std::string code_bytes = co_code.str();
  MS_LOG(DEBUG) << "encode bytes:" << std::endl << PrintCodeBytes(code_bytes);
  return {py::bytes(code_bytes), py::bytes(co_lnotab.data(), co_lnotab.size())};
}

py::bytes EncodeExceptionTable(const CodeGenerator::Code &ccode) {
  ExceptionTableEncoder encoder;
  for (const auto &i : ccode.co_exceptiontable) {
    ExceptionTableItem item{i.begin_->bci(), i.end_->bci() + i.end_->InstrSize(), i.jump_->bci(), i.stack_, i.lasti_};
    MS_LOG(DEBUG) << "add item: " << item;
    encoder.WriteItem(item);
  }
  return encoder.Result();
}

py::bytes EncodeLocationTable(const CodeGenerator::Code &ccode) {
  LocationTableEncoder encoder(ccode.co_firstlineno);
  CodeLocation loc;
  int bci_delta = 0;
  for (const auto &i : ccode.co_code) {
    const CodeLocation &cur = i->location();
    if (loc != cur) {
      encoder.Add(bci_delta, loc);
      loc = cur;
      bci_delta = 0;
    }
    bci_delta += i->InstrSize();
  }
  encoder.Add(bci_delta, loc);
  return encoder.Result();
}

static void SetNamedInstrIndex(const std::unique_ptr<Instr> &i, std::unordered_map<std::string, int> *co_names) {
  if (!Opcode(i->op()).HasName()) {
    return;
  }
  int arg;
  auto iter = co_names->find(i->name());
  if (iter != co_names->end()) {
    arg = iter->second;
  } else {
    arg = SizeToInt(co_names->size());
    co_names->insert({i->name(), arg});
  }
#if IS_PYTHON_3_11_PLUS
  if (i->op() == LOAD_GLOBAL) {
    arg = arg << 1;
    arg |= static_cast<bool>(i->arg() & 1);  // PUSH_NULL
  }
#endif
#if IS_PYTHON_3_12_PLUS
  if (i->op() == LOAD_ATTR) {
    arg = arg << 1;
    arg |= static_cast<bool>(i->arg() & 1);  // PUSH_NULL
  }
#endif
  i->set_arg(arg);
}

static void SetLoadConstIndex(const std::unique_ptr<Instr> &i, const py::dict &consts) {
  if (!Opcode(i->op()).HasConst()) {
    return;
  }
  PyObject *co_consts = consts.ptr();
  PyObject *cnst = i->cnst().ptr();
  MS_EXCEPTION_IF_CHECK_FAIL(cnst != nullptr, "LOAD_CONST instruction not set object");

  PyObject *key = _PyCode_ConstantKey(cnst);
  if (key != nullptr) {
    PyObject *index = PyDict_GetItem(co_consts, key);
    Py_ssize_t arg;
    if (index != nullptr) {
      arg = PyLong_AsLong(index);
    } else {
      arg = PyDict_GET_SIZE(co_consts);
      PyDict_SetItem(co_consts, key, py::int_(arg).ptr());
    }
    i->set_arg(arg);
    Py_DECREF(key);
    if (!PyErr_Occurred()) {
      return;
    }
  }
  throw py::error_already_set();
}

static py::tuple ConstsMapToTuple(const py::dict &consts) {
  const Py_ssize_t size = PyDict_GET_SIZE(consts.ptr());
  py::tuple co_consts(size);

  PyObject *key;
  PyObject *val;
  Py_ssize_t pos = 0;
  while (PyDict_Next(consts.ptr(), &pos, &key, &val)) {
    Py_ssize_t index = PyLong_AsLong(val);
    if (PyTuple_CheckExact(key)) {
      key = PyTuple_GET_ITEM(key, 1);
    }
    Py_INCREF(key);
    PyTuple_SET_ITEM(co_consts.ptr(), index, key);
  }
  return co_consts;
}

static py::tuple NamesMapToTuple(const std::unordered_map<std::string, int> &names) {
  py::tuple co_names(names.size());
  for (const auto &i : names) {
    PyTuple_SET_ITEM(co_names.ptr(), i.second, PyUnicode_FromStringAndSize(i.first.data(), i.first.size()));
  }
  return co_names;
}

static py::object ConvertVector(const std::vector<std::string> &names, bool to_tuple = true) {
  size_t size = names.size();
  PyObject *list = to_tuple ? PyTuple_New(size) : PyList_New(size);
  for (; size > 0; --size) {
    const std::string &n = names[size - 1];
    if (to_tuple) {
      PyTuple_SET_ITEM(list, size - 1, PyUnicode_FromStringAndSize(n.data(), n.size()));
    } else {
      PyList_SET_ITEM(list, size - 1, PyUnicode_FromStringAndSize(n.data(), n.size()));
    }
  }
  return py::reinterpret_steal<py::object>(list);
}

static py::tuple FillVariableName(const std::vector<std::string> &varnames, int nlocals) {
  MS_EXCEPTION_IF_CHECK_FAIL(varnames.size() <= static_cast<size_t>(nlocals), "too small local count !!");
  std::set<std::string> vars;
  py::tuple co_varnames(nlocals);
  int size = SizeToInt(varnames.size());
  for (int i = 0; i < nlocals; ++i) {
    std::string n;
    if (i < size) {
      n = varnames[i];
    } else {
      n = std::to_string(i) + "_local";
    }
    while (vars.find(n) != vars.end()) {
      n = n + "_" + std::to_string(i);
    }
    vars.insert(n);
    PyTuple_SET_ITEM(co_varnames.ptr(), i, PyUnicode_FromStringAndSize(n.data(), n.size()));
  }
  return co_varnames;
}

static std::string AttachCodeID(const std::string &co_name) {
  static size_t id = 0;
  constexpr const char *mark = "I.";
  constexpr const char *reg_mark = "\\d+I.";
  return std::to_string(id++) + mark + std::regex_replace(co_name, std::regex(reg_mark), "");
}

static std::string MakeCompiledName(const std::string &co_name) {
  static size_t id = 0;
  constexpr const char *reg_mark = "<compile\\[\\d+\\]>";
  return "<compile[" + std::to_string(id++) + "]>" + std::regex_replace(co_name, std::regex(reg_mark), "");
}

static std::string MakeBrkName(const std::string &co_name, int bci) {
  constexpr const char *mark = "B.";
  constexpr const char *reg_mark = "\\d+B.";
  return std::to_string(bci) + mark + std::regex_replace(co_name, std::regex(reg_mark), "");
}

static std::unique_ptr<Instr> MakeCallFunInstrPtr(int oparg) {
  return std::make_unique<Instr>(IS_PYTHON_3_11_PLUS ? CALL : CALL_FUNCTION, oparg);
}

py::object CodeGenerator::Transform(const Code &ccode) {
  std::unordered_map<std::string, int> names;
  py::dict consts;
  int co_stacksize;

  for (const auto &i : ccode.co_code) {
    SetNamedInstrIndex(i, &names);
    SetLoadConstIndex(i, consts);
  }
  co_stacksize = CalculateStackSize(ccode);
  if (co_stacksize < 0) {
    MS_LOG(ERROR) << "\n" << PrintInstr(ccode.co_code);
    MS_EXCEPTION_IF_CHECK_FAIL(co_stacksize >= 0, "check instruction list, computer stack size failed");
  }

  std::pair<py::bytes, py::bytes> code_info = ConvertToCodeBytes(ccode);
  py::bytes co_code = std::move(code_info.first);
  py::bytes co_lnotab = std::move(code_info.second);
  py::tuple co_consts = ConstsMapToTuple(consts);
  py::tuple co_names = NamesMapToTuple(names);
  py::object co_varnames = FillVariableName(ccode.co_varnames, ccode.co_nlocals);
  py::object co_freevars = ConvertVector(ccode.co_freevars);
  py::object co_cellvars = ConvertVector(ccode.co_cellvars);
  py::str co_name(AttachCodeID(ccode.co_name));

#if IS_PYTHON_3_11_PLUS
  py::bytes co_linetable = EncodeLocationTable(ccode);
  py::bytes co_exceptiontable = EncodeExceptionTable(ccode);
  py::str co_qualname = ccode.co_qualname.empty() ? co_name : py::str(ccode.co_qualname);
  co_lnotab = co_linetable;
#endif

  PyCodeObject *new_code = PyCode_New(ccode.co_argcount,         // co_argcount
                                      ccode.co_kwonlyargcount,   // co_kwonlyargcount
                                      ccode.co_nlocals,          // co_nlocals
                                      co_stacksize,              // co_stacksize
                                      ccode.co_flags,            // co_flags
                                      co_code.ptr(),             // co_code
                                      co_consts.ptr(),           // co_consts
                                      co_names.ptr(),            // co_names
                                      co_varnames.ptr(),         // co_varnames
                                      co_freevars.ptr(),         // co_freevars
                                      co_cellvars.ptr(),         // co_cellvars
                                      ccode.co_filename.ptr(),   // co_filename
                                      co_name.ptr(),             // co_name
#if IS_PYTHON_3_11_PLUS                                          // format code aligned
                                      co_qualname.ptr(),         // co_qualname
                                      ccode.co_firstlineno,      // co_firstlineno
                                      co_lnotab.ptr(),           // co_linetable
                                      co_exceptiontable.ptr());  // co_exceptiontable
#else
                                      ccode.co_firstlineno,  // co_firstlineno
                                      co_lnotab.ptr());      // co_lnotab
#endif
  if (new_code != nullptr) {
    return py::reinterpret_steal<py::object>(reinterpret_cast<PyObject *>(new_code));
  }
  throw py::error_already_set();
}

std::vector<std::unique_ptr<Instr>> CodeGenerator::ByteCodePrefix() const {
  std::vector<std::unique_ptr<Instr>> prefix;
  if (code_.co_freevars.size() != 0) {
    prefix.push_back(std::make_unique<Instr>(COPY_FREE_VARS, code_.co_freevars.size()));
  }
  const auto make_cell = [](const std::string &name) { return std::make_unique<Instr>(MAKE_CELL, 0, name); };
  std::transform(code_.co_cellvars.begin(), code_.co_cellvars.end(), std::back_inserter(prefix), make_cell);
  prefix.push_back(std::make_unique<Instr>(RESUME));
  return prefix;
}

void CodeGenerator::FixLocalOffset(int invalid_index) {
  int new_index = code_.co_nlocals;
  code_.co_nlocals++;
  for (const auto &i : code_.co_code) {
    if (Opcode(i->op()).IsLocalAccess() && i->arg() == invalid_index) {
      i->set_arg(new_index);
    }
  }
}

void CodeGenerator::FixFreeOffset(const std::vector<std::string> &other_closure_names) {
  // reset offset
  for (const auto &i : code_.co_code) {
    int op = i->op();
    if (Opcode(op).HasFree()) {
      auto iter = std::find(other_closure_names.begin(), other_closure_names.end(), i->name());
      if (iter != other_closure_names.end()) {
        i->set_arg(code_.co_nlocals + static_cast<int>(iter - other_closure_names.begin()));
      } else {
        auto var_iter = std::find(code_.co_varnames.begin(), code_.co_varnames.end(), i->name());
        MS_EXCEPTION_IF_CHECK_FAIL(var_iter != code_.co_varnames.end(), "can't find cell var name: " + i->name());
        i->set_arg(var_iter - code_.co_varnames.begin());
      }
    }
  }
}

void CodeGenerator::FixOffset() {
  if (code_.co_cellvars.empty() && code_.co_freevars.empty()) {
    return;
  }
  // remove duplicate names
  std::vector<std::string> closure_names;
  closure_names.insert(closure_names.end(), code_.co_cellvars.begin(), code_.co_cellvars.end());
  closure_names.insert(closure_names.end(), code_.co_freevars.begin(), code_.co_freevars.end());
  if (!code_.co_cellvars.empty()) {
    for (size_t i = 0, size = code_.co_varnames.size(); i < size; ++i) {
      auto iter = std::find(closure_names.begin(), closure_names.end(), code_.co_varnames[i]);
      if (iter != closure_names.end()) {
        FixLocalOffset(i);
        closure_names.erase(iter);
      }
    }
  }
  FixFreeOffset(closure_names);
}

py::object CodeGenerator::NewCode() {
  EraseUnusedInstr();
#if IS_PYTHON_3_11_PLUS
  auto &instr = code_.co_code;
  auto prefix = ByteCodePrefix();
  instr.insert(instr.begin(), std::make_move_iterator(prefix.begin()), std::make_move_iterator(prefix.end()));
  FixOffset();
#endif
  return Transform(code_);
}

std::vector<std::unique_ptr<Instr>> CodeGenerator::CopyInstr(const std::vector<std::unique_ptr<Instr>> &list,
                                                             size_t start_bci, size_t end_bci, bool erase_invalid_jump,
                                                             bool is_loop_body) {
  std::vector<std::pair<size_t, size_t>> edges;
  std::vector<std::unique_ptr<Instr>> instrs;

  bool insert_nop_to_end = false;
  size_t size = std::min(list.size(), end_bci);
  for (size_t bci = start_bci; bci < size; ++bci) {
    const auto &i = list[bci];
    size_t index = (size_t)i->bci() - start_bci;
    instrs.emplace_back(std::make_unique<Instr>(*i));
    if (i->extra_jump()) {
      size_t tar = i->extra_jump()->bci();
      // If the jump dest inside the loop body points to the beginning of the loop body,
      // modify the dest to the end of the loop body
      if (is_loop_body && tar == start_bci - 1) {
        tar = size;
      }
      bool valid = i->bci() == SizeToInt(bci) && start_bci <= tar && tar <= size;
      if (valid) {
        insert_nop_to_end |= (tar == size);
        edges.push_back({index, tar - start_bci});
      } else if (erase_invalid_jump) {
        i->set_op(Opcode(i->op()).IsNotFall() ? NOP : POP_TOP);
      } else {
        MS_LOG(INTERNAL_EXCEPTION) << "check instruction index failed," << i->bci() << " == " << bci << " && "
                                   << start_bci << " <= " << tar << " && " << tar << " <= " << size;
      }
    }
  }
  if (insert_nop_to_end) {
    instrs.emplace_back(std::make_unique<Instr>(NOP, 0, instrs.size()));
  }
  for (const auto &i : edges) {
    instrs[i.first]->set_extra_jump(instrs[i.second].get());
  }
  return instrs;
}

std::vector<std::unique_ptr<Instr>> CodeGenerator::CopyAndReplaceInstr(
  const std::vector<std::unique_ptr<Instr>> &list, size_t start_bci, size_t end_bci,
  const std::vector<std::unique_ptr<Instr>> &replacement) {
  std::vector<std::pair<size_t, size_t>> edges;
  std::vector<std::unique_ptr<Instr>> instrs;
  int bciDelta = replacement.size() - (end_bci - start_bci);
  MS_LOG(INFO) << "CopyAndReplaceInstr start replacement size:" << replacement.size() << ", start_bci:" << start_bci
               << ", end_bci:" << end_bci;
  size_t index = 0;
  // instructions before start_bci
  for (size_t bci = static_cast<size_t>(list.front()->bci()); bci < start_bci; ++bci) {
    const auto &i = list[bci];
    (void)instrs.emplace_back(std::make_unique<Instr>(*i));
    if (i->extra_jump()) {
      size_t tar = i->extra_jump()->bci();
      if (tar > end_bci) {
        edges.push_back({index, tar + bciDelta});
      }
    }
    index++;
  }
  // Insert replacement instructions
  // The current implementation does not consider jumps to the tail because there are no tail jumps in the context of
  // loop encapsulation.
  for (size_t bci = 0; bci < replacement.size(); ++bci) {
    const auto &i = replacement[bci];
    (void)instrs.emplace_back(std::make_unique<Instr>(*i));
    MS_EXCEPTION_IF_CHECK_FAIL(i->extra_jump() == nullptr, "should not exist jump here");
    index++;
  }
  // instructions after end_bci
  for (size_t bci = end_bci; bci < list.size(); ++bci) {
    const auto &i = list[bci];
    (void)instrs.emplace_back(std::make_unique<Instr>(*i));
    if (i->extra_jump()) {
      size_t tar = i->extra_jump()->bci();
      if (tar > end_bci) {
        edges.push_back({index, tar + bciDelta});
      }
    }
    index++;
  }
  for (const auto &i : edges) {
    instrs[i.first]->set_extra_jump(instrs[i.second].get());
  }
  return instrs;
}

void CodeGenerator::ResetExceptionCodeItem(InstructionList::const_iterator erased) {
  if (code_.co_exceptiontable.empty()) {
    return;
  }
  // python3.11+
  const InstructionList &list = code_.co_code;
  for (auto e_iter = code_.co_exceptiontable.begin(); e_iter != code_.co_exceptiontable.end(); ++e_iter) {
    auto &i = *e_iter;
    bool erase_item = (erased->get() == i.begin_ && erased == list.end() - 1) ||  // item begin is last instruction
                      (erased->get() == i.end_ && erased == list.begin());        // item end is first instruction
    if (erase_item) {
      // maybe shouldn't reach here ?
      MS_LOG(INFO) << "erase exception item";
      e_iter = code_.co_exceptiontable.erase(e_iter);
      continue;
    }
    if (erased->get() == i.begin_) {
      i.begin_ = (erased + 1)->get();
    }
    if (erased->get() == i.end_) {
      i.end_ = (erased - 1)->get();
    }
    if (erased->get() == i.jump_) {
      MS_EXCEPTION_IF_CHECK_FAIL(erased != list.begin(), "exception item can't jump to first instruction");
      i.jump_ = (erased + (erased == list.end() - 1 ? -1 : 1))->get();
    }
  }
}

void CodeGenerator::EraseUnusedInstr() {
  InstructionList *list = &code_.co_code;
  auto NeedRemove = [](const std::vector<std::unique_ptr<Instr>>::iterator &i) {
    int op = (*i)->op();
    if (Opcode(op).GetClass() == Opcode::kNop) {
      return true;
    }
    if (op == JUMP_ABSOLUTE || op == JUMP_FORWARD) {  // jump to next
      return (*i)->extra_jump() == (i + 1)->get();
    }
    return false;
  };
  // mark unused instruction
  auto erase_iter = list->begin();
  int bci = 0;
  for (auto i = list->begin(); i != list->end(); ++i) {
    if (NeedRemove(i)) {
      (*i)->set_bci(-1);
      (*i)->set_extra_jump((i + 1)->get());
      ResetExceptionCodeItem(i);
    } else {
      (*i)->set_bci(bci);
      std::swap(*erase_iter, *i);
      ++erase_iter;
      ++bci;
    }
  }
  if (erase_iter == list->end()) {
    return;
  }
  // reset jump
  for (auto i = list->begin(); i != erase_iter; ++i) {
    Instr *tar = (*i)->extra_jump();
    if (tar == nullptr) {
      continue;
    }
    while (tar->bci() == -1) {
      MS_EXCEPTION_IF_NULL(tar->extra_jump());
      tar = tar->extra_jump();
    }
    (*i)->set_extra_jump(tar);
  }
  list->erase(erase_iter, list->end());
}

std::vector<std::unique_ptr<Instr>> CodeGenerator::RotStack(int stack) {
  std::vector<std::unique_ptr<Instr>> res;
  if (stack == 0) {
    return res;
#if PY_MAJOR_VERSION == 3 && PY_MINOR_VERSION == 11
  } else {
    res.push_back(std::make_unique<Instr>(SWAP, stack + 1));
#elif PY_MAJOR_VERSION == 3 && PY_MINOR_VERSION == 10
  } else {
    res.push_back(std::make_unique<Instr>(ROT_N, stack + 1));
#elif PY_MAJOR_VERSION == 3 && PY_MINOR_VERSION < 10 && PY_MINOR_VERSION >= 7
  } else if (stack == 1) {
    res.push_back(std::make_unique<Instr>(ROT_TWO));
  } else if (stack == 2) {
    res.push_back(std::make_unique<Instr>(ROT_THREE));
#if (PY_MAJOR_VERSION == 3) && (PY_MINOR_VERSION > 7)
  } else if (stack == 3) {
    res.push_back(std::make_unique<Instr>(ROT_FOUR));
#endif
  } else {
    MS_LOG(DEBUG) << "Too many stack nodes, use BUILD_TUPLE and UNPACK_SEQUENCE";
    // Assuming the elements on the stack are 2,3,4,5,1, where the left side is bottom and the right side is top of the
    // stack, the goal is to transform it into 1,2,3,4,5.
    res.insert(res.end(), std::make_unique<Instr>(BUILD_TUPLE, stack + 1));      // 2,3,4,5,1 -> (2,3,4,5,1)
    res.insert(res.end(), std::make_unique<Instr>(UNPACK_SEQUENCE, stack + 1));  // -> 1,5,4,3,2
    res.insert(res.end(), std::make_unique<Instr>(BUILD_TUPLE, stack));          // -> 1,(5,4,3,2)
    res.insert(res.end(), std::make_unique<Instr>(UNPACK_SEQUENCE, stack));      // -> 1,2,3,4,5
#endif
  }

  return res;
}

/**
 * traverse all values in reverse order, set alive time for each input value
 * the inputs of all values is before these values
 */
void CodeGenerator::MarkAlive() {
  for (auto i : nodes_->outputs) {
    MarkAlive(i);
  }
  for (int index = nodes_->operations.size() - 1; index >= 0; --index) {
    ValueNode *node = nodes_->operations[index];
    for (auto input : node->inputs()) {
      MarkAlive(input, index);
    }
  }
}

void CodeGenerator::MarkAlive(ValueNode *node, int order) {
  int *cur = &nodes_alive_[node];
  *cur = std::max(*cur, order);
  auto iter = locals_map_.find(node);
  if (iter == locals_map_.end()) {
    return;
  }
  for (auto find_it = locals_map_.begin(); find_it != locals_map_.end(); ++find_it) {
    if (iter->second == find_it->second) {
      nodes_alive_[find_it->first] = *cur;
    }
  }
}

void CodeGenerator::MakeSameLocal(ValueNode *node, ValueNode *other_node, bool clear) {
  auto iter = locals_map_.find(node);
  if (iter != locals_map_.end()) {
    locals_map_[other_node] = iter->second;
    clear ? (void)locals_map_.erase(iter) : ((void)0);
    return;
  }
  LoadValue(node);
  NewInstr(STORE_FAST, AllocLocal(other_node, 0));
}

int CodeGenerator::AllocLocal(ValueNode *node, int index) {
  auto iter = locals_map_.find(node);
  if (iter != locals_map_.end()) {
    return iter->second;
  }
  int res;
  std::set<int> used_slots;  // order set
  for (iter = locals_map_.begin(); iter != locals_map_.end(); ++iter) {
    if (index != INT_MAX && nodes_alive_[iter->first] <= index) {
      res = iter->second;
      locals_map_.erase(iter);
      locals_map_.insert({node, res});
      return res;
    }
    used_slots.insert(iter->second);
  }
  res = 0;
  for (auto i = used_slots.begin(); i != used_slots.end() && res == (*i); ++i, ++res) {
  }
  locals_map_.insert({node, res});
  SetLocalsCount(std::max(locals_map_.size(), static_cast<size_t>(res)));
  return res;
}

void CodeGenerator::NewInstr(int op, int arg, int line) { AddInstr(std::make_unique<Instr>(op, arg, -1, line)); }
void CodeGenerator::AddInstr(std::unique_ptr<Instr> &&instr) { code_.co_code.emplace_back(std::move(instr)); }

void CodeGenerator::AddInstrs(std::vector<std::unique_ptr<Instr>> &&l) {
  code_.co_code.insert(code_.co_code.end(), std::make_move_iterator(l.begin()), std::make_move_iterator(l.end()));
}

void CodeGenerator::AddInstructionWithExceptionTable(const CFG *cfg, int start, int end) {
  AddInstrs(CopyInstr(cfg->instr_pool(), start, end));
  CollectExceptionTableItem(cfg, start, end);
}

void CodeGenerator::CollectExceptionTableItem(const CFG *cfg, int start, int end) {
  const auto &table = cfg->exc_table();
  if (table.empty()) {
    return;
  }
  auto iter = cfg->FindTryWithBlock(start);
  if (iter == table.end()) {
    iter = table.lower_bound(start);
  }
  /**
   * cfg->instr_pool():   |...|copied instructions|...|
   * this->code_.co_code: |...|copied instructions|
   */
  int origin_code_size = cfg->instr_pool().size();
  int off_end = origin_code_size - end;
  int code_size = code_.co_code.size();
  MS_LOG(DEBUG) << "[" << start << ", " << end << ") code_size: " << code_size
                << ", origin_code_size: " << origin_code_size;
  MS_EXCEPTION_IF_CHECK_FAIL(static_cast<size_t>(end - start) <= code_.co_code.size(), "too less instructions");
  for (; iter != table.end() && iter->second.end_ < end; ++iter) {
    int new_handler = code_size - (origin_code_size - iter->second.jump_ - off_end);
    int new_start = code_size - (origin_code_size - iter->second.begin_ - off_end);
    int new_end = code_size - (origin_code_size - iter->second.end_ - off_end);
    new_end--;
    MS_EXCEPTION_IF_CHECK_FAIL(new_handler < code_size && new_start < code_size && new_end < code_size,
                               "find an exception item out of copy instructions range");
    code_.co_exceptiontable.push_back({code_.co_code[new_start].get(), code_.co_code[new_end].get(),
                                       code_.co_code[new_handler].get(), iter->second.stack_, iter->second.lasti_});
  }
}

void CodeGenerator::AddCallInstr(size_t load_args_offset, int oparg) {
  /**
   * if no self
   * python3.11 ~ python3.12, call stack is [NULL, callable, args...]
   * python3.13 call stack is [callable, NULL, args...]
   */
#if IS_PYTHON_3_11_PLUS
  auto iter = code_.co_code.begin() + load_args_offset + IS_PYTHON_3_13_PLUS;
  const auto &load_instr = *iter;
  if (load_instr->op() == LOAD_GLOBAL) {
    load_instr->set_arg(1);
  } else {
    code_.co_code.insert(iter, std::make_unique<Instr>(PUSH_NULL));
  }
#endif
  AddInstr(MakeCallFunInstrPtr(oparg));
}

void CodeGenerator::LoadValue(ValueNode *node) {
  MS_EXCEPTION_IF_NULL(node);
  MS_LOG(DEBUG) << "node: " << node->ToString();
  auto iter = locals_map_.find(node);
  if (iter != locals_map_.end()) {
    NewInstr(LOAD_FAST, iter->second);
    return;
  }
  int opcode = node->GetOpcode();
  if (opcode == LOAD_DEREF) {
    AddInstr(std::make_unique<Instr>(opcode, node->GetOparg(), node->GetName()));
    return;
  }
  std::string key = node->GetName();
  if (opcode == LOAD_GLOBAL) {
    PyObject *globals = node->GetGraph() ? node->GetGraph()->GetGlobals().ptr() : nullptr;
    MS_EXCEPTION_IF_NULL(globals);
    if (globals != GetGlobals().ptr()) {
      py::str key_object(key);
      PyObject *value = PyObject_GetItem(globals, key_object.ptr());
      if (value != nullptr) {
        py::object handle_value = py::reinterpret_steal<py::object>(value);
        MapAdd(GetGlobals(), key, handle_value, &key);
      } else {
        // name error, global undefined
        PyErr_Clear();
      }
    }
    NewInstr(LOAD_GLOBAL);
    code_.co_code.back()->set_name(key);
    return;
  }

  if (opcode == LOAD_CONST) {
    LoadConst(node->GetVobj()->GetPyObject());
    return;
  }
  if (missing_value_to_undefine_) {
    std::stringstream name;
    const size_t limit = 40;
    auto abs = node->abstract_wrapper() == nullptr ? nullptr : node->abstract_wrapper()->abstract();
    auto str = abs == nullptr ? "<NULL>" : abs->ToString();
    str = str.size() < limit ? str : str.substr(0, limit) + "...";
    std::replace(str.begin(), str.end(), '\n', ' ');
    name << "<missing value " << (node->GetOpcode() <= 0 ? "" : Opcode(node->GetOpcode()).name()) << " -> " << str;
    NewInstr(LOAD_NAME);
    code_.co_code.back()->set_name(name.str());
    return;
  }
  MS_LOG(INTERNAL_EXCEPTION) << "missing value, [" << node->ToString() << "]";
}

void CodeGenerator::LoadConst(const py::object &cnst) {
  MS_EXCEPTION_IF_NULL(cnst.ptr());
  if (CheckConstPyObject(cnst.ptr())) {
    NewInstr(LOAD_CONST);
    code_.co_code.back()->set_cnst(cnst);
    return;
  }
  py::object name = py::getattr(cnst, "__qualname__", nullptr);
  if (name.ptr() == nullptr) {
    name = py::getattr(cnst, "__name__", nullptr);
  }
  std::string key = name.ptr() && PyUnicode_Check(name.ptr()) ? name.cast<std::string>() : GenerateObjectKey(cnst);
  MapAdd(GetGlobals(), key, cnst, &key);
  NewInstr(LOAD_GLOBAL);
  code_.co_code.back()->set_name(key);
}

void CodeGenerator::BuildOper(ValueNode *node, int index) {
  MS_EXCEPTION_IF_NULL(node);
  MS_LOG(DEBUG) << "index: " << index << " node: " << node->ToString();
  static const std::set<int> not_value_oper = {
    STORE_DEREF,  DELETE_DEREF,  STORE_GLOBAL, DELETE_GLOBAL, STORE_ATTR, DELETE_ATTR,
    STORE_SUBSCR, DELETE_SUBSCR, IMPORT_STAR,  RAISE_VARARGS, RERAISE,
  };
  static const std::unordered_map<int, int> const_arg_oper = {
    {LIST_APPEND, 1}, {LIST_EXTEND, 1}, {DICT_MERGE, 1}, {DICT_UPDATE, 1}, {SET_UPDATE, 1}, {SET_ADD, 1}, {MAP_ADD, 2},
  };

  if (IsNonLocalValue(node)) {
    return;
  }
  if ((vm_mode_ && !node->IsVmNode()) || (!vm_mode_ && !node->IsGraphNode())) {
    MS_LOG(INFO) << "Codegen mode and node mode mismatch. codegen mode:" << (vm_mode_ ? "vm" : "graph")
                 << ", node: " << ToString(node);
  }

  int load_args_offset = code_.co_code.size();
  for (auto param : node->inputs()) {
    LoadValue(param);
  }
#if IS_PYTHON_3_11_PLUS
  if (node->GetType() == ValueNode::Call && static_cast<CallNode *>(node)->kw_names() != nullptr) {
    AddInstr(std::make_unique<Instr>(KW_NAMES, 0, static_cast<CallNode *>(node)->kw_names()));
  }
#endif

  Opcode op(node->GetOpcode());
  int arg = pijit::Opcode(op).HasArg() ? node->GetOparg() : 0;
  auto const_arg_oper_iter = const_arg_oper.find(op);
  if (const_arg_oper_iter != const_arg_oper.end()) {
    arg = const_arg_oper_iter->second;
  }
  if (op.IsCall()) {
    AddCallInstr(load_args_offset, arg);
    code_.co_code.back()->set_op(op);
  } else {
    NewInstr(op, arg, node->GetLineNo());
  }
  code_.co_code.back()->set_line(node->GetLineNo());
  code_.co_code.back()->set_name(node->GetName());

  if (not_value_oper.find(op) != not_value_oper.end()) {
    return;
  }
  if (nodes_alive_[node] == 0) {
    NewInstr(POP_TOP);
  } else {
    NewInstr(STORE_FAST, AllocLocal(node, index), node->GetLineNo());
  }
}

void CodeGenerator::Init() {
  const int size = SizeToInt(nodes_->inputs.size());
  code_.co_nlocals = size;
  for (int i = 0; i < size; ++i) {
    ValueNode *param = nodes_->inputs[i];
    locals_map_[param] = i;

    MS_LOG(DEBUG) << "code gen init status: local " << i << " = [" << param->ToString();
  }
}

void CodeGenerator::Build() {
  // build operations
  MarkAlive();
  for (size_t index = 0; index < nodes_->operations.size(); ++index) {
    BuildOper(nodes_->operations[index], index);
  }
  SetLocalsCount(locals_map_.size());
}

void CodeGenerator::GenReturn() {
  for (const auto &i : nodes_->outputs) {
    LoadValue(i);
  }
  if (nodes_->outputs.size() > 1) {
    NewInstr(BUILD_TUPLE, nodes_->outputs.size());
  }
  if (nodes_->outputs.size() == 0) {
    NewInstr(LOAD_CONST, 0);
    code_.co_code.back()->set_cnst(py::none());
  }
  NewInstr(RETURN_VALUE);
  SetLocalsCount(locals_map_.size());
}

namespace {
bool IsNotNeedTrack(const std::vector<std::unique_ptr<Instr>> &list, int start = -1) {
  if (list.empty() || start == -1) {
    return true;
  }
  auto iter = std::find_if(list.begin() + start, list.end(), [](const std::unique_ptr<Instr> &i) {
    return Opcode(i->op()).IsCall() || Opcode(i->op()).IsBinaryMath();
  });
  return iter == list.end();
}

std::vector<std::unique_ptr<Instr>> MakeFunc(const py::object &code, const std::string &name,
                                             const std::vector<std::string> &closures) {
  std::vector<std::unique_ptr<Instr>> instrs;
  for (size_t i = 0; i < closures.size(); ++i) {
    instrs.emplace_back(std::make_unique<Instr>(LOAD_CLOSURE, i, closures[i]));
  }
  unsigned make_oparg = 0;
  if (closures.size() != 0) {
    make_oparg |= 0x08;
    instrs.emplace_back(std::make_unique<Instr>(BUILD_TUPLE, closures.size()));
  }
  instrs.emplace_back(std::make_unique<Instr>(LOAD_CONST, 0, code));
#if !IS_PYTHON_3_11_PLUS
  instrs.emplace_back(std::make_unique<Instr>(LOAD_CONST, 0, py::str(name)));
#endif
  instrs.emplace_back(std::make_unique<Instr>(MAKE_FUNCTION, make_oparg));
  return instrs;
}

// return co_cellvars and co_freevars
std::vector<std::string> GetClosureNames(PyCodeObject *code) {
  PyCodeWrapper co(code);
  py::tuple cells = co.CellVars();
  py::tuple frees = co.FreeVars();

  std::vector<std::string> names;
  for (Py_ssize_t i = 0; i < PyTuple_GET_SIZE(cells.ptr()); ++i) {
    names.push_back(PyUnicode_AsUTF8(PyTuple_GET_ITEM(cells.ptr(), i)));
  }
  for (Py_ssize_t i = 0; i < PyTuple_GET_SIZE(frees.ptr()); ++i) {
    names.push_back(PyUnicode_AsUTF8(PyTuple_GET_ITEM(frees.ptr(), i)));
  }
  return names;
}
}  // namespace

py::object CodeBreakGenerator::MakeCapturedCode(std::vector<std::unique_ptr<Instr>> &&load_oper,  // prepare parameters
                                                int argc, unsigned code_flag) const {
  CodeGenerator code_gen(&captured_);
  code_gen.set_missing_value_to_undefine(true);
  code_gen.SetGlobals(globals_);
  code_gen.Init();
  code_gen.AddInstrs(std::move(load_oper));
  code_gen.Build();
  code_gen.GenReturn();

  unsigned flags = co_->co_flags & ~(CO_VARARGS | CO_VARKEYWORDS);
  code_gen.SetArgsInfo(argc, 0);
  code_gen.SetCodeFlags(flags | code_flag);
  code_gen.SetFirstLineNumber(captured_.operations[0]->GetLineNo());
  code_gen.SetFreeVariableNames(GetClosureNames(co_));
  code_gen.SetCodeName(MakeCompiledName(py::str(co_->co_name)));
  code_gen.SetFileName(py::cast<py::object>(co_->co_filename));
  code_gen.SetQualName("<pijit.compile>");

  py::object code = code_gen.NewCode();
  auto parent = GetJitCompileResults(co_);
  MS_EXCEPTION_IF_NULL(parent);
  JitCompileResults *child = CreateJitCompileResults(code.ptr());
  MS_EXCEPTION_IF_NULL(child);
  child->set_stat(JitCompileResults::GRAPH_CAPTURED);
  child->set_conf(parent->conf());
  child->set_tbs(parent->tbs());

  auto jcr = GetJitCompileResults(co_);
  if (jcr->conf()->GetBoolConfig(GraphJitConfig::kInterpretCapturedCode)) {
    return code;
  }
  auto name = PyUnicode_AsUTF8(reinterpret_cast<PyCodeObject *>(code.ptr())->co_name);
  Compile(name, argc, 0, code_flag, code);
  return code;
}

void CodeBreakGenerator::CallCapturedCode(CodeGenerator *code_gen) {
  if (captured_.operations.empty()) {
    MS_LOG(DEBUG) << "No captured operations, no need to generate code";
    return;
  }
  MS_LOG(DEBUG) << "Do codegen for calling graph";
  GraphParameterBuilder param_info;
  BuildGraphParameters(code_gen->GetLocalsMap(), &param_info);
  py::object code = MakeCapturedCode(std::move(param_info.sort_), param_info.args_.size(), 0);

  PyCodeWrapper co(co_);

  int load_args_offset = code_gen->GetCode().co_code.size();
  code_gen->AddInstrs(MakeFunc(code, "<pijit.compile>", GetClosureNames(co_)));
  code_gen->AddInstrs(std::move(param_info.load_));
  code_gen->AddCallInstr(load_args_offset, param_info.args_.size());
  extra_local_ = code_gen->AllocLocal(nullptr);
  code_gen->NewInstr(STORE_FAST, extra_local_);
  code_gen->AddInstrs(std::move(param_info.dele_));
}

bool CodeBreakGenerator::IsCopyCapturedInstructions() const {
  return !IS_PYTHON_3_11_PLUS && no_graph_ && !NeedHandleBreakAtCall();
}

void CodeBreakGenerator::FixInterpretOuput(CodeGenerator *code_gen) {
  if (!captured_.outputs.empty()) {
    MS_LOG(DEBUG) << "Do codegen for graph outputs";
    MS_EXCEPTION_IF_CHECK_FAIL(extra_local_ != -1, "can't find graph output");
    if (captured_.outputs.size() > 1) {
      // fill interpret local map
      code_gen->NewInstr(LOAD_FAST, extra_local_);
      code_gen->NewInstr(UNPACK_SEQUENCE, captured_.outputs.size());
      for (auto i : captured_.outputs) {
        code_gen->MarkAlive(i);
        code_gen->NewInstr(STORE_FAST, code_gen->AllocLocal(i, 0));
      }
    } else {
      code_gen->MarkAlive(captured_.outputs[0]);
      // The placeholder of captured_.outputs[0] is nullptr, invalid node, should be replaced.
      auto iter = code_gen->GetLocalsMap().find(nullptr);
      MS_EXCEPTION_IF_CHECK_FAIL(iter != code_gen->GetLocalsMap().end(), "Can't find the placeholder of output.");
      code_gen->GetLocalsMap()[captured_.outputs[0]] = iter->second;
      code_gen->GetLocalsMap().erase(iter);
    }
  }
  // reconstruct interpret values if need
  HandleOutputOpt(code_gen);
}

void CodeBreakGenerator::HandleOutputOpt(CodeGenerator *cg) {
  if (replaced_nodes_.empty() && outputs_optimize_.operations.empty()) {
    MS_LOG(DEBUG) << "No outputs_optimize nodes, no need to do codegen";
    return;
  }
  cg->ClearAlive();
  auto handle_replaced = [this, &cg](bool is_pre) {
    for (const auto &node : interpret_.outputs) {
      cg->MarkAlive(node);
      auto iter = replaced_nodes_.find(node);
      if (iter == replaced_nodes_.end()) {
        continue;
      }
      if (is_pre) {
        cg->MarkAlive(iter->second);
        continue;
      }
      if (cg->GetLocalsMap().find(node) != cg->GetLocalsMap().end()) {
        continue;
      }
      bool not_a_local = cg->GetLocalsMap().find(iter->second) == cg->GetLocalsMap().end();
      if (not_a_local && iter->second->GetOpcode() != LOAD_CONST) {
        MS_LOG(INTERNAL_EXCEPTION) << iter->second->ToString() << " should be a local var.";
      }
      cg->MakeSameLocal(iter->second, node);
    }
  };
  handle_replaced(true);
  std::swap(interpret_.operations, outputs_optimize_.operations);
  MS_LOG(DEBUG) << "Do codegen for outputs_optimize";
  cg->Build();
  std::swap(interpret_.operations, outputs_optimize_.operations);
  handle_replaced(false);
}

void CodeBreakGenerator::RestoreStack(CodeGenerator *code_gen) const {
  auto begin = interpret_.outputs.begin();
  auto end = interpret_.outputs.end() - alive_locals_.size();

#if IS_PYTHON_3_11_PLUS
  const auto &break_instr = GetCFG()->instr_pool()[break_bci_];
  int stack_count = end - begin;
  if (Opcode(break_instr->op()).IsCall()) {
    int index = break_instr->op() != CALL_FUNCTION_EX ? break_instr->arg() : 1 + ((break_instr->arg() & 1) ? 1 : 0);
    MS_EXCEPTION_IF_CHECK_FAIL(index < stack_count, "error stack status, can't find callable object");
    auto func_iter = begin + (stack_count - index - (!IS_PYTHON_3_13_PLUS));
    for (auto iter = begin; iter != end; ++iter) {
      bool skip_push_null_if_break_comprehension_if_cp311 = false;
#if !IS_PYTHON_3_12_PLUS
      // python3.11 only, use iterable object as self. Although the oparg is 0, actual number of args is 1
      skip_push_null_if_break_comprehension_if_cp311 =
        (*func_iter)->GetOpcode() == GET_ITER && func_iter != begin && (*(func_iter - 1))->GetOpcode() == MAKE_FUNCTION;
#endif
      if (iter == func_iter && !skip_push_null_if_break_comprehension_if_cp311) {
        code_gen->NewInstr(PUSH_NULL);
      }
      if (*iter == &ValueNode::kStackNull) {
        continue;
      }
      code_gen->LoadValue(*iter);
    }
    if (break_instr->cnst().ptr() != nullptr) {
      code_gen->AddInstr(std::make_unique<Instr>(KW_NAMES, 0, break_instr->cnst()));
    }
    return;
  }
#endif

  std::for_each(begin, end, [code_gen](ValueNode *i) {
    if (i == &ValueNode::kStackNull) {
      return;
    }
    code_gen->LoadValue(i);
  });
}

void CodeBreakGenerator::RestoreLocals(CodeGenerator *code_gen, bool only_load) const {
  auto begin = interpret_.outputs.end() - alive_locals_.size();
  auto end = interpret_.outputs.end();
  if (only_load) {
    std::for_each(begin, end, [code_gen](ValueNode *i) { code_gen->LoadValue(i); });
    return;
  }
  std::vector<std::unique_ptr<Instr>> st;
  auto index_iter = alive_locals_.begin();
  for (auto node_iter = begin; node_iter != end; ++node_iter, ++index_iter) {
    auto target = code_gen->GetLocalsMap().find(*node_iter);
    if (target != code_gen->GetLocalsMap().end() && target->second == *index_iter) {
      continue;
    }
    MS_EXCEPTION_IF_CHECK_FAIL(index_iter != alive_locals_.end(), "error alive local");
    code_gen->LoadValue(*node_iter);
    st.push_back(std::make_unique<Instr>(STORE_FAST, *index_iter));
  }
  std::reverse(st.begin(), st.end());
  code_gen->AddInstrs(std::move(st));
}

namespace {
std::vector<std::unique_ptr<Instr>> MakeUntrackedCodeHelper(const std::vector<std::unique_ptr<Instr>> &instr_pool,
                                                            int start, int stack_count,
                                                            const std::vector<int> &alive_locals,
                                                            const std::vector<ValueNode *> &stack = {},
                                                            int *argc = nullptr, bool is_call_stack_effect = false) {
  /**
   * arguments layout
   * stack value is sorted from bottom to top, locals is sorted by local index
   * | -- stack of func -- | -- locals of func -- |
   */

  // restore stack and locals
  MS_EXCEPTION_IF_CHECK_FAIL(stack_count >= 0, "stack_count should >= 0, but is " + std::to_string(stack_count));

  std::vector<std::unique_ptr<Instr>> load;
  std::vector<std::unique_ptr<Instr>> store;
  int arg_i = 0;

#if IS_PYTHON_3_11_PLUS
  for (int i = 0; i < stack_count; ++i) {
    if (static_cast<size_t>(i) < stack.size() && stack[i] == &ValueNode::kStackNull) {
      load.push_back(std::make_unique<Instr>(PUSH_NULL));
    } else {
      load.push_back(std::make_unique<Instr>(LOAD_FAST, arg_i++));
    }
  }
  if (is_call_stack_effect && load.back()->op() == PUSH_NULL) {
    load.back()->set_op(LOAD_FAST);
    load.back()->set_arg(arg_i++);
  }
#else
  for (int i = 0; i < stack_count; ++i) {
    (void)load.emplace_back(std::make_unique<Instr>(LOAD_FAST, arg_i++));
  }
#endif

  for (auto iter = alive_locals.begin(); iter != alive_locals.end(); ++iter) {
    (void)load.emplace_back(std::make_unique<Instr>(LOAD_FAST, arg_i++));
    (void)store.emplace_back(std::make_unique<Instr>(STORE_FAST, *iter));
  }
  if (argc != nullptr) {
    *argc = arg_i;
  }

  std::vector<std::unique_ptr<Instr>> list = std::move(load);
  std::move(store.rbegin(), store.rend(), std::back_inserter(list));

  // copy untracked bytes
  std::vector<std::unique_ptr<Instr>> untracked = CodeGenerator::CopyInstr(instr_pool, start);
  int first_line = untracked[0]->line();
  std::for_each(list.begin(), list.end(), [first_line](const auto &instr) { instr->set_line(first_line); });
  std::move(untracked.begin(), untracked.end(), std::back_inserter(list));
  return list;
}
}  // namespace

py::object CodeBreakGenerator::MakeUntrackedCode(int untracked_bci, int untracked_stack_effect, int *argc) const {
  const int stack_count = interpret_.outputs.size() - alive_locals_.size();
  const int start_stack = stack_count + untracked_stack_effect;
  bool is_call_stack_effect = Opcode(GetCFG()->instr_pool()[break_bci_]->op()).IsCall();
  MS_LOG(DEBUG) << "start stack: " << start_stack << " stack_effect " << untracked_stack_effect
                << " is_call_stack_effect " << is_call_stack_effect;
  auto list = MakeUntrackedCodeHelper(GetCFG()->instr_pool(), untracked_bci, start_stack, alive_locals_,
                                      interpret_.outputs, argc, is_call_stack_effect);
  int first_line = list[0]->line();
  int nlocals = GetCFG()->GetLocalCount();

  CodeGenerator::Code ccode = {
    *argc,
    0,
    std::max(*argc, nlocals),
    (signed)co_->co_flags & ~(CO_VARARGS | CO_VARKEYWORDS),
    first_line,
    std::move(list),
    py::cast<std::vector<std::string>>(PyCodeWrapper(co_).VarNames()),
    std::vector<std::string>(),
    GetClosureNames(co_),
    MakeBrkName(PyUnicode_AsUTF8(co_->co_name), untracked_bci),
    py::reinterpret_borrow<py::object>(co_->co_filename),
    "<pijit.resume>",
  };
  CodeGenerator cg(std::move(ccode));
  cg.CollectExceptionTableItem(GetCFG(), untracked_bci, GetCFG()->instr_pool().size());
  py::object code = cg.NewCode();
  auto parent = GetJitCompileResults(co_);
  JitCompileResults *child = CreateJitCompileResults(code.ptr());
  MS_EXCEPTION_IF_NULL(child);
  child->set_stat(JitCompileResults::GRAPH_CANDIDATE);
  MS_EXCEPTION_IF_NULL(parent);
  child->set_conf(parent->conf());
  child->set_tbs(parent->tbs());
  return code;
}

void CodeBreakGenerator::ReconstructStack(CodeGenerator *code_gen, int untracked_bci,
                                          int untracked_stack_effect) const {
  const auto &instr = GetCFG()->instr_pool()[break_bci_];
  RestoreStack(code_gen);
  if (break_bci_ == untracked_bci) {
    return;
  }
  if (!Opcode(instr->op()).IsCall()) {
    code_gen->AddInstrs(CodeGenerator::CopyInstr(cfg_->instr_pool(), break_bci_, untracked_bci));
    return;
  }
  code_gen->NewInstr(instr->op(), instr->arg(), instr->line());
  code_gen->GetCode().co_code.back()->set_name(instr->name());
  code_gen->GetCode().co_code.back()->set_cnst(instr->cnst());
}

void CodeBreakGenerator::BreakAtIf(CodeGenerator *code_gen) const {
  const auto &list = GetCFG()->instr_pool();
  int op = list[break_bci_]->op();
  int stack_effect = -1;
  int stack_count = SizeToInt(interpret_.outputs.size() - alive_locals_.size());
  PyCodeWrapper co(co_);
  auto closures = GetClosureNames(co_);
  py::object code;

  MS_EXCEPTION_IF_CHECK_FAIL(stack_count >= 1, "error stack");

  RestoreStack(code_gen);
  code_gen->NewInstr(op);
  Instr *if_instr = code_gen->GetCode().co_code.back().get();

  // fall-branch
  int argc;
  code = MakeUntrackedCode(break_bci_ + 1, stack_effect, &argc);
  int load_args_offset = code_gen->GetCode().co_code.size();
  code_gen->AddInstrs(MakeFunc(code, "<pijit.resume>", closures));
  code_gen->AddInstrs(CodeGenerator::RotStack(stack_count + stack_effect));
  RestoreLocals(code_gen, true);
  code_gen->AddCallInstr(load_args_offset, argc);
  code_gen->NewInstr(RETURN_VALUE);

  // jump-branch
  stack_effect = (op == JUMP_IF_TRUE_OR_POP || op == JUMP_IF_FALSE_OR_POP) ? 0 : -1;
  code = MakeUntrackedCode(list[break_bci_]->extra_jump()->bci(), stack_effect, &argc);
  load_args_offset = code_gen->GetCode().co_code.size();
  code_gen->AddInstrs(MakeFunc(code, "<pijit.resume>", closures));
  code_gen->AddInstrs(CodeGenerator::RotStack(stack_count + stack_effect));
  RestoreLocals(code_gen, true);
  code_gen->AddCallInstr(load_args_offset, argc);
  code_gen->NewInstr(RETURN_VALUE);
  if_instr->set_extra_jump(code_gen->GetCode().co_code[load_args_offset].get());
}

void CodeBreakGenerator::BreakAtBlock(CodeGenerator *code_gen, int untracked_bci, int untracked_stack_effect) {
  RestoreStack(code_gen);
  RestoreLocals(code_gen, false);
  const auto &instr_list = GetCFG()->instr_pool();
  code_gen->AddInstructionWithExceptionTable(GetCFG(), break_bci_, untracked_bci);

  BitMap alive = GetCFG()->liveness()->CollectAlive(untracked_bci);
  BitMap defined(alive.size());
  for (int i = break_bci_; i < untracked_bci; ++i) {
    if (instr_list[i]->op() == STORE_FAST) {
      defined.Set(instr_list[i]->arg());
    }
  }
  std::for_each(alive_locals_.begin(), alive_locals_.end(), [&defined](int i) { defined.Set(i); });
  alive.And(defined);

  alive_locals_.clear();
  for (BitMap::Iter iter(&alive, true), end(&alive, false); iter != end; ++iter) {
    alive_locals_.push_back(*iter);
  }

  interpret_.outputs.resize(alive_locals_.size(), &ValueNode::kUnboundLocal);
  untracked_stack_effect = 0;

  int argc;
  py::object code = MakeUntrackedCode(untracked_bci, untracked_stack_effect, &argc);
  PyCodeWrapper co(co_);

  int load_args_offset = code_gen->GetCode().co_code.size();
  code_gen->AddInstrs(MakeFunc(code, "<pijit.resume>", GetClosureNames(co_)));
  for (auto i : alive_locals_) {
    code_gen->NewInstr(LOAD_FAST, i);
  }
  code_gen->AddCallInstr(load_args_offset, argc);
  code_gen->NewInstr(RETURN_VALUE);
}

namespace {
py::object PackNestedFuncCodes(const std::vector<Graph *> &call_stack, int top_argc, int bottom_untracked_bci,
                               int bottom_stack_effect) {
  MS_ASSERT(!call_stack.empty());
  py::object code;
  int argc = 0;
  int offset = top_argc;
  std::vector<std::unique_ptr<Instr>> oper;
  // Iterate from bottom-graph to top-graph.
  for (auto iter = call_stack.rbegin(); iter != call_stack.rend(); ++iter) {
    bool is_bottom = iter == call_stack.rbegin();
    bool is_top = (iter + 1) == call_stack.rend();
    Graph *g = *iter;
    PyCodeWrapper co(g->GetCodeObj());
    const Graph::BreakInfo &info = g->break_info();
    const std::vector<int> &alive_locals = info.alive_locals_;
    const Instr *break_point = info.break_point_;
    const ValueNode *break_point_node = info.break_point_node_;
    int untracked_bci = is_bottom ? bottom_untracked_bci : info.bci_ + 1;
    int alive_size = SizeToInt(info.alive_nodes_.size());
    int alive_local_size = SizeToInt(alive_locals.size());
    int stack_effect =
      is_bottom ? bottom_stack_effect : GetOpcodeMaxStackEffect(break_point->op(), break_point->arg(), false);
    argc += alive_size;
    offset -= alive_size;
    int activate_argc = alive_size + stack_effect;
    int stack_count = activate_argc - alive_local_size;
    MS_LOG(INFO) << "Codegen for func '" << co.Name() << "' at \"" << co.FileName() << ":" << co.FirstLine()
                 << "\", break at line: " << break_point->line() << ", node: " << ToString(break_point_node);

    if (code.ptr() != nullptr) {
      py::object func = GraphBuilder::FindPyFunc(break_point_node->input(0)->GetVobj());
      MS_EXCEPTION_IF_NULL(func.ptr());
      auto func_ptr = reinterpret_cast<PyFunctionObject *>(func.ptr());
      auto new_code = reinterpret_cast<PyCodeObject *>(code.ptr());
      func_ptr = FunctionNew(func_ptr, new_code);
      func = py::reinterpret_steal<py::object>(reinterpret_cast<PyObject *>(func_ptr));
      // Set name for function.
      const std::string &key = GenerateObjectKey(func);
      MapAdd(call_stack[0]->GetGlobals(), key, func);
      const std::unique_ptr<Instr> &load_global = *(oper.end() - 1 - (new_code->co_argcount + 1));
      MS_EXCEPTION_IF_CHECK_FAIL(load_global->op() == LOAD_GLOBAL,
                                 "Instr should be LOAD_GLOBAL, but is: " + load_global->ToString());
      load_global->set_name(key);
    }
    // Restore stack and alive locals, copy bytecodes after break bci.
    std::vector<std::unique_ptr<Instr>> ops =
      MakeUntrackedCodeHelper(g->GetCFG()->instr_pool(), untracked_bci, activate_argc - alive_local_size, alive_locals);

    if (is_top) {
      // update the LOAD_FAST arg of alive locals, plus stack effect
      std::for_each(ops.begin() + stack_count, ops.begin() + activate_argc, [stack_effect](const auto &i) {
        MS_EXCEPTION_IF_CHECK_FAIL(i->op() == LOAD_FAST, "Instr should be LOAD_FAST, but is: " + i->ToString());
        i->set_arg(i->arg() - stack_effect);
      });
      // For top function, stack value is break point return value, so no need load
      int stack_back = stack_count - 1;
      ops.erase(ops.begin() + stack_back);
      ops.insert(ops.begin() + stack_back, std::make_move_iterator(oper.begin()), std::make_move_iterator(oper.end()));
    } else if (is_bottom) {
      argc += stack_effect;
      offset -= stack_effect;
    } else {
      oper.push_back(std::make_unique<Instr>(STORE_FAST, offset + stack_count - 1));
    }

    int co_argcount = is_top ? top_argc : activate_argc;
    if (!is_top) {
      MS_EXCEPTION_IF_CHECK_FAIL(co.CellVarsSize() == 0, "Cell vars size should be 0");
    }
    std::string co_name = std::string(co.Name()) + "_at_" + std::to_string(untracked_bci);
    CodeGenerator::Code ccode = {
      co_argcount,
      0,
      std::max(co_argcount, co.LocalSize()),
      static_cast<int>(co.ptr()->co_flags & ~(CO_VARARGS | CO_VARKEYWORDS)),
      break_point->line(),
      std::move(ops),
      co.VarNames().cast<std::vector<std::string>>(),
      std::vector<std::string>(),  // cellvars
      GetClosureNames(co.ptr()),   // freevars
      co_name,
      py::reinterpret_borrow<py::object>(co.ptr()->co_filename),
      co_name,
    };
    code = CodeGenerator(std::move(ccode)).NewCode();

    if (is_top) {
      continue;
    }
    // function name will be set in the next iteration.
    // python3.11 add PUSH_NULL
    oper.push_back(std::make_unique<Instr>(LOAD_GLOBAL, IS_PYTHON_3_11_PLUS, ""));
    int load_offset = offset;
    for (int i = 0; i < stack_count; ++i) {
      oper.push_back(std::make_unique<Instr>(LOAD_FAST, load_offset++));
    }
    load_offset -= (is_bottom ? 0 : stack_effect);
    for (int i = 0; i < alive_local_size; ++i) {
      oper.push_back(std::make_unique<Instr>(LOAD_FAST, load_offset++));
    }
    oper.push_back(MakeCallFunInstrPtr(co_argcount));
  }
  if (argc != top_argc) {
    MS_LOG(INTERNAL_EXCEPTION) << "Expect argc to be " << top_argc << ", but is " << argc;
  }
  return code;
}
}  // namespace

bool CodeBreakGenerator::NeedHandleBreakAtCall() const {
  return !IS_PYTHON_3_11_PLUS && is_break_at_call_ && !call_stack_.empty();
}

// Codegen for subgraph break optimization.
void CodeBreakGenerator::BreakAtCall(CodeGenerator *cg) const {
  MS_LOG(DEBUG) << "Do codegen for subgraph break optimization";
  MS_EXCEPTION_IF_CHECK_FAIL(!call_stack_.empty(), "graphs should not be empty!");

  const Instr *break_point = call_stack_.back()->break_info().break_point_;  // break point of bottom function
  Opcode break_op(break_point->op());
  if (break_op.HasJump() && !break_op.IsNotFall()) {
    BreakAtCalleeIfCondition(cg);
    return;
  }

  int stack_effect = GetOpcodeMaxStackEffect(break_point->op(), break_point->arg(), false);
  int argc = SizeToInt(interpret_.outputs.size()) + stack_effect;
  // Pack the uncaptured bytecodes of multiple nested functions into a new function
  py::object code = MakeUntrackedCodeForNestedCalls(call_stack_, argc, break_point->bci() + 1, stack_effect);
  cg->AddInstrs(MakeFunc(code, "<pijit.resume>", GetClosureNames(co_)));

  const Graph::BreakInfo &info = call_stack_.back()->break_info();
  auto alive_local_it = interpret_.outputs.end() - SizeToInt(info.alive_locals_.size());
  for (auto it = interpret_.outputs.begin(); it != alive_local_it; ++it) {
    cg->LoadValue(*it);  // Restore stack.
  }
  // Restore break point op.
  cg->AddInstr(std::make_unique<Instr>(break_point->op(), break_point->arg(), break_point->name()));
  for (auto it = alive_local_it; it != interpret_.outputs.end(); ++it) {
    cg->LoadValue(*it);  // Restore alive locals of bottom-callee-function.
  }
  cg->AddInstr(MakeCallFunInstrPtr(argc));
  cg->NewInstr(RETURN_VALUE);
}

py::object CodeBreakGenerator::MakeUntrackedCodeForNestedCalls(const std::vector<Graph *> &call_stack, int top_argc,
                                                               int untracked_bci, int stack_effect) const {
  MS_LOG(DEBUG) << "Make code object for nested function calls, call stack depth: " << call_stack.size()
                << ", total argc: " << top_argc << ", start bci: " << untracked_bci
                << ", break point stack effect: " << stack_effect;
  py::object code = PackNestedFuncCodes(call_stack_, top_argc, untracked_bci, stack_effect);
  MS_EXCEPTION_IF_NULL(code.ptr());

  auto parent = GetJitCompileResults(co_);
  auto child = CreateJitCompileResults(code);
  child->set_stat(JitCompileResults::GRAPH_CANDIDATE);
  child->set_conf(parent->conf());
  child->set_tbs(parent->tbs());

  auto new_name = py::str(MakeBrkName(PyUnicode_AsUTF8(co_->co_name), untracked_bci));
  auto new_code = reinterpret_cast<PyCodeObject *>(code.ptr());
  REPLACE_PY_MEMBER(new_code->co_name, new_name.ptr());
  if (top_argc != new_code->co_argcount) {
    MS_LOG(INTERNAL_EXCEPTION) << "Expect argc to be " << top_argc << ", but is " << new_code->co_argcount;
  }
  return code;
}

void CodeBreakGenerator::BreakAtCalleeIfCondition(CodeGenerator *cg) const {
  MS_EXCEPTION_IF_CHECK_FAIL(!call_stack_.empty(), "call_stack should not be empty!");
  MS_EXCEPTION_IF_NULL(call_stack_.back());
  const Graph::BreakInfo &break_info = call_stack_.back()->break_info();
  MS_LOG(DEBUG) << "Graph break at condition statement, do codegen for two branches. Break point: "
                << break_info.break_point_->ToString();
  // Restore stack
  int stack_count = SizeToInt(interpret_.outputs.size() - break_info.alive_locals_.size());
  MS_EXCEPTION_IF_CHECK_FAIL(stack_count >= 1, "error stack");
  for (auto it = interpret_.outputs.begin(); it != interpret_.outputs.begin() + stack_count; ++it) {
    cg->LoadValue(*it);
  }
  // Restore break point op
  const Instr *break_point = call_stack_.back()->break_info().break_point_;  // break point of bottom function
  cg->AddInstr(std::make_unique<Instr>(break_point->op(), break_point->arg(), break_point->name()));
  Instr *if_instr = cg->GetCode().co_code.back().get();

  auto MakeCodeForBranch = [this, cg, stack_count](int untracked_bci, int stack_effect, int argc) {
    py::object code = MakeUntrackedCodeForNestedCalls(call_stack_, argc, untracked_bci, stack_effect);

    cg->AddInstrs(MakeFunc(code, "<pijit.resume>", GetClosureNames(co_)));
    cg->AddInstrs(CodeGenerator::RotStack(stack_count + stack_effect));
    for (auto it = interpret_.outputs.begin() + stack_count; it != interpret_.outputs.end(); ++it) {
      cg->LoadValue(*it);  // Restore bottom function alive locals
    }
    cg->AddInstr(MakeCallFunInstrPtr(argc));
    cg->NewInstr(RETURN_VALUE);
  };
  // fall-branch
  MS_LOG(DEBUG) << "Codegen for fall-through branch";
  int stack_effect = -1;
  int argc = SizeToInt(interpret_.outputs.size()) + stack_effect;
  MakeCodeForBranch(break_point->bci() + 1, stack_effect, argc);
  // jump-branch
  MS_LOG(DEBUG) << "Codegen for jump branch";
  int target_bci = SizeToInt(cg->GetCode().co_code.size());
  stack_effect = (break_point->op() == JUMP_IF_TRUE_OR_POP || break_point->op() == JUMP_IF_FALSE_OR_POP) ? 0 : -1;
  argc = SizeToInt(interpret_.outputs.size()) + stack_effect;
  MakeCodeForBranch(break_point->extra_jump()->bci(), stack_effect, argc);
  if_instr->set_extra_jump(cg->GetCode().co_code[target_bci].get());
}

void CodeBreakGenerator::CallUntrackedCode(CodeGenerator *code_gen) {
  if (break_bci_ == -1) {
    return;
  }
  MS_LOG(DEBUG) << "Do codegen for untracked code";
  if (NeedHandleBreakAtCall()) {
    BreakAtCall(code_gen);
    return;
  }
  const auto &list = GetCFG()->instr_pool();
  int start_bci = break_bci_;
  Opcode start_op(list[start_bci]->op());

  int untracked_bci;
  int untracked_stack_effect;
  bool find_block = FindBlock(start_bci, GetCFG(), &untracked_bci, &untracked_stack_effect);
  untracked_bci++;
  if (IsNotNeedTrack(GetCFG()->instr_pool(), std::min(untracked_bci + 1, SizeToInt(list.size())))) {
    RestoreStack(code_gen);
    RestoreLocals(code_gen, false);
    code_gen->AddInstructionWithExceptionTable(GetCFG(), break_bci_, GetCFG()->instr_pool().size());
    return;
  }
  if (find_block) {
    BreakAtBlock(code_gen, untracked_bci, untracked_stack_effect);
    return;
  }
  if (start_op.HasJump() && !start_op.IsNotFall()) {
    BreakAtIf(code_gen);
    return;
  }
  if (start_op != JUMP_ABSOLUTE && start_op != JUMP_FORWARD) {
    MS_EXCEPTION_IF_CHECK_FAIL(list[start_bci]->extra_jump() == nullptr, "unexpected jump instruction");
    // break at unsupported bytecode
    untracked_stack_effect = GetOpcodeMaxStackEffect(start_op, list[start_bci]->arg(), false);
    untracked_bci++;
  }

  int argc;
  py::object code = MakeUntrackedCode(untracked_bci, untracked_stack_effect, &argc);
  PyCodeWrapper co(co_);
  int load_args_offset = code_gen->GetCode().co_code.size();
  code_gen->AddInstrs(MakeFunc(code, "<pijit.resume>", GetClosureNames(co_)));

  ReconstructStack(code_gen, untracked_bci, untracked_stack_effect);
  RestoreLocals(code_gen, true);

  code_gen->AddCallInstr(load_args_offset, argc);
  code_gen->NewInstr(RETURN_VALUE);
}

py::object CodeBreakGenerator::MakeDispatchCode() {
  CodeGenerator code_gen(&interpret_, true);

  if (IsCopyCapturedInstructions()) {
    MS_LOG(DEBUG) << "No graph captured";
    int stack_count = interpret_.outputs.size() - alive_locals_.size();
    int output_index = 0;
    std::vector<std::unique_ptr<Instr>> instrs;
    std::vector<ValueNode *> locals(co_->co_nlocals + stack_count, &ValueNode::kUnboundLocal);
    for (; output_index < stack_count; ++output_index) {
      locals[co_->co_nlocals + output_index] = interpret_.outputs[output_index];
      instrs.push_back(std::make_unique<Instr>(STORE_FAST, co_->co_nlocals + output_index));
    }
    for (size_t i = 0, size = alive_locals_.size(); i < size; ++i, ++output_index) {
      locals[alive_locals_[i]] = interpret_.outputs[output_index];
    }
    std::swap(locals, interpret_.inputs);
    code_gen.Init();
    code_gen.AddInstrs(CodeGenerator::CopyInstr(GetCFG()->instr_pool(), 0, break_bci_, true));
    code_gen.AddInstrs({std::make_move_iterator(instrs.rbegin()), std::make_move_iterator(instrs.rend())});
    std::swap(locals, interpret_.inputs);
  } else {
    code_gen.SetGlobals(globals_);
    code_gen.Init();
    for (auto i : captured_.inputs) {
      code_gen.MarkAlive(i);
    }
    for (auto i : outputs_optimize_.inputs) {
      code_gen.MarkAlive(i);
    }
    code_gen.Build();

    CallCapturedCode(&code_gen);
    FixInterpretOuput(&code_gen);
  }
  CallUntrackedCode(&code_gen);
  MakeReturn(&code_gen);

  std::string co_name = PyUnicode_AsUTF8(co_->co_name);
  auto jcr = GetJitCompileResults(co_);
  MS_EXCEPTION_IF_NULL(jcr);
  co_name = std::to_string(jcr->IncCodeCount()) + "R." + co_name;

  int nlocals = SizeToInt(code_gen.GetLocalsMap().size());
  nlocals = std::max(nlocals, co_->co_nlocals);
  nlocals = std::max(nlocals, cfg_->GetLocalCount());

  ExtendCodeInfo(&code_gen, false);
  code_gen.SetLocalsCount(nlocals);
  code_gen.SetCodeName(co_name);

  py::object result = code_gen.NewCode();
  return result;
}

void CodeBreakGenerator::MakeReturn(CodeGenerator *code_gen) const {
  if (IsCopyCapturedInstructions() && break_bci_ == -1) {
    return;
  }
  if (break_bci_ != -1) {
    // call untracked nodes
    return;
  }
  if (captured_.operations.empty()) {
    // all values is interpret produce
    code_gen->GenReturn();
    return;
  }
  // not break graph, mix interpret and graph
  MS_EXCEPTION_IF_CHECK_FAIL(interpret_.outputs.size() > 0, "error outputs");
  ValueNode *rv = interpret_.outputs[0];
  auto iter = code_gen->GetLocalsMap().find(rv);
  if (iter != code_gen->GetLocalsMap().end() || IsNonLocalValue(rv)) {
    code_gen->LoadValue(rv);
    code_gen->NewInstr(RETURN_VALUE);
    return;
  }
  MS_EXCEPTION_IF_CHECK_FAIL(captured_.outputs.size() == 1 && extra_local_ != -1,
                             "can't find return value from interpret locals and graph locals");
  code_gen->NewInstr(LOAD_FAST, extra_local_);
  code_gen->NewInstr(RETURN_VALUE);
}

py::object CodeBreakGenerator::MakeInterpretCapturedCode() const {
  auto jcr = GetJitCompileResults(co_);
  MS_EXCEPTION_IF_NULL(jcr);

  CodeGenerator code_gen(&interpret_, true);
  code_gen.SetGlobals(globals_);
  code_gen.Init();
  code_gen.Build();
  code_gen.GenReturn();

  std::string co_name = MakeCompiledName(PyUnicode_AsUTF8(co_->co_name));
  co_name = std::to_string(jcr->IncCodeCount()) + "R." + co_name;

  int nlocals = SizeToInt(code_gen.GetLocalsMap().size());
  nlocals = std::max(nlocals, co_->co_nlocals);
  nlocals = std::max(nlocals, cfg_->GetLocalCount());

  ExtendCodeInfo(&code_gen, true);
  code_gen.SetLocalsCount(nlocals);
  code_gen.SetCodeName(co_name);

  py::object result = code_gen.NewCode();

  JitCompileResults *child = CreateJitCompileResults(result.ptr());
  MS_EXCEPTION_IF_NULL(child);
  child->set_stat(JitCompileResults::GRAPH_CAPTURED);
  child->set_conf(jcr->conf());
  child->set_tbs(jcr->tbs());
  return result;
}

void CodeBreakGenerator::ExtendCodeInfo(CodeGenerator *cg, bool merge_kw_only) const {
  int argc = merge_kw_only ? (co_->co_argcount) + co_->co_kwonlyargcount : co_->co_argcount;
  int kw_only = merge_kw_only ? 0 : co_->co_kwonlyargcount;

  PyCodeWrapper co(co_);
  auto varnames = co.VarNames();
  auto cellvars = co.CellVars();
  auto freevars = co.FreeVars();
  cg->SetArgsInfo(argc, kw_only);
  cg->SetLocalsCount(co_->co_nlocals);
  cg->SetCodeFlags(co_->co_flags);
  cg->SetFirstLineNumber(co_->co_firstlineno);
  cg->SetVariableNames(py::cast<std::vector<std::string>>(varnames));
  cg->SetCellVariableNames(py::cast<std::vector<std::string>>(cellvars));
  cg->SetFreeVariableNames(py::cast<std::vector<std::string>>(freevars));
  cg->SetFileName(py::reinterpret_borrow<py::object>(co_->co_filename));
}

void CodeBreakGenerator::Init(const GraphAnalyzer &analyzer, Graph *graph) {
  alive_locals_ = graph->break_info().alive_locals_;
  break_bci_ = graph->GetStopTraceBci();
  cfg_ = graph->GetCFG().get();
  const GraphAnalyzer::CapturedInfo &info = analyzer.GetCaptureInfo();
  interpret_.inputs = info.interpret_.inputs;
  interpret_.outputs = info.interpret_.outputs;
  interpret_.operations = info.interpret_.operations;
  captured_.inputs = info.captured_.inputs;
  captured_.outputs = info.captured_.outputs;
  captured_.operations = info.captured_.operations;
  outputs_optimize_.inputs = info.outputs_optimize_.inputs;
  outputs_optimize_.outputs = info.outputs_optimize_.outputs;
  outputs_optimize_.operations = info.outputs_optimize_.operations;
  replaced_nodes_ = info.replaced_nodes_;
  no_graph_ = captured_.operations.empty();

  const auto &break_info = analyzer.graph_break_info();
  is_break_at_call_ = break_info.is_break_at_call;
  if (break_info.is_break_at_call && !break_info.captured_subgraphs.empty()) {
    // For subgraph break optimization.
    call_stack_.push_back(graph);  // top-graph
    call_stack_.insert(call_stack_.end(), break_info.captured_subgraphs.begin(), break_info.captured_subgraphs.end());
  }

  if (analyzer.NeedInterpret()) {
    return;
  }
  // all parameters is graph support
  captured_.inputs.clear();
  captured_.outputs.clear();
  interpret_.operations = std::move(captured_.operations);
}

void CodeBreakGenerator::BuildGraphParameters(const std::unordered_map<ValueNode *, int> &locals,
                                              GraphParameterBuilder *builder) {
  // NOTE: if *vargs is cell variable, it is not parameter node
  MS_EXCEPTION_IF_CHECK_FAIL(co_->co_nlocals == SizeToInt(interpret_.inputs.size()),
                             "interpret inputs must be same as locals");

  builder->Init(captured_.inputs);
  builder->Build(locals);
}

std::string GraphParameterBuilder::Key(int index, ValueNode *n) {
  static uint64_t kId = 0;
  PyTypeObject *tp = n->GetVobj() ? n->GetVobj()->GetTypeObject() : nullptr;
  std::string descr = AObject::GetTypeDesc(n->GetVobj() ? n->GetVobj()->GetType() : AObject::kTypeAnyValue);
  std::stringstream s;
  s << "<" << index << ">" << (tp ? (tp->tp_name ? tp->tp_name : "<unnamed>") : descr) << "<" << (kId++) << ">";
  return s.str();
}

void GraphParameterBuilder::Init(const std::vector<ValueNode *> &args) { args_ = args; }

void GraphParameterBuilder::Build(const std::unordered_map<ValueNode *, int> &locals) {
  auto Load = [&locals](ValueNode *param) {
    // 1.Search from local variables
    auto iter = locals.find(param);
    if (iter != locals.end()) {
      return std::make_unique<Instr>(LOAD_FAST, iter->second);
    }
    // 2.Maybe a closure variable
    if (param->GetOpcode() == LOAD_DEREF) {
      return std::make_unique<Instr>(LOAD_DEREF, param->GetOparg(), param->GetName());
    }
    MS_EXCEPTION_IF_CHECK_FAIL(false, "Can't find graph parameters from interpret-locals and closures");
  };
  std::transform(args_.begin(), args_.end(), std::back_inserter(load_), Load);
}

// e.g. while..., for..., while...else..., for...else...,
static int FindLoopEnd(int start, const CFG *cfg) {
  Block *loop_begin = cfg->GetBlockByBci(start);
  if (!loop_begin->is_loop_head()) {
    return start - 1;
  }

  const auto &instrs = cfg->instr_pool();
  int loop_exit = loop_begin->begin_ci();
  int target = loop_exit;
  for (const auto &i : loop_begin->succ_bbs()) {
    target = std::max(target, i->begin_ci());
  }
  // find loop last exit
  for (; loop_exit != target; ++loop_exit) {
    Instr *jump = instrs[loop_exit]->extra_jump();
    if (jump == nullptr) {
      continue;
    }
    if (target < jump->bci()) {
      // if jump forward out of loop branch target, reset target
      target = jump->bci();
    }
  }
  // find last backward edge, get next instruction
  int result = 0;
  for (auto i : loop_begin->pred_bbs()) {
    result = std::max(result, i->end_ci());
  }
  return std::max(result, target) - 1;
}

#if IS_PYTHON_3_11_PLUS

static size_t FindTryWithBlockEnd(const CFG *cfg, ExceptionTable::const_iterator begin) {
  const InstructionList &i_list = cfg->instr_pool();
  int handler = begin->second.jump_;
  int max_bci = i_list.size();
  MS_EXCEPTION_IF_CHECK_FAIL(handler > 0, "exception item can't jump to first instruction");
  if (i_list[handler - 1]->extra_jump() == nullptr) {
    return max_bci - 1;
  }
  int origin_bci = i_list[handler - 1]->extra_jump()->bci();
  int end_bci = origin_bci;
  // for each item, find last jump
  for (auto iter = begin; iter != cfg->exc_table().end() && iter->second.end_ < end_bci; ++iter) {
    handler = iter->second.jump_;
    if (i_list[handler]->op() == PUSH_EXC_INFO && handler != 0) {
      Instr *tar = i_list[handler - 1]->extra_jump();
      end_bci = std::max(tar ? tar->bci() : max_bci, end_bci);
    }
  }
  if (origin_bci != end_bci) {
    MS_LOG(DEBUG) << "maybe has finally block at range [" << begin->second.begin_ << "," << end_bci << ")";
  }
  return end_bci - 1;
}

static bool FindBlock(int start_bci, const CFG *cfg, int *end_bci, int *stack_effect) {
  const std::vector<std::unique_ptr<Instr>> &list = cfg->instr_pool();
  *stack_effect = 0;
  int opcode = list[start_bci]->op();
  auto exception_table_iter = cfg->FindTryWithBlock(start_bci);
  if (exception_table_iter != cfg->exc_table().end()) {
    *end_bci = FindTryWithBlockEnd(cfg, exception_table_iter);
    return *end_bci;
  } else if (opcode == FOR_ITER) {
    *stack_effect = -1;
  }
  *end_bci = FindLoopEnd(start_bci, cfg);
  if (list[start_bci]->op() == FOR_ITER && *end_bci == start_bci - 1) {
    // break at FOR_ITER and it is not a loop
    *end_bci = list[start_bci]->extra_jump()->bci() - 1;
  }
  return *end_bci != start_bci - 1;
}

#elif !IS_PYTHON_3_9_PLUS

static size_t FindTryBlockEnd(int start_bci, const CFG *cfg) {
  const auto &list = cfg->instr_pool();
  size_t block_end = list[start_bci]->extra_jump()->bci();
  for (; block_end < list.size() && list[block_end]->op() != END_FINALLY; ++block_end) {
  }
  if (list[block_end - 1]->extra_jump()) {
    size_t jump = list[block_end - 1]->extra_jump()->bci();
    block_end = std::max(block_end, jump);
  }
  return block_end;
}

static bool FindBlock(int start_bci, const CFG *cfg, int *end_bci, int *stack_effect) {
  const auto &list = cfg->instr_pool();
  size_t block_end = 0;
  *stack_effect = 0;
  int opcode = list[start_bci]->op();
  if (opcode == Opcode::k_ILLEGAL_OPCODE) {
    MS_LOG(INTERNAL_EXCEPTION) << "shouldn't reach here";

#if (PY_MAJOR_VERSION == 3) && (PY_MINOR_VERSION == 7)
  } else if (opcode == SETUP_EXCEPT) {
    block_end = FindTryBlockEnd(start_bci, cfg);
  } else if (opcode == SETUP_LOOP) {
    block_end = list[start_bci]->extra_jump()->bci() - 1;
  } else if (opcode == FOR_ITER) {
    block_end = FindLoopEnd(start_bci, cfg);
    *stack_effect = -1;
#endif
#if (PY_MAJOR_VERSION == 3) && (PY_MINOR_VERSION == 8)
  } else if (opcode == BEGIN_FINALLY || opcode == CALL_FINALLY) {
    MS_EXCEPTION_IF_CHECK_FAIL(false, "shouldn't reach here, must be break at SETUP_FINALLY");
  } else if (opcode == FOR_ITER) {
    block_end = FindLoopEnd(start_bci, cfg);
    *stack_effect = -1;
#endif
  } else if (opcode == SETUP_WITH || opcode == SETUP_FINALLY) {
    if (opcode == SETUP_WITH) {
      *stack_effect = -1;
    }
    block_end = FindTryBlockEnd(start_bci, cfg);
  } else {
    block_end = FindLoopEnd(start_bci, cfg);
  }
  if (list[start_bci]->op() == FOR_ITER && SizeToInt(block_end) == start_bci - 1) {
    // break at FOR_ITER and it is not a loop
    block_end = list[start_bci]->extra_jump()->bci() - 1;
  }
  *end_bci = block_end;
  return SizeToInt(block_end) != start_bci - 1;
}

#else

static int FindWithBlockEnd(int start_bci, const CFG *cfg) {
  const auto &list = cfg->instr_pool();
  size_t tar = (size_t)list[start_bci]->extra_jump()->bci();
  bool validate = tar + 1 < list.size() && list[tar]->op() == WITH_EXCEPT_START && list[tar + 1]->extra_jump();
  MS_EXCEPTION_IF_CHECK_FAIL(validate, "can't find with block");
  return list[tar - 1]->extra_jump() ? list[tar - 1]->extra_jump()->bci() - 1 : list.back()->bci();
}

static int FindTryBlockEnd(int start, const CFG *cfg) {
  const auto &list = cfg->instr_pool();
  Instr *tar = list[start]->extra_jump();
  MS_EXCEPTION_IF_NULL(tar);

  size_t res = (size_t)tar->bci();
  auto is_copy_top_op = tar->op() == DUP_TOP || (tar->op() == COPY && tar->arg() == 1);
  if (is_copy_top_op) {
    // try block without finally
    MS_EXCEPTION_IF_CHECK_FAIL(res + 2 < list.size(), "can't find try block");
    while (res < list.size() && list[res]->op() != RERAISE) {
      res = list[res + 2]->extra_jump()->bci();
    }
    if (list[res - 1]->op() == JUMP_FORWARD) {
      res = list[res - 1]->extra_jump()->bci();
    }
    return res;
  }

  int tryForwordBci = tar->bci() - 1;
  if (list[tryForwordBci]->op() != JUMP_FORWARD) {
    return list.back()->bci();
  }
  int finallyOrElseBci = list[tryForwordBci]->extra_jump()->bci();
  if (list.size() < static_cast<unsigned int>(finallyOrElseBci)) {
    return list.back()->bci();
  }
  constexpr auto preced2 = 2;
  int precedFinallyOrElseOp = list[finallyOrElseBci - 1]->op();
  int precedFinallyOrElseOp2 = list[finallyOrElseBci - preced2]->op();
  if (precedFinallyOrElseOp == RERAISE && precedFinallyOrElseOp2 == JUMP_FORWARD) {
    // try/except/else 无finally的场景
    return list[finallyOrElseBci - preced2]->extra_jump()->bci();
  } else if (precedFinallyOrElseOp == RERAISE) {
    // try/except/else/finally
    return finallyOrElseBci;
  } else {
    return list.back()->bci();
  }
}

static bool FindBlock(int start_bci, const CFG *cfg, int *end_bci, int *stack_effect) {
  const std::vector<std::unique_ptr<Instr>> &list = cfg->instr_pool();
  *stack_effect = 0;
  int opcode = list[start_bci]->op();
  if (opcode == SETUP_FINALLY) {
    *end_bci = FindTryBlockEnd(start_bci, cfg);
    return true;
  } else if (opcode == SETUP_WITH) {
    *end_bci = FindWithBlockEnd(start_bci, cfg);
    *stack_effect = -1;
    return true;
  } else if (opcode == FOR_ITER) {
    *stack_effect = -1;
  }
  *end_bci = FindLoopEnd(start_bci, cfg);
  if (list[start_bci]->op() == FOR_ITER && *end_bci == start_bci - 1) {
    // break at FOR_ITER and it is not a loop
    *end_bci = list[start_bci]->extra_jump()->bci() - 1;
  }
  return *end_bci != start_bci - 1;
}
#endif

py::object MakeCodeFromCodeGen(const GraphBuilderPtr &builder, const GraphAnalyzerPtr &analyzer, PyObject *globals) {
  TimeRecorder time_recorder(__FUNCTION__, kPIJitConfigDefault.GetBoolConfig(GraphJitConfig::kLogPerf));
  MS_LOG(INFO) << "Start MakeCodeFromCodeGen";
  int break_bci = builder->GetGraph()->GetStopTraceBci();
  bool skip_compile = analyzer->GetCaptureInfo().captured_.operations.empty() && break_bci == -1;
#ifdef IS_PYTHON_3_11_PLUS
  {
    const auto &cfg = builder->GetGraph()->GetCFG();
    auto iter = cfg->FindTryWithBlock(break_bci);
    auto not_found = cfg->exc_table().end();
    if (iter == not_found && cfg->FindExcTableItem(break_bci) != not_found) {
      MS_LOG(INFO) << "not implemented break at try while handle exception";
      skip_compile = true;
    }
  }
#endif
  if (skip_compile) {
    MS_LOG(INFO) << "skip graph capture because of no graph in the function";
    // no graph, no guard needed
    builder->GetGraph()->RemoveAllGuardItems();
    return PyCodeWrapper(builder->GetGraph()->GetCodeObj()).DeepCopy();
  }
  auto graph = builder->GetGraph();
  auto cg = std::make_shared<CodeBreakGenerator>(builder, py::cast<py::dict>(globals), graph->GetCodeObj());
  cg->Init(*analyzer, graph);
  py::object code = analyzer->NeedInterpret() ? cg->MakeDispatchCode() : cg->MakeCapturedCode();
  MS_LOG(INFO) << "End MakeCodeFromCodeGen";
  return code;
}

std::string PrintInstr(const std::vector<std::unique_ptr<Instr>> &list) {
  std::stringstream s;
  for (const auto &i : list) {
    s << i->ToString() << "\n";
  }
  return s.str();
}

std::string PrintNodeSet(const NodeSet &nodes) {
  std::stringstream s;
  s << "inputs: \n";
  for (auto i : nodes.inputs) {
    s << i->ToString() << "\n";
  }
  s << "outputs: \n";
  for (auto i : nodes.outputs) {
    s << i->ToString() << "\n";
  }
  s << "operations: \n";
  for (auto i : nodes.operations) {
    s << i->ToString() << "\n";
  }
  return s.str();
}

py::object CodeBreakGenerator::MakeCapturedCode() const {
  auto jcr = GetJitCompileResults(co_);
  MS_EXCEPTION_IF_NULL(jcr);
  if (jcr->conf()->GetBoolConfig(GraphJitConfig::kInterpretCapturedCode)) {
    return MakeInterpretCapturedCode();
  }
  auto name = std::to_string(jcr->IncCodeCount()) + "R." + MakeCompiledName(PyUnicode_AsUTF8(co_->co_name));
  Compile(name, co_->co_argcount, co_->co_kwonlyargcount, co_->co_flags, py::object());
  return py::object();
}

void CodeBreakGenerator::Compile(const std::string &co_name, int co_argcount, int co_kwonlyargcount, int co_flags,
                                 const py::object &stub) const {
  TimeRecorder compile_time("MindCodeCompile", kPIJitConfigDefault.GetBoolConfig(GraphJitConfig::kLogPerf));

  auto compile_result = FGBuilder()->GetCompileResult();
  if (compile_result.first.empty() || compile_result.second == nullptr) {
    // Compile graph.
    FGBuilder()->ClearNodeAbstract();
    FGBuilder()->SetGraphName(co_name);
    auto func_graph = FGBuilder()->graph();
    auto origin_top_input_num = FGBuilder()->origin_top_input_num();
    GraphCompiler::CompileInfo compile_info{py::cast<std::string>(co_->co_filename),
                                            co_name,
                                            co_->co_firstlineno,
                                            co_argcount,
                                            co_kwonlyargcount,
                                            co_flags,
                                            origin_top_input_num};
    compile_result = GraphCompiler::Compile(func_graph, compile_info);
  }
  // Set NativeFunc.
  auto parent = GetJitCompileResults(co_);
  MS_EXCEPTION_IF_NULL(parent);
  if (stub.ptr() == nullptr) {
    parent->code()->SetNativeFunc(compile_result.first, compile_result.second, nullptr);
    parent->set_stat(JitCompileResults::GRAPH_CALLABLE);
  } else {
    JitCompileResults *child = CreateJitCompileResults(stub.ptr());
    MS_EXCEPTION_IF_NULL(child);
    MS_EXCEPTION_IF_CHECK_FAIL(child->code() == nullptr, "must be a new stub code");
    child->set_code(child->codehub()->AddOptTarget(OptOption::CreateOptionByPoint(child)));
    child->code()->SetNativeFunc(compile_result.first, compile_result.second, nullptr);
    child->set_stat(JitCompileResults::GRAPH_CALLABLE);
    child->set_conf(parent->conf());
    child->set_tbs(parent->tbs());
  }
}

std::string LoopBodyReCaptureCodeGenerator::makeLoopBodyFuncName(int loopBodyStartBci, int loopBodyEndBci) const {
  const std::string &co_name = PyUnicode_AsUTF8(co_->co_name);
  auto name =
    co_name + ".wrapped_loop_body_func." + std::to_string(loopBodyStartBci) + "." + std::to_string(loopBodyEndBci);
  return name;
}

std::string LoopBodyReCaptureCodeGenerator::makeFuncName(int loopBodyStartBci, int loopBodyEndBci) const {
  const std::string &co_name = PyUnicode_AsUTF8(co_->co_name);
  auto name =
    co_name + ".loop_body_recaptured." + std::to_string(loopBodyStartBci) + "." + std::to_string(loopBodyEndBci);
  return name;
}

/**
 * Get all closure names of the current function
 */
std::vector<std::string> LoopBodyReCaptureCodeGenerator::GetClosureNames() const {
  PyCodeWrapper co(co_);
  py::tuple cells = co.CellVars();
  py::tuple frees = co.FreeVars();

  std::vector<std::string> names;
  for (Py_ssize_t i = 0; i < PyTuple_GET_SIZE(cells.ptr()); ++i) {
    (void)names.emplace_back(PyUnicode_AsUTF8(PyTuple_GET_ITEM(cells.ptr(), i)));
  }
  for (Py_ssize_t i = 0; i < PyTuple_GET_SIZE(frees.ptr()); ++i) {
    (void)names.emplace_back(PyUnicode_AsUTF8(PyTuple_GET_ITEM(frees.ptr(), i)));
  }
  return names;
}

/**
 * @brief Encapsulates a loop body into a new function based on the given bytecode index (BCI) range and
 *        generates the corresponding PyCodeObject.
 *
 * This function extracts the loop body defined by the bytecode range from `loopBodyStartBci` to `loopBodyEndBci`,
 * and packages it into a new function. It uses the provided lists of live variables at the start and end of the loop
 * (`inputLocals` and `outputLocals`) to define the function's input and output.
 *
 * @param loopBodyStartBci The bytecode index marking the start of the loop body.
 * @param loopBodyEndBci The bytecode index marking the end of the loop body.
 * @param inputLocals A vector of live variables at the start of the loop body (inputs for the new function).
 * @param outputLocals A vector of live variables at the end of the loop body (outputs for the new function).
 * @param ifForLoop ifForLoop
 *
 * @return A PyCodeObject representing the new encapsulated function for the loop body.
 */
py::object LoopBodyReCaptureCodeGenerator::MakeLoopBodyCode(int loopBodyStartBci, int loopBodyEndBci,
                                                            const std::vector<int> &inputLocals,
                                                            const std::vector<int> &outputLocals,
                                                            bool ifForLoop) const {
  int stack_count = is_for_loop_ ? 1 : 0;
  const int argc = static_cast<int>(inputLocals.size()) + stack_count;

  // Parameter assembly rule: first the stackEffect, then the local live variables.
  std::vector<std::unique_ptr<Instr>> ld;
  std::vector<std::unique_ptr<Instr>> st;
  for (int i = 0; i < stack_count; ++i) {
    (void)ld.emplace_back(std::make_unique<Instr>(LOAD_FAST, i));
  }
  int index = stack_count;
  // Here, the input parameter slots are mapped to the slots used by the original bytecode,
  // so there's no need to modify the slots in the encapsulated bytecode.
  for (auto iter = inputLocals.begin(); iter != inputLocals.end(); ++iter, ++index) {
    if (*iter != index) {
      (void)ld.emplace_back(std::make_unique<Instr>(LOAD_FAST, index));
      (void)st.emplace_back(std::make_unique<Instr>(STORE_FAST, *iter));
    }
  }
  std::vector<std::unique_ptr<Instr>> resultInstrs = std::move(ld);
  std::move(st.rbegin(), st.rend(), std::back_inserter(resultInstrs));
  std::vector<std::unique_ptr<Instr>> loopBodyInstrs =
    CodeGenerator::CopyInstr(graph_->GetCFG()->instr_pool(), loopBodyStartBci, loopBodyEndBci, false, true);
  int first_line = loopBodyInstrs[0]->bci();
  std::move(loopBodyInstrs.begin(), loopBodyInstrs.end(), std::back_inserter(resultInstrs));
  // Returns the live variables at the end of the loop body.
  std::vector<std::unique_ptr<Instr>> returnInstrs;
  std::transform(outputLocals.begin(), outputLocals.end(), std::back_inserter(returnInstrs),
                 [](const auto &i) { return std::make_unique<Instr>(LOAD_FAST, i); });
  if (outputLocals.size() > 1) {
    (void)returnInstrs.emplace_back(std::make_unique<Instr>(BUILD_TUPLE, outputLocals.size()));
  }
  if (outputLocals.size() == 0) {
    (void)returnInstrs.emplace_back(std::make_unique<Instr>(LOAD_CONST, 0));
    returnInstrs.back()->set_cnst(py::none());
  }
  (void)returnInstrs.emplace_back(std::make_unique<Instr>(RETURN_VALUE));
  std::move(returnInstrs.begin(), returnInstrs.end(), std::back_inserter(resultInstrs));
  if (IsPiJitLogOn(LogCfg::kOthers)) {
    std::stringstream ss;
    for (auto &instr : resultInstrs) {
      ss << instr->ToString() << std::endl;
    }
    MS_LOG(WARNING) << "Instrs of wrapped loop body:" << std::endl << ss.str();
  }
  auto varnames = py::cast<std::vector<std::string>>(PyCodeWrapper(co_).VarNames());
  CodeGenerator::Code ccode = {argc,
                               0,
                               static_cast<int>(varnames.size()),
                               (signed)co_->co_flags & ~(CO_VARARGS | CO_VARKEYWORDS),
                               first_line,
                               std::move(resultInstrs),
                               varnames,
                               std::vector<std::string>(),
                               GetClosureNames(),
                               makeLoopBodyFuncName(loopBodyStartBci, loopBodyEndBci),
                               py::reinterpret_borrow<py::object>(co_->co_filename),
                               "<pijit.loopBody>"};
  py::object code = CodeGenerator(std::move(ccode)).NewCode();
  auto parent = GetJitCompileResults(co_);
  MS_EXCEPTION_IF_NULL(parent);
  JitCompileResults *child = CreateJitCompileResults(code.ptr());
  MS_EXCEPTION_IF_NULL(child);
  child->set_stat(JitCompileResults::GRAPH_CANDIDATE);
  child->set_conf(parent->conf());
  child->set_tbs(parent->tbs());
  child->set_is_for_loop_body_wrapper(is_for_loop_);
  return code;
}

bool LoopBodyReCaptureCodeGenerator::Prepare() {
  Block *breakBlock = nullptr;
  auto break_bci = graph_->GetStopTraceBci();
  for (auto &bb : graph_->GetCFG()->bb_pool()) {
    if (break_bci < bb->end_ci() && break_bci >= bb->begin_ci()) {
      breakBlock = bb.get();
      break;
    }
  }
  if (!breakBlock) {
    MS_LOG(WARNING) << "Failed to get block of break bci, skip...";
    return false;
  }
  Block *loopHeadBB = breakBlock->loop_head_bb();
  if (!loopHeadBB) {
    MS_LOG(WARNING) << "Failed to find loop head, skip...";
    return false;
  }
  Instr *loopControlInstr = graph_->GetCFG()->GetBlockTail(loopHeadBB);
  is_for_loop_ = loopControlInstr->op() == FOR_ITER;
  std::vector<Block *> loopBodySortedBBs;
  for (auto bb : loopHeadBB->loop_body_bbs()) {
    if (!bb->is_loop_head() && bb->is_loop_body()) {
      (void)loopBodySortedBBs.emplace_back(bb);
    }
  }
  std::sort(loopBodySortedBBs.begin(), loopBodySortedBBs.end(),
            [](Block *left, Block *right) { return left->id() < right->id(); });
  if (IsPiJitLogOn(LogCfg::kOthers)) {
    std::stringstream ss;
    ss << "====>DUMP LOOP BB START<====" << std::endl;
    for (auto b : loopBodySortedBBs) {
      ss << "" << b->Dump() << std::endl;
    }
    ss << "====>DUMP LOOP BB END<====" << std::endl;
    MS_LOG(WARNING) << ss.str();
  }
  loopBodyStartBci_ = loopBodySortedBBs.front()->begin_ci();
  // Remove the jump instruction at the end of the loop body (jumping back to the loop header).
  loopBodyEndBci_ = loopBodySortedBBs.back()->end_ci() - 1;
  MS_LOG(WARNING) << "loopBodyStartBci:" << loopBodyStartBci_ << ",loopBodyEndBci:" << loopBodyEndBci_;
  // Traverse the bytecode and exit the process if unsupported bytecodes (e.g., return/break/continue) are found.
  std::vector<int> unsupported_instructions = {
    RETURN_VALUE, YIELD_VALUE, RAISE_VARARGS, POP_BLOCK, END_FINALLY,
  };
  std::vector<int> jump_instructions = {
    JUMP_ABSOLUTE,
    JUMP_FORWARD,
  };
  for (int bci = loopBodyStartBci_; bci < loopBodyEndBci_; ++bci) {
    auto &instr = graph_->GetCFG()->instr_pool()[bci];
    auto unsupported_it = std::find(unsupported_instructions.begin(), unsupported_instructions.end(), instr->op());
    if (unsupported_it != unsupported_instructions.end()) {
      MS_LOG(WARNING) << "Loop body contains unsupported bytecode " << Opcode(instr->op()).name();
      return false;
    }
    auto jump_it = std::find(jump_instructions.begin(), jump_instructions.end(), instr->op());
    if (jump_it != jump_instructions.end()) {
      if (instr->extra_jump() &&
          (instr->extra_jump()->bci() > loopBodyEndBci_ || instr->extra_jump()->bci() < loopBodyStartBci_ - 1)) {
        MS_LOG(WARNING) << "The loop body contains jump instructions that jump outside the loop. bytecode: "
                        << Opcode(instr->op()).name() << ", target:" << instr->extra_jump()->bci();
        return false;
      }
    }
  }
  return true;
}

py::object LoopBodyReCaptureCodeGenerator::Build() {
  if (IsPiJitLogOn(LogCfg::kOthers)) {
    std::stringstream ss;
    ss << "Instrs Before ReCapture:" << std::endl;
    for (auto &instr : graph_->GetCFG()->instr_pool()) {
      ss << instr->ToString() << std::endl;
    }
    MS_LOG(WARNING) << ss.str();
  }
  BitMap aliveAtStart = graph_->GetCFG()->GetLiveness()->CollectAlive(loopBodyStartBci_);
  BitMap aliveAtEnd = graph_->GetCFG()->GetLiveness()->CollectAlive(loopBodyEndBci_);
  std::vector<int> localsForInput;
  for (BitMap::Iter iter(&aliveAtStart, true), end(&aliveAtStart, false); iter != end; ++iter) {
    localsForInput.push_back(*iter);
  }
  std::vector<int> localsForOutput;
  for (BitMap::Iter iter(&aliveAtEnd, true), end(&aliveAtEnd, false); iter != end; ++iter) {
    localsForOutput.push_back(*iter);
  }

  // When in a For loop, the stack effect is the FOR_ITER.
  int stack_effect = is_for_loop_ ? 1 : 0;
  auto LoopBodyCode =
    MakeLoopBodyCode(loopBodyStartBci_, loopBodyEndBci_, localsForInput, localsForOutput, is_for_loop_);
  std::vector<std::unique_ptr<Instr>> newLoopBodyInstrs;
  std::transform(localsForInput.begin(), localsForInput.end(), std::back_inserter(newLoopBodyInstrs),
                 [](int &local) { return std::make_unique<Instr>(LOAD_FAST, local); });
  PyCodeWrapper co(co_);
  // keep origin closures
  const auto &closures = GetClosureNames();
  auto func_instrs = MakeFunc(LoopBodyCode, "<pijit.loopBody>", closures);
  std::move(func_instrs.begin(), func_instrs.end(), std::back_inserter(newLoopBodyInstrs));
  auto rot_stack_instrs = CodeGenerator::RotStack(localsForInput.size() + stack_effect);
  std::move(rot_stack_instrs.begin(), rot_stack_instrs.end(), std::back_inserter(newLoopBodyInstrs));
  (void)newLoopBodyInstrs.emplace_back(MakeCallFunInstrPtr(stack_effect + localsForInput.size()));
  if (!localsForOutput.empty()) {
    if (localsForOutput.size() > 1) {
      (void)newLoopBodyInstrs.emplace_back(std::make_unique<Instr>(UNPACK_SEQUENCE, localsForOutput.size()));
    }
    for (auto iter = localsForOutput.begin(); iter != localsForOutput.end(); ++iter) {
      (void)newLoopBodyInstrs.emplace_back(std::make_unique<Instr>(STORE_FAST, *iter));
    }
  }
  std::vector<std::unique_ptr<Instr>> resultInstrs = CodeGenerator::CopyAndReplaceInstr(
    graph_->GetCFG()->instr_pool(), loopBodyStartBci_, loopBodyEndBci_, newLoopBodyInstrs);
  if (IsPiJitLogOn(LogCfg::kOthers)) {
    std::stringstream ss;
    ss << "Instrs After ReCapture:" << std::endl;
    for (auto &instr : resultInstrs) {
      ss << instr->ToString() << std::endl;
    }
    MS_LOG(INFO) << ss.str();
  }
  CodeGenerator::Code ccode = {
    co.ArgCount(),
    co.ArgCount() - co.PositionOnlyArgCount(),
    co.FastLocalSize(),
    co_->co_flags,
    resultInstrs.front()->line(),
    std::move(resultInstrs),
    py::cast<std::vector<std::string>>(PyCodeWrapper(co_).VarNames()),
    std::vector<std::string>(),
    closures,
    makeFuncName(loopBodyStartBci_, loopBodyEndBci_),
    py::reinterpret_borrow<py::object>(co_->co_filename),
  };
  py::object result = CodeGenerator(std::move(ccode)).NewCode();
  return result;
}

void LocationTableEncoder::WriteItemStart(int code, int len) {
  const int flag = 1 << 7;
  const int len_width = 3;
  WriteByte(flag | (code << len_width) | (len - 1));
}

void LocationTableEncoder::WriteVariant(uint32_t value) {
  const int kValidBits = 6;
  const int kValueMask = (1 << kValidBits) - 1;
  while (value >= (1 << kValidBits)) {
    WriteByte((1 << kValidBits) | (value & kValueMask));
    value >>= kValidBits;
  }
  WriteByte(value);
}

void LocationTableEncoder::WriteSignedVariant(int32_t value) {
  uint32_t ui = static_cast<uint32_t>(value);
  ui = value < 0 ? (((0 - ui) << 1) | 1) : (ui << 1);
  WriteVariant(ui);
}

void LocationTableEncoder::NoneEntry(int offset) { WriteItemStart(kCodeLocationNone, offset); }

void LocationTableEncoder::NoColumn(int offset, int line_delta) {
  WriteItemStart(kCodeLocationNoColumns, offset);
  WriteSignedVariant(line_delta);
}

void LocationTableEncoder::OneLineEntry(int offset, int line_delta, int column, int end_column) {
  WriteItemStart(kCodeLocationOneLine0 + line_delta, offset);
  WriteByte(column);
  WriteByte(end_column);
}

void LocationTableEncoder::ShortEntry(int offset, int column, int end_column) {
  const int kValidBits = 3;
  const int kValueMask = (1 << kValidBits) - 1;
  int column_value = column & kValueMask;
  int column_group = column >> kValidBits;
  WriteItemStart(kCodeLocationShort0 + column_group, offset);
  WriteByte((column_value << (kValidBits + 1)) | (end_column - column));
}

void LocationTableEncoder::LongEntry(int offset, const CodeLocation &loc) {
  WriteItemStart(kCodeLocationLong, offset);
  WriteSignedVariant(loc.start_line_ - pre_line_);
  WriteVariant(loc.end_line_ - loc.start_line_);
  WriteVariant(loc.start_column_ + 1);
  WriteVariant(loc.end_column_ + 1);
}

void LocationTableEncoder::AddEntry(int offset, const CodeLocation &loc) {
  if (loc.start_line_ < 0) {
    NoneEntry(offset);
    return;
  }
  const int kShortMaxColValue = 80;
  const int kShortMaxColDelta = 16;
  const int kMaxColValue = 128;
  const int kOneLineGroup = kCodeLocationOneLine2 - kCodeLocationOneLine0 + 1;

  int line_delta = loc.start_line_ - pre_line_;
  int col = loc.start_column_;
  int end_col = loc.end_column_;
  if (col < 0 || end_col < 0) {
    if (loc.end_line_ == loc.start_line_ || loc.end_line_ == -1) {
      NoColumn(offset, line_delta);
      pre_line_ = loc.start_line_;
      return;
    }
  } else if (loc.end_line_ == loc.start_line_) {
    if (line_delta == 0 && col < kShortMaxColValue && end_col - col < kShortMaxColDelta && end_col >= col) {
      ShortEntry(offset, col, end_col);
      return;
    }
    if (line_delta >= 0 && line_delta < kOneLineGroup && col < kMaxColValue && end_col < kMaxColValue) {
      OneLineEntry(offset, line_delta, col, end_col);
      pre_line_ = loc.start_line_;
      return;
    }
  }
  LongEntry(offset, loc);
  pre_line_ = loc.start_line_;
}

void LocationTableEncoder::Add(int bci_delta, const CodeLocation &loc) {
  if (bci_delta == 0) {
    return;
  }
  const int entry_size = 8;
  while (bci_delta > entry_size) {
    AddEntry(entry_size, loc);
    bci_delta -= entry_size;
  }
  AddEntry(bci_delta, loc);
}

void ExceptionTableEncoder::WriteVariant(int value, int mask_bits) {
  const int kValidBits = 6;
  const int kContinuationBit = 1 << kValidBits;
  const int kValueMask = kContinuationBit - 1;
  const int kStart = 24;
  if (value >= (1 << kStart)) {
    WriteByte((value >> kStart) | kContinuationBit | mask_bits);
    mask_bits = 0;
  }
  for (int left = kStart - kValidBits; left; left -= kValidBits) {
    if (value >= (1 << left)) {
      WriteByte(((value >> left) & kValueMask) | kContinuationBit | mask_bits);
      mask_bits = 0;
    }
  }
  WriteByte((value & kValueMask) | mask_bits);
}

void ExceptionTableEncoder::WriteItem(const ExceptionTableItem &item) {
  const int kFlagBit = 7;
  WriteVariant(item.begin_, (1 << kFlagBit));
  WriteVariant(item.end_ - item.begin_, 0);
  WriteVariant(item.jump_, 0);
  WriteVariant((item.stack_ << 1) | (item.lasti_ ? 1 : 0), 0);
}

}  // namespace pijit
}  // namespace mindspore
