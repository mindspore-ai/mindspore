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
#include "pipeline/jit/pi/graph_capture/constant_info.h"
#include <set>
#include <vector>
#include "pipeline/jit/pi/pydef.h"
#include "pipeline/jit/pi/graph_capture/node.h"
#include "pipeline/jit/pi/graph_capture/graph.h"

namespace mindspore {
namespace pijit {

constexpr const char kModuleName[] = "mindspore";

void ConstantInfo::set_value(const py::object &op) {
  value_ = op;
  if (op.ptr() == nullptr) {
    return;
  }
  set_type(Py_TYPE(op.ptr()));
}

std::string ConstantInfo::ToString() const {
  std::stringstream s;
  if (type() != nullptr) {
    s << "type=" << (type()->tp_name ? type()->tp_name : "<unnamed>") << ", ";
  }
  if (value().ptr() != nullptr) {
    s << "value=" << std::string(py::str(value().ptr())) << ", ";
  }
  if (len() != -1) {
    s << "len=" << len() << ", ";
  }
  if (attrs_.empty()) {
    return s.str();
  }
  s << "attrs={ ";
  for (const auto &i : attrs_) {
    s << i.first << ":{" << i.second.ToString() << "}, ";
  }
  s << "}";
  return s.str();
}

bool IsConstantValue(int op, const std::vector<ValueNode *> &inputs) {
  static const std::set<int> support_constant_op = {
    BINARY_SUBSCR, COMPARE_OP, IS_OP,     CONTAINS_OP, LOAD_ATTR,           LIST_TO_TUPLE,
    BUILD_TUPLE,   BUILD_LIST, BUILD_MAP, BUILD_SLICE, BUILD_CONST_KEY_MAP,
  };
  if (op == LOAD_CONST) {
    return true;
  }
  auto iter = std::find_if_not(inputs.begin(), inputs.end(), [](ValueNode *i) { return i->IsConstantValue(); });
  if (iter != inputs.end()) {
    return false;
  }
  if (support_constant_op.find(op) != support_constant_op.end()) {
    return true;
  }
  if (Utils::IsBinaryMathOp(op) && Utils::IsGeneralNoSideEffectOp(op)) {
    return true;
  }
  return false;
}

static void MakeConstantFold(ValueNode *node) {
  node->SetConstantValue(IsConstantValue(node->GetOpcode(), node->getInputs()));
}

static void MakeCodeConstantInfo(ValueNode *node) {
  static const std::map<int, PyTypeObject *> constant_type = {
    {BUILD_TUPLE, &PyTuple_Type},    {BUILD_LIST, &PyList_Type},        {BUILD_SET, &PySet_Type},
    {BUILD_MAP, &PyDict_Type},       {BUILD_SLICE, &PySlice_Type},      {BUILD_CONST_KEY_MAP, &PyDict_Type},
    {BUILD_STRING, &PyUnicode_Type}, {LIST_TO_TUPLE, &PyTuple_Type},    {IS_OP, Py_TYPE(Py_True)},
    {CONTAINS_OP, Py_TYPE(Py_True)}, {MAKE_FUNCTION, &PyFunction_Type},
  };
  static const std::set<int> constant_len = {BUILD_TUPLE, BUILD_LIST, BUILD_SET, BUILD_MAP, BUILD_CONST_KEY_MAP};

  int opcode = node->GetOpcode();
  int oparg = node->GetOparg();
  PyTypeObject *tp = nullptr;
  Py_ssize_t len = -1;
  auto iter1 = constant_type.find(opcode);
  if (iter1 != constant_type.end()) {
    tp = iter1->second;
  }
  if (constant_len.find(opcode) != constant_len.end()) {
    len = oparg;
  }
  if (tp != nullptr || len != -1) {
    node->MakeConstantInfo()->set_type(tp);
    node->MakeConstantInfo()->set_len(len);
  }
}

static void MakeShapeInfoOfTensor(ValueNode *node) {
  // NOTE: MetaTensor shape is list, mindspore._c_expression.Tensor and mindspore.Tensor is tuple
  node->MakeConstantInfo()->set_type(&PyTuple_Type);
}

bool CheckConstantAttr(ValueNode *node) {
  const auto &src_cnst_info = node->input(0)->GetConstantInfo();
  const std::string &name = node->GetName();
  if (src_cnst_info != nullptr && src_cnst_info->HasAttr(name)) {
    node->MakeConstantInfo()->set_value(src_cnst_info->GetAttr(name)->value());
  }

  if (node->GetVobj() == nullptr || node->input(0)->GetVobj() == nullptr) {
    return false;
  }
  AObject *src_info = node->input(0)->GetVobj();
  if (src_info->GetType() == AObject::kTypeTensor) {
    if (name == "shape") {
      MakeShapeInfoOfTensor(node);
    }
    return false;
  }
  if (src_info->GetType() == AObject::kTypeModule && src_info->GetPyObject().ptr() != nullptr) {
    // mindspore module attribute
    const char *module_name = PyModule_GetName(src_info->GetPyObject().ptr());
    if (module_name == nullptr) {
      PyErr_Clear();
      return false;
    }
    return strncmp(module_name, kModuleName, sizeof(kModuleName) - 1) == 0;
  }
  return false;
}

bool CheckConstantGlobal(ValueNode *node) {
  const char *module_name = node->GetGraph()->GetModuleName();
  return strncmp(module_name, kModuleName, sizeof(kModuleName) - 1) == 0;
}

bool CheckConstantIs(ValueNode *node) {
  const auto &l_cnst_info = node->input(0)->GetConstantInfo();
  const auto &r_cnst_info = node->input(1)->GetConstantInfo();
  if (l_cnst_info == nullptr || r_cnst_info == nullptr) {
    return false;
  }
  if (l_cnst_info->type() != nullptr && r_cnst_info->type() != nullptr) {
    return true;
  }
  return false;
}

bool CheckConstantContains(ValueNode *node) {
  ValueNode *value = node->input(0);
  ValueNode *container = node->input(1);
  int container_op = container->GetOpcode();
  bool support = container_op == BUILD_LIST || container_op == BUILD_TUPLE || container_op == BUILD_MAP;
  if (!value->IsConstantValue() || !support) {
    return false;
  }

  PyObject *target = value->GetConstantInfo()->value().ptr();
  const auto IsTarget = [&target](ValueNode *i) {
    if (i->IsConstantValue()) {
      return false;
    }
    int res = PyObject_RichCompareBool(i->GetConstantInfo()->value().ptr(), target, Py_EQ);
    if (PyErr_Occurred()) {
      PyErr_Clear();
      return false;
    }
    return res > 0;
  };

  const auto &items = container->getInputs();
  if (container_op == BUILD_LIST || container_op == BUILD_TUPLE) {
    return std::any_of(items.begin(), items.end(), IsTarget);
  }
  if (container_op == BUILD_MAP) {
    for (size_t i = 0; i < items.size(); i += 2) {
      if (IsTarget(items[i])) {
        return true;
      }
    }
    return false;
  }
  return false;
}

static void MakeSpecializeConstantValue(ValueNode *node) {
  static const std::map<int, bool (*)(ValueNode *)> specialize = {
    {LOAD_ATTR, CheckConstantAttr},
    {LOAD_GLOBAL, CheckConstantGlobal},
    {IS_OP, CheckConstantIs},
    {CONTAINS_OP, CheckConstantContains},
  };
  auto iter = specialize.find(node->GetOpcode());
  if (iter == specialize.end()) {
    return;
  }
  if (!iter->second(node)) {
    return;
  }
  node->SetConstantValue(true);
}

void ConstantInfo::CollectConstantInfo(ValueNode *node) {
  MakeConstantFold(node);
  MakeCodeConstantInfo(node);
  MakeSpecializeConstantValue(node);
}

void MakeConstantInfoOfPrimScalarToTensor(ValueNode *node) {
  node->MakeConstantInfo()->GetAttr("shape")->set_value(py::tuple());
}

void MakeConstantInfoOfPrimCast(ValueNode *node) {
  ValueNode *dtype = node->input(2);
  if (dtype->IsConstantValue()) {
    node->MakeConstantInfo()->GetAttr("dtype")->set_value(dtype->GetConstantInfo()->value());
  }
}

void MakeConstantInfoOfPrimIsShapeUnKnown(ValueNode *node) {
  // primitive IsShapeUnKnown only accept tuple and list, pynative mode it's always False
  node->SetVobj(AObject::Convert(Py_False));
  node->SetConstantValue(true);
}

static const std::map<std::string, void (*)(ValueNode *)> &GetConstantPrimitiveMap() {
  static const std::map<std::string, void (*)(ValueNode *)> cnst_prim = {
    {"ScalarToTensor", MakeConstantInfoOfPrimScalarToTensor},
    {"Cast", MakeConstantInfoOfPrimCast},
    {"IsShapeUnKnown", MakeConstantInfoOfPrimIsShapeUnKnown},
    {"Shape", MakeShapeInfoOfTensor},
  };
  return cnst_prim;
}

void ConstantInfo::CollectPrimitiveConstantInfo(CallNode *node) {
  MS_EXCEPTION_IF_CHECK_FAIL(node->input(0)->GetVobj()->GetType() == AObject::kTypePrimitive, "must be primitive");
  AObject *info = node->GetVobj();
  if (info == nullptr) {
    return;
  }
  // assume primitive return type is always constant !!!
  const auto &cnst = node->MakeConstantInfo();
  cnst->set_type(info->GetTypeObject());

  std::string prim_key = node->input(0)->GetVobj()->GetPyObject().attr("name").cast<std::string>();
  auto iter = GetConstantPrimitiveMap().find(prim_key);
  if (iter == GetConstantPrimitiveMap().end()) {
    return;
  }
  iter->second(node);
}

static bool CheckConstantLen(ValueNode *node) {
  const auto &t = node->input(1)->GetConstantInfo();
  bool cnst = t != nullptr && t->len() != -1;
  PyObject *len = node->GetVobj()->GetPyObject().ptr();
  MS_EXCEPTION_IF_CHECK_FAIL(!cnst || t->len() == PyLong_AsSsize_t(len), "error constant len");
  return cnst;
}

static bool MakeConstantTypeCheck(ValueNode *node) {
  const auto &c1 = node->input(1)->GetConstantInfo();
  bool cnst = c1 != nullptr && c1->type() != nullptr;
  return cnst && node->GetGraph()->GuardValueNode(node->input(2));
}

#define DECLARE_BUILTIN_CFUNCTION(func_name, handler)           \
  func = PyDict_GetItemString(PyEval_GetBuiltins(), func_name); \
  cfunc = PyCFunction_GET_FUNCTION(func);                       \
  cnst_func.insert({cfunc, handler});

static const std::map<PyCFunction, bool (*)(ValueNode *)> &GetConstantBuiltinFuncMap() {
  static std::map<PyCFunction, bool (*)(ValueNode *)> cnst_func = {};
  if (!cnst_func.empty()) {
    return cnst_func;
  }
  PyObject *func;
  PyCFunction cfunc;
  DECLARE_BUILTIN_CFUNCTION("len", CheckConstantLen);
  DECLARE_BUILTIN_CFUNCTION("isinstance", MakeConstantTypeCheck);
  DECLARE_BUILTIN_CFUNCTION("issubclass", MakeConstantTypeCheck);
  return cnst_func;
}
#undef DECLARE_BUILTIN_CFUNCTION

void ConstantInfo::CollectBuiltinFuncConstantInfo(CallNode *node) {
  MS_EXCEPTION_IF_NULL(node->input(0)->GetVobj()->GetPyObject().ptr());
  PyObject *func = node->input(0)->GetVobj()->GetPyObject().ptr();
  if (PyMethod_Check(func)) {
    func = PyMethod_GET_FUNCTION(func);
  }
  MS_EXCEPTION_IF_CHECK_FAIL(PyCFunction_Check(func), "must be builtin function or method");
  PyCFunction cfunc = PyCFunction_GET_FUNCTION(func);

  auto iter = GetConstantBuiltinFuncMap().find(cfunc);
  if (iter == GetConstantBuiltinFuncMap().end()) {
    return;
  }
  if (iter->second(node)) {
    node->SetConstantValue(true);
  }
}

}  // namespace pijit
}  // namespace mindspore
