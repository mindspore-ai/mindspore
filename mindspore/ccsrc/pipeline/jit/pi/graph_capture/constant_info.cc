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
#include <functional>
#include "pipeline/jit/pi/pydef.h"
#include "pipeline/jit/pi/graph_capture/node.h"
#include "pipeline/jit/pi/graph_capture/graph.h"

namespace mindspore {
namespace pijit {

constexpr const char kModuleName[] = "mindspore";
constexpr const char kTensorShapeName[] = "shape";
constexpr const char kTensorDtypeName[] = "dtype";

static void MakePrimitiveConstantInfoCommon(ValueNode *node);

void ConstantInfo::set_value(const py::object &op) {
  value_ = op;
  if (op.ptr() == nullptr) {
    return;
  }
  set_type(Py_TYPE(op.ptr()));
  if (type() == &PyTuple_Type) {
    set_len(PyTuple_GET_SIZE(op.ptr()));
  }
  if (type() == &PyList_Type) {
    set_len(PyList_GET_SIZE(op.ptr()));
  }
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
  for (const auto &i : attrs_) {
    s << i.first << "=" << std::string(py::str(i.second.ptr())) << ", ";
  }
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

static void MakeDimInfoOfTensor(ValueNode *node) {
  const auto &cnst = node->GetConstantInfo();
  if (cnst == nullptr) {
    return;
  }
  node->SetConstantValue(cnst->HasAttr(kTensorShapeName));
}

static void MakeConstantInfoOfTensorAttr(ValueNode *node) {
  const std::string &name = node->GetName();
  if (name == kTensorShapeName) {
    MakeShapeInfoOfTensor(node);
  }
  if (name == "ndim") {
    MakeDimInfoOfTensor(node);
  }
}

bool CheckConstantAttr(ValueNode *node) {
  const auto &src_cnst_info = node->input(0)->GetConstantInfo();
  const std::string &name = node->GetName();
  if (src_cnst_info != nullptr && src_cnst_info->HasAttr(name)) {
    node->MakeConstantInfo()->set_value(src_cnst_info->GetAttr(name));
  }

  if (node->GetVobj() == nullptr || node->input(0)->GetVobj() == nullptr) {
    return false;
  }
  AObject *src_info = node->input(0)->GetVobj();
  if (src_info->GetType() == AObject::kTypeTensor) {
    MakeConstantInfoOfTensorAttr(node);
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
    // if type not equal, IS_OP always False
    return l_cnst_info->type() != r_cnst_info->type();
  }
  return false;
}

bool MakeConstantBinary(ValueNode *node) {
  AObject *res_info = node->GetVobj();
  if (res_info == nullptr) {
    return false;
  }
  AObject::Type type = res_info->GetType();
  if (type != AObject::kTypeTensor) {
    return false;
  }
  const auto &l_cnst = node->input(0)->GetConstantInfo();
  if (l_cnst == nullptr) {
    return false;
  }
  if (l_cnst->type() != nullptr) {
    MakePrimitiveConstantInfoCommon(node);
  }
  return false;
}

bool MakeConstantBinarySubscr(ValueNode *node) {
  const auto &r_cnst = node->input(1)->GetConstantInfo();
  if (r_cnst == nullptr || r_cnst->type() == nullptr) {
    return false;
  }
  ValueNode *map_node = node->input(0);
  if (map_node->GetOpcode() == LOAD_ATTR) {
    ValueNode *src_node = map_node->input(0);
    bool is_shape = src_node->GetVobj()->GetType() == AObject::kTypeTensor && map_node->GetName() == kTensorShapeName;
    if (is_shape && r_cnst->type() == &PyLong_Type) {
      node->MakeConstantInfo()->set_type(&PyLong_Type);
      return false;
    }
  }
  const auto &l_cnst = node->input(0)->GetConstantInfo();
  if (l_cnst == nullptr || l_cnst->type() == nullptr) {
    return false;
  }
  if (r_cnst->type() == &PySlice_Type) {
    if (l_cnst->type() == &PyTuple_Type || l_cnst->type() == &PyList_Type) {
      node->MakeConstantInfo()->set_type(l_cnst->type());
      return false;
    }
  }
  return MakeConstantBinary(node);
}

static void MakeSpecializeConstantValue(ValueNode *node) {
  if (node->IsConstantValue()) {
    return;
  }
  if (Utils::IsBinaryMathOp(node->GetOpcode())) {
    MakeConstantBinary(node);
  }
  static const std::map<int, bool (*)(ValueNode *)> specialize = {
    {LOAD_ATTR, CheckConstantAttr},   {LOAD_GLOBAL, CheckConstantGlobal},
    {IS_OP, CheckConstantIs},         {BINARY_SUBSCR, MakeConstantBinarySubscr},
    {COMPARE_OP, MakeConstantBinary},
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
  node->MakeConstantInfo()->SetAttr(kTensorShapeName, py::tuple());
}

void MakeConstantInfoOfPrimCast(ValueNode *node) {
  ValueNode *dtype = node->input(2);
  if (dtype->IsConstantValue()) {
    node->MakeConstantInfo()->SetAttr(kTensorDtypeName, dtype->GetConstantInfo()->value());
  }
}

void MakeConstantInfoOfPrimIsShapeUnKnown(ValueNode *node) {
  // primitive IsShapeUnKnown only accept tuple and list, pynative mode it's always False
  node->SetVobj(AObject::Convert(Py_False));
  node->SetConstantValue(true);
}

static void MakeConvertToMsTensorInfo(ValueNode *node) {
  const auto &cnst = node->input(1)->GetConstantInfo();
  if (cnst == nullptr) {
    return;
  }
  *node->MakeConstantInfo() = *cnst;
}

static void MakeReshapeInfo(ValueNode *node) {
  const auto &shape_cnst = node->input(2)->GetConstantInfo();
  if (shape_cnst == nullptr || shape_cnst->value().ptr() == nullptr) {
    return;
  }
  const auto &cnst_info = node->input(1)->GetConstantInfo();

  PyObject *out_shape = shape_cnst->value().ptr();
  PyObject **begin = PyList_Check(out_shape) ? &PyList_GET_ITEM(out_shape, 0) : &PyTuple_GET_ITEM(out_shape, 0);
  PyObject **end = begin + (PyList_Check(out_shape) ? PyList_GET_SIZE(out_shape) : PyTuple_GET_SIZE(out_shape));
  bool is_dynamic_shape = std::any_of(begin, end, [](PyObject *op) { return PyLong_AsLong(op) == -1; });
  bool is_constant_shape = !is_dynamic_shape || (cnst_info != nullptr && cnst_info->HasAttr(kTensorShapeName));
  if (is_constant_shape) {
    py::object cnst_shape = node->GetVobj()->GetPyObject().attr(kTensorShapeName);
    node->MakeConstantInfo()->SetAttr(kTensorShapeName, cnst_shape);
  }
}

static const std::map<std::string, void (*)(ValueNode *)> &GetConstantPrimitiveMap() {
  static const std::map<std::string, void (*)(ValueNode *)> cnst_prim = {
    {"ScalarToTensor", MakeConstantInfoOfPrimScalarToTensor},
    {"Cast", MakeConstantInfoOfPrimCast},
    {"IsShapeUnKnown", MakeConstantInfoOfPrimIsShapeUnKnown},
    {"Shape", MakeShapeInfoOfTensor},
    {"Reshape", MakeReshapeInfo},
    {"ConvertToMsTensor", MakeConvertToMsTensorInfo},
    {"ConvertToAdapterTensor", MakeConvertToMsTensorInfo},
  };
  return cnst_prim;
}

static void MakePrimitiveConstantInfoCommon(ValueNode *node) {
  AObject *info = node->GetVobj();
  if (info == nullptr) {
    return;
  }
  // assume primitive return type is always constant !!!
  const auto &cnst = node->MakeConstantInfo();
  cnst->set_type(info->GetTypeObject());

  if (info->GetType() != AObject::kTypeTensor) {
    return;
  }
  // check all inputs tensor shape is constant, other inputs is constant
  const auto &inputs = node->getInputs();
  bool constant_shape = std::none_of(inputs.begin(), inputs.end(), [](ValueNode *i) {
    const auto &cnst = i->GetConstantInfo();
    if (cnst == nullptr) {
      return true;
    }
    if (i->GetVobj()->GetType() == AObject::kTypeTensor) {
      return !cnst->HasAttr(kTensorShapeName);
    }
    return cnst->value().ptr() != nullptr;
  });
  if (constant_shape) {
    cnst->SetAttr(kTensorShapeName, info->GetPyObject().attr(kTensorShapeName));
  }
}

void ConstantInfo::CollectPrimitiveConstantInfo(CallNode *node) {
  MS_EXCEPTION_IF_CHECK_FAIL(node->input(0)->GetVobj()->GetType() == AObject::kTypePrimitive, "must be primitive");
  MakePrimitiveConstantInfoCommon(node);

  std::string prim_key = node->input(0)->GetVobj()->GetPyObject().attr("name").cast<std::string>();
  auto iter = GetConstantPrimitiveMap().find(prim_key);
  if (iter == GetConstantPrimitiveMap().end()) {
    return;
  }
  iter->second(node);
}

static bool CheckConstantLen(ValueNode *node) {
  const auto &cnst = node->input(1)->GetConstantInfo();
  if (cnst == nullptr || cnst->len() == -1) {
    return false;
  }
  PyObject *len = node->GetVobj() ? node->GetVobj()->GetPyObject().ptr() : nullptr;
  if (len != nullptr) {
    MS_EXCEPTION_IF_CHECK_FAIL(cnst->len() == PyLong_AsSsize_t(len), "error constant len");
  } else {
    node->SetVobj(AObject::Convert(py::int_(cnst->len()).ptr()));
  }
  return true;
}

static bool CheckConstantInstanceCheck(ValueNode *node) {
  const auto &c1 = node->input(1)->GetConstantInfo();
  bool cnst = c1 != nullptr && c1->type() != nullptr;
  constexpr int second_arg = 2;
  return cnst && node->input(second_arg)->IsConstantValue();
}

static const std::map<PyCFunction, bool (*)(ValueNode *)> &GetConstantBuiltinFuncMap() {
  using Handler = bool (*)(ValueNode *);
  static std::map<PyCFunction, Handler> cnst_func = {};
  static auto func_map_init = [](const char *func_name, Handler handler) {
    auto func = PyDict_GetItemString(PyEval_GetBuiltins(), func_name);
    auto cfunc = PyCFunction_GET_FUNCTION(func);
    cnst_func.insert({cfunc, handler});
  };
  if (!cnst_func.empty()) {
    return cnst_func;
  }
  func_map_init("len", CheckConstantLen);
  func_map_init("isinstance", CheckConstantInstanceCheck);
  return cnst_func;
}

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
