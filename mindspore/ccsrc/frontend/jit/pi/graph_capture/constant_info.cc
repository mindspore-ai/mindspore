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
#include "frontend/jit/pi/graph_capture/constant_info.h"
#include <set>
#include <vector>
#include <functional>
#include "include/utils/tensor_py.h"
#include "frontend/jit/pi/python_adapter/pydef.h"
#include "frontend/jit/pi/graph_capture/node.h"
#include "frontend/jit/pi/graph_capture/graph.h"

namespace mindspore {
namespace pijit {

constexpr const char kModuleName[] = "mindspore";
constexpr const char kTensorShapeName[] = "shape";
constexpr const char kTensorDtypeName[] = "dtype";
constexpr const size_t CmpSize = 2;

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

namespace {
bool IsDynamicShapeTensor(const py::object &obj) {
  if (obj.ptr() == nullptr || !tensor::IsTensorPy(obj)) {
    return false;
  }
  auto tensor = tensor::ConvertToTensor(obj);
  MS_EXCEPTION_IF_NULL(tensor);
  const ShapeVector &shape = tensor->shape();
  return std::any_of(shape.begin(), shape.end(), [](ShapeValueDType dim) { return dim < 0; });
}

std::string PyObjToString(const py::object &obj) {
  if (obj.ptr() == nullptr) {
    return "NULL";
  } else if (IsDynamicShapeTensor(obj)) {
    // Dynamic shape tensor to str, will raise a ValueError: negative dimensions are not allowed.
    return "<unknown>";
  }
  try {
    return std::string(py::str(obj));
  } catch (py::error_already_set &e) {
    MS_LOG(INFO) << "Failed to print python obj " << obj.ptr() << ". " << e.what();
    return "<ERROR DATA>";
  }
}
}  // namespace

std::string ConstantInfo::ToString() const {
  auto Limit = [](const std::string &s) {
    constexpr size_t limit = 120;
    auto str = s.size() < limit ? s : s.substr(0, limit) + "...";
    std::replace(str.begin(), str.end(), '\n', ' ');
    return str;
  };
  std::stringstream s;
  if (type() != nullptr) {
    s << "type=" << (type()->tp_name ? type()->tp_name : "<unnamed>") << ", ";
  }
  if (value().ptr() != nullptr) {
    s << "value=" << Limit(PyObjToString(value_)) << ", ";
  }
  if (len() != -1) {
    s << "len=" << len() << ", ";
  }
  for (const auto &i : attrs_) {
    s << i.first << "=" << Limit(PyObjToString(i.second)) << ", ";
  }
  return s.str();
}

bool IsConstantValue(int op, const std::vector<ValueNode *> &inputs) {
  static const std::set<int> support_constant_op = {
    BINARY_SUBSCR, COMPARE_OP, IS_OP,     CONTAINS_OP, LOAD_ATTR,           LIST_TO_TUPLE,
    BUILD_TUPLE,   BUILD_LIST, BUILD_MAP, BUILD_SLICE, BUILD_CONST_KEY_MAP,
  };
  Opcode code_info(op);
  if (code_info.HasConst()) {
    return true;
  }
  auto iter = std::find_if_not(inputs.begin(), inputs.end(), [](ValueNode *i) { return i->IsConstantValue(); });
  if (iter != inputs.end()) {
    return false;
  }
  if (support_constant_op.find(op) != support_constant_op.end()) {
    return true;
  }
  if (code_info.IsBinaryMath() && code_info.MayDelete()) {
    return true;
  }
  return false;
}

static void MakeConstantFold(ValueNode *node) {
  node->SetConstantValue(IsConstantValue(node->GetOpcode(), node->inputs()));
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
  if (Opcode(node->GetOpcode()).IsBinaryMath()) {
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

static void MakeSpecificConstantInfo(ValueNode *node) {
  if (!node) {
    return;
  }
  // os.environ
  if (node->GetOpcode() == LOAD_ATTR && node->input(0)->GetVobj() &&
      node->input(0)->GetVobj()->GetType() == AObject::kTypeModule && node->input(0)->GetVobj()->GetPyObject().ptr()) {
    auto module_obj = node->input(0)->GetVobj()->GetPyObject().ptr();
    const std::string &name = node->GetName();
    const char *module_name = PyModule_GetName(module_obj);
    if (module_name == nullptr) {
      PyErr_Clear();
      return;
    }
    if (strncmp(module_name, "os", CmpSize) == 0 && name == "environ") {
      auto env_obj = PyObject_GetAttrString(module_obj, "environ");
      node->SetConstantValue(true);
      node->MakeConstantInfo()->set_value(env_obj);
      node->SetOpcode(LOAD_CONST);
      node->SetOparg(-1);
      node->ClearInputs();
      return;
    }
  }
}

void ConstantInfo::CollectConstantInfo(ValueNode *node) {
  MakeConstantFold(node);
  MakeCodeConstantInfo(node);
  MakeSpecializeConstantValue(node);
  MakeSpecificConstantInfo(node);
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
  const auto &inputs = node->inputs();
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
  if (PyInstanceMethod_Check(func)) {
    func = PyInstanceMethod_GET_FUNCTION(func);
  }
  if (!PyCFunction_Check(func)) {
    return;
  }
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
