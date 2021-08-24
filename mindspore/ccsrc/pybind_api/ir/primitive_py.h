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

#ifndef MINDSPORE_CCSRC_UTILS_PRIMITIVE_PY_H_
#define MINDSPORE_CCSRC_UTILS_PRIMITIVE_PY_H_

#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "abstract/abstract_value.h"
#include "frontend/parallel/ops_info/operator_info.h"
#include "ir/primitive.h"
#include "ir/signature.h"
#include "pybind11/pybind11.h"
#include "utils/log_adapter.h"
#include "utils/misc.h"

namespace py = pybind11;
namespace mindspore {

class PrimitivePy;
using PrimitivePyPtr = std::shared_ptr<PrimitivePy>;
using PrimitivePyWeakPtr = std::weak_ptr<PrimitivePy>;

class PrimitivePyAdapter;
using PrimitivePyAdapterPtr = std::shared_ptr<PrimitivePyAdapter>;

class PrimitivePy : public Primitive {
 public:
  explicit PrimitivePy(const std::string &name);
  PrimitivePy(const py::object &python_obj, const PrimitivePyAdapterPtr &adapter);
  ~PrimitivePy() override;
  MS_DECLARE_PARENT(PrimitivePy, Primitive);
  py::function GetBpropFunction();

  void set_signatures(const std::vector<Signature> &signatures);

  const std::vector<Signature> &signatures() const { return signatures_; }

  void CopyHookFunction(const PrimitivePtr &primitive) override;

  py::dict GetAttrDict();
  void set_hook(const py::function &hook) { hook_ = hook; }
  py::function hook() const { return hook_; }
  BaseRef RunHookFunction(const VectorRef &args) const override;
  BaseRef RunCellBpropFunction(const py::tuple &py_args) const;
  BaseRef RunCellHookFunction(const py::tuple &py_args) const;
  BaseRef RunVariableHookFunction(const py::tuple &py_args) const;
  BaseRef RunComputeFunction(const VectorRef &args) const override;
  py::object RunPyComputeFunction(const py::tuple &py_args) const;
  bool HasComputeFunction() const;
  const bool parse_info_ = true;
  const py::object &GetPyObj() const { return python_obj_; }
  py::dict RunInfer(const py::tuple &args);
  void RunCheck(const py::tuple &args);
  py::object RunInferValue(const py::tuple &args);
  bool HasPyObj() { return python_obj_.operator bool(); }
  PrimitivePtr Clone() override;
  PrimitivePyAdapterPtr adapter() const { return adapter_; }

 private:
  py::function GetComputeFunction() const;
  void ConvertCTensorToPyTensor(const py::tuple &input_args, py::tuple *convert_args) const;
  void CheckHookConsistency(const py::object &grad_out, const py::object &expected_grad_out) const;
  py::object python_obj_;
  PrimitivePyAdapterPtr adapter_;
  py::function hook_;
  std::vector<Signature> signatures_;
  static std::map<std::string, py::object> hook_grad_;
};

class PrimitivePyAdapter {
 public:
  explicit PrimitivePyAdapter(const py::str &name);
  ~PrimitivePyAdapter() = default;
  void AddPyAttr(const py::str &name, const py::object &obj);
  void DelPyAttr(const py::str &name);
  py::dict GetAttrDict();
  void set_prim_type(const PrimType t);
  void set_const_prim(bool is_const_prim);
  void set_const_input_indexes(const std::vector<size_t> &const_input_indexes);
  void set_signatures(const std::vector<Signature> &signatures);
  void set_hook(const py::function &hook);
  void set_instance_name(const std::string &s);
  void set_attached_primitive(const PrimitivePyPtr &prim);
  PrimitivePyPtr attached_primitive() { return attached_primitive_.lock(); }
  std::string name() const { return name_; }
  void set_name(const std::string &name) { name_ = name; }
  const bool parse_info_ = true;

 private:
  friend PrimitivePy;
  std::string name_;
  PrimitivePyWeakPtr attached_primitive_;
  std::unordered_map<std::string, ValuePtr> attrs_;
  PrimType prim_type_{kPrimTypeBuiltIn};
  bool is_const_prim_{false};
  std::vector<size_t> const_input_indexes_;
  std::vector<Signature> signatures_;
  py::function hook_;
  std::string instance_name_;
};
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_UTILS_PRIMITIVE_PY_H_
