/**
 * Copyright 2019-2025 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_INCLUDE_FRONTEND_OPERATOR_PRIMITIVE_PY_H_
#define MINDSPORE_CCSRC_INCLUDE_FRONTEND_OPERATOR_PRIMITIVE_PY_H_

#include <map>
#include <string>
#include <utility>
#include <vector>
#include <memory>
#include <unordered_map>

#include "utils/hash_map.h"
#include "utils/ordered_map.h"
#include "ir/primitive.h"
#include "ir/signature.h"
#include "pybind11/pybind11.h"
#include "include/utils/convert_utils_py.h"

namespace py = pybind11;
namespace mindspore {

class PrimitivePy;
using PrimitivePyPtr = std::shared_ptr<PrimitivePy>;
using PrimitivePyWeakPtr = std::weak_ptr<PrimitivePy>;

class PrimitivePyAdapter;
using PrimitivePyAdapterPtr = std::shared_ptr<PrimitivePyAdapter>;

class PrimitiveFunctionAdapter;
using PrimitiveFunctionAdapterPtr = std::shared_ptr<PrimitiveFunctionAdapter>;

// For hook type
enum class HookType {
  // Custom op bprop
  kCustomOpBprop = 0,
  // Cell custom bprop
  kCellCustomBprop,
  // HookBackward op
  kHookBackwardOp,
  // TensorHook
  kTensorHook,
  // Backward pre hook
  kBackwardPreHook,
  // Backward hook
  kBackwardHook,
  // Default
  kUnknown,
};

class FRONTEND_EXPORT PrimitivePy : public Primitive {
 public:
  explicit PrimitivePy(const std::string &name);
  PrimitivePy(const PrimitivePy &prim_py);
  PrimitivePy &operator=(const PrimitivePy &other);
  explicit PrimitivePy(const py::object &python_obj);
  ~PrimitivePy() override;
  MS_DECLARE_PARENT(PrimitivePy, Primitive);
  const bool parse_info_ = true;
  py::function GetVmapRuleFunction(const bool is_side_effect = false, int axis_size = 0);
  py::function GetBpropFunction();
  py::function GetTaylorRuleFunction();
  HookType hook_type() { return hook_type_; }
  std::string HookTypeToString() const;
  const py::function &hook_fn() const { return hook_fn_; }
  void CopyHookFunction(const PrimitivePyPtr &primitive_py);
  void AddBpropCutPrim(const PrimitivePyPtr &bprop_cut_prim);
  void SetHookFn(const py::function &hook_fn, HookType hook_type);
  BaseRef RunComputeFunction(const VectorRef &args) const override;
  py::object RunPyComputeFunction(const py::tuple &py_args) const;
  bool HasComputeFunction() const;
  py::dict GetAttrDict();
  const py::object &GetPyObj() const { return python_obj_; }
  bool HasPyObj() const { return python_obj_.operator bool(); }
  void RunCheck(const py::tuple &args);
  py::dict RunInfer(const py::tuple &args);
  py::object RunInferValue(const py::tuple &args);
  PrimitivePtr Clone() override;
  PrimitivePyAdapterPtr adapter() const { return adapter_; }
  void set_bprop_cls_name(const std::string &name) { bprop_cls_name_ = name; }
  const std::string &bprop_cls_name() const { return bprop_cls_name_; }
  static void ProcessUnPairedCellHook(bool execute_hook_fn);
  static void ClearHookRes();
  bool IsPythonPrim() override { return true; }
  py::object UnpackRetValueOfCellHook(const py::object &grad_out) const;
  void CheckHookConsistency(const py::object &grad_out, const py::object &expected_grad_out,
                            const py::object &co_name) const;
  void EmplaceUnpairBackwardHookGrad(const std::string &ret, const py::function &hook_fn) {
    unpair_backward_hook_grad_.emplace(ret, hook_fn);
  }

  void EraseUnpairBackwardHookGrad(const std::string &name) { unpair_backward_hook_grad_.erase(name); }

  void ClearUnpairBackwardHookGrad() { unpair_backward_hook_grad_.clear(); }

 private:
  py::function GetComputeFunction() const;
  py::object python_obj_;
  std::string bprop_cls_name_;
  PrimitivePyAdapterPtr adapter_;
  std::vector<Signature> signatures_;
  std::vector<PrimitivePyWeakPtr> bprop_cut_prims_;
  HookType hook_type_{HookType::kUnknown};
  py::function hook_fn_;
  // If a cell registers a backward hook, but the inputs of the cell does not calculate the derivative, and the
  // parameters in the cell need to calculate the derivative, then the hook function will not be executed
  static mindspore::OrderedMap<std::string, py::function> unpair_backward_hook_grad_;
};

class FRONTEND_EXPORT PrimitivePyAdapter {
 public:
  explicit PrimitivePyAdapter(const py::str &name);
  PrimitivePyAdapter(const PrimitivePyAdapter &adapter);
  PrimitivePyAdapter &operator=(const PrimitivePyAdapter &other);
  ~PrimitivePyAdapter() {
    // cppcheck-suppress unreadVariable
    py::gil_scoped_acquire acquire_gil;
    hook_fn_ = py::object();
  }
  const mindspore::HashMap<std::string, ValuePtr> &attrs() const { return attrs_; }
  void AddPyAttr(const py::str &name, const py::object &obj);
  void DelPyAttr(const py::str &name);
  py::dict GetAttrDict();
  void SetHookFn(const py::function &hook_fn, HookType hook_type);
  void set_prim_type(const PrimType t);
  void set_const_prim(bool is_const_prim);
  void set_inplace_prim(bool inplace_prim);
  void set_const_input_indexes(const std::vector<size_t> &const_input_indexes);
  void set_signatures(const std::vector<Signature> &signatures);
  void set_instance_name(const std::string &s);
  void set_attached_primitive(const PrimitivePyPtr &prim);
  PrimitivePyPtr attached_primitive() const { return attached_primitive_.lock(); }
  uint64_t id() const { return id_; }
  std::string name() const { return name_; }
  void set_name(const std::string &name) { name_ = name; }

  struct PrimitiveUserData {
    py::object obj;
    ~PrimitiveUserData() {
      // cppcheck-suppress unreadVariable
      py::gil_scoped_acquire acquire_gil;
      obj = py::object();
    }
  };

  void SetUserData(const py::str &key, const py::object &value);
  py::object GetUserData(const py::str &key) const;

  const bool parse_info_ = true;

 private:
  friend PrimitivePy;

  template <typename T>
  void set_user_data(const std::string &key, const std::shared_ptr<T> &value) {
    user_data_.set<T>(key, value);
  }
  template <typename T>
  std::shared_ptr<T> user_data(const std::string &key) const {
    return user_data_.get<T>(key);
  }

  bool const_prim_{false};
  bool inplace_prim_{false};
  uint64_t id_;
  std::string name_;
  std::string instance_name_;
  PrimType prim_type_{kPrimTypeBuiltIn};
  PrimitivePyWeakPtr attached_primitive_;
  mindspore::HashMap<std::string, ValuePtr> attrs_;
  std::vector<size_t> const_input_indexes_;
  std::vector<Signature> signatures_;
  HookType hook_type_{HookType::kUnknown};
  py::function hook_fn_;
  UserData user_data_;
};

class FRONTEND_EXPORT PrimitiveFunctionAdapter {
 public:
  PrimitiveFunctionAdapter() = default;
  virtual ~PrimitiveFunctionAdapter() = default;
  void set_attached_primitive_function(const PrimitivePtr &prim_func) { attached_primitive_function_ = prim_func; }
  PrimitivePtr attached_primitive_function() { return attached_primitive_function_; }
  virtual std::string name() { return py::str(attached_primitive_function_->name()).cast<std::string>(); }
  py::object has_label(const std::string &label) { return py::bool_(attached_primitive_function_->HasAttr(label)); }
  void set_label(const std::string &label, const py::object &value);
  py::object get_label(const std::string &label) { return ValueToPyData(attached_primitive_function_->GetAttr(label)); }
  py::object clone();

  const bool parse_info_ = true;

 private:
  std::string name_;
  PrimitivePtr attached_primitive_function_;
};
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_INCLUDE_FRONTEND_OPERATOR_PRIMITIVE_PY_H_
