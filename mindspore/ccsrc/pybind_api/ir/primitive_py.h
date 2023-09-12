/**
 * Copyright 2019-2021 Huawei Technologies Co., Ltd
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
#include <string>
#include <vector>
#include <memory>
#include <utility>

#include "utils/hash_map.h"
#include "abstract/abstract_value.h"
#include "ir/primitive.h"
#include "ir/signature.h"
#include "pybind11/pybind11.h"

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
  PrimitivePy(const PrimitivePy &prim_py);
  PrimitivePy &operator=(const PrimitivePy &other);
  explicit PrimitivePy(const py::object &python_obj);
  ~PrimitivePy() override;
  MS_DECLARE_PARENT(PrimitivePy, Primitive);
  const bool parse_info_ = true;
  py::function GetVmapRuleFunction(const bool is_side_effect = false, int axis_size = 0);
  py::function GetBpropFunction();
  py::function GetTaylorRuleFunction();
  void set_signatures(const std::vector<Signature> &signatures);
  const std::vector<Signature> &signatures() const { return signatures_; }
  const std::map<int, py::function> &backward_hook_fn() const { return backward_hook_fn_; }
  void CopyHookFunction(const PrimitivePyPtr &primitive_py);
  void AddBpropCutPrim(const PrimitivePyPtr &bprop_cut_prim);
  void AddBackwardHookFn(const int &key, const py::function &backward_hook_fn);
  void RemoveBackwardHookFn(const int &key);
  BaseRef RunHookFunction(const VectorRef &args) const;
  BaseRef RunCellBpropFunction(const py::tuple &py_args) const;
  BaseRef RunOpBpropFunction(const py::tuple &py_args) const;
  BaseRef RunCellHookFunction(const py::tuple &py_args) const;
  BaseRef RunVariableHookFunction(const py::tuple &py_args) const;
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
  static void ProcessUnPairedCellHook(bool execute_hook_fn);
  static void ClearHookRes();

 private:
  py::function GetComputeFunction() const;
  py::object UnpackRetValueOfCellHook(const py::object &grad_out) const;
  void CheckHookConsistency(const py::object &grad_out, const py::object &expected_grad_out, const py::object &code_obj,
                            const py::object &co_name) const;
  py::object python_obj_;
  std::string bprop_cls_name_;
  PrimitivePyAdapterPtr adapter_;
  std::vector<Signature> signatures_;
  std::vector<PrimitivePyWeakPtr> bprop_cut_prims_;
  std::map<int, py::function> backward_hook_fn_;
  static std::map<std::string, std::pair<std::map<int, py::function>, py::object>> hook_grad_;
};

class PrimitivePyAdapter {
 public:
  explicit PrimitivePyAdapter(const py::str &name);
  PrimitivePyAdapter(const PrimitivePyAdapter &adapter);
  PrimitivePyAdapter &operator=(const PrimitivePyAdapter &other);
  ~PrimitivePyAdapter() = default;
  void AddPyAttr(const py::str &name, const py::object &obj);
  void DelPyAttr(const py::str &name);
  py::dict GetAttrDict();
  int AddBackwardHookFn(const py::function &backward_hook_fn);
  void RemoveBackwardHookFn(int key);
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
      obj = py::none();
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
  int backward_hook_fn_key_{-1};
  uint64_t id_;
  std::string name_;
  std::string instance_name_;
  PrimType prim_type_{kPrimTypeBuiltIn};
  PrimitivePyWeakPtr attached_primitive_;
  mindspore::HashMap<std::string, ValuePtr> attrs_;
  std::vector<size_t> const_input_indexes_;
  std::vector<Signature> signatures_;
  std::map<int, py::function> backward_hook_fn_;
  UserData user_data_;
};

/// \brief OpPrimPyRegister defines the singleton to save primitivepy which has no attrs.
class OpPrimPyRegister {
 public:
  /// \brief Destructor of OpPrimPyRegister.
  ~OpPrimPyRegister() {}

  /// \brief Get the OpPrimPyRegister singleton.
  ///
  /// \return The OpPrimPyRegister singleton.
  static OpPrimPyRegister &GetInstance() {
    static OpPrimPyRegister instance{};
    return instance;
  }

  /// \brief Get PrimPyMap of the OpPrimPyRegister singleton.
  ///
  /// \return The PrimPyMap of the OpPrimPyRegister singleton.
  const HashMap<std::string, ValuePtr> &GetPrimPyMap() const { return primpy_map_; }

  /// \brief Add an element into the PrimPyMap of the OpPrimPyRegister singleton.
  ///
  /// param[in] name The operator name.
  /// param[in] primpy The primitivepy of the operator.
  void SetPrimPyMap(const std::string &name, const ValuePtr &primpy) { primpy_map_[name] = primpy; }

  /// \brief Clear the PrimPyMap before the pyobject destroyed.
  void Clear() { primpy_map_.clear(); }

 private:
  OpPrimPyRegister() {}
  HashMap<std::string, ValuePtr> primpy_map_;  // op_name, primpy
};
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_UTILS_PRIMITIVE_PY_H_
