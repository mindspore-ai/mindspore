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
class PrimitivePy : public Primitive {
 public:
  PrimitivePy(const py::str &name, const py::object &python_obj);
  ~PrimitivePy() override;
  MS_DECLARE_PARENT(PrimitivePy, Primitive);
  py::function GetBpropFunction();

  void set_signatures(const std::vector<Signature> &signatures);

  const std::vector<Signature> &signatures() const { return signatures_; }

  void CopyHookFunction(const PrimitivePtr &primitive) override;

  void AddPyAttr(const py::str &name, const py::object &obj);

  void DelPyAttr(const py::str &name);

  py::dict GetAttrDict();
  void set_hook(const py::function &hook) { hook_ = hook; }
  py::function hook() const { return hook_; }
  BaseRef RunHookFunction(const VectorRef &args) const override;
  BaseRef RunBpropHookFunction(const py::tuple &py_args) const;
  BaseRef RunComputeFunction(const VectorRef &args) const override;
  py::object RunPyComputeFunction(const py::tuple &py_args) const;
  bool HasComputeFunction() const;
  const bool parse_info_ = true;
  const py::object &GetPyObj() const { return python_obj_; }
  void SetPyObj(const py::object &obj);
  py::dict RunInfer(const py::tuple &args);
  void RunCheck(const py::tuple &args);
  py::object RunInferValue(const py::tuple &args);
  bool ObjHasAttr(const char *attr_name) { return py::hasattr(python_obj_, attr_name); }
  bool HasPyObj() { return python_obj_.operator bool(); }
  PrimitivePtr Clone() override;
  bool is_tuple_input_ = false;

 private:
  py::function GetComputeFunction() const;
  void ConvertCTensorToPyTensor(const py::tuple &input_args, py::tuple *convert_args) const;
  void CheckHookConsistency(const py::object &grad_out, const py::object &expected_grad_out) const;
  py::object python_obj_;
  py::function hook_;
  std::vector<Signature> signatures_;
  static std::map<std::string, py::object> hook_grad_;
};

using PrimitivePyPtr = std::shared_ptr<PrimitivePy>;
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_UTILS_PRIMITIVE_PY_H_
