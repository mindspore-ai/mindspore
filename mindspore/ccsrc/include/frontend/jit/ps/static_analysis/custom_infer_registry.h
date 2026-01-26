/**
 * Copyright 2026 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_INCLUDE_FRONTEND_JIT_PS_STATIC_ANALYSIS_CUSTOM_INFER_REGISTRY_H_
#define MINDSPORE_CCSRC_INCLUDE_FRONTEND_JIT_PS_STATIC_ANALYSIS_CUSTOM_INFER_REGISTRY_H_

#include <unordered_map>
#include <string>
#include <memory>
#include <utility>

#include "pybind11/pybind11.h"
#include "pybind11/functional.h"
#include "utils/log_adapter.h"
#include "include/utils/visible.h"

namespace py = pybind11;

namespace mindspore {
/// \brief Information structure for operation inference functions.
/// struct OpInferInfo {
///     std::vector<std::string> func_names;     ///< Names of inference functions
///     std::vector<py::function> funcs;         ///< Actual function objects
/// };
struct OpInferInfo {
  std::vector<std::string> func_names;
  std::vector<py::function> funcs;
};

/// \brief Global registry for custom inference functions.
///
/// This class manages the registration and lookup of custom inference functions
/// used by operators.
class FRONTEND_EXPORT OpInferRegistry {
 public:
  static std::shared_ptr<OpInferRegistry> GetInstance() {
    static auto custom_infer_registry = std::make_shared<OpInferRegistry>();
    return custom_infer_registry;
  }
  ~OpInferRegistry() = default;
  OpInferRegistry() = default;

  void RegisterFunction(const py::object &op_name, const py::object &func_name, py::function func);
  std::vector<py::function> GetFunctions(const std::string &op_name) const;
  std::vector<std::string> GetFunctionNames(const std::string &op_name) const;
  void Clear();

 private:
  std::unordered_map<std::string, OpInferInfo> op_infer_map_;
};

class FRONTEND_EXPORT OpWrapperRegistry {
 public:
  static std::shared_ptr<OpWrapperRegistry> GetInstance() {
    static auto custom_infer_registry = std::make_shared<OpWrapperRegistry>();
    return custom_infer_registry;
  }
  ~OpWrapperRegistry() = default;
  OpWrapperRegistry() = default;

  void RegisterFunction(const py::object &op_name, py::function func);
  py::function GetFunction(const std::string &op_name) const;
  bool inferring_wrapped_func() const { return inferring_wrapped_func_; }
  void set_inferring_wrapped_func(bool inferring_flag) { inferring_wrapped_func_ = inferring_flag; }
  bool type_inference_flag() const { return type_inference_flag_; }
  void set_type_inference_flag(bool type_inference_flag) { type_inference_flag_ = type_inference_flag; }
  void Clear();

 private:
  std::unordered_map<std::string, py::function> func_map_;
  bool inferring_wrapped_func_ = false;
  bool type_inference_flag_ = false;
};

}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_INCLUDE_FRONTEND_JIT_PS_STATIC_ANALYSIS_CUSTOM_INFER_REGISTRY_H_
