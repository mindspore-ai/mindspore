/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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

#ifndef MS_CUSTOM_OPS_MODULE_H_
#define MS_CUSTOM_OPS_MODULE_H_

#include "ms_extension/api.h"
#include "kernel/ascend/custom/kernel_mod_impl/custom_kernel_factory.h"
#include <functional>
#include <pybind11/pybind11.h>
#include <string>
#include <vector>

// Define the type of module registration functions
using ModuleRegisterFunction = std::function<void(pybind11::module_ &)>;

// Module registry class
class ModuleRegistry {
 public:
  // Get the singleton instance
  static ModuleRegistry &Instance() {
    static ModuleRegistry instance;
    return instance;
  }

  // Register a module function
  void Register(ModuleRegisterFunction func, bool pynative_node = true) {
    auto &target = pynative_node ? pynative_reg_functions_ : graph_reg_functions_;
    target.emplace_back(std::move(func));
  }

  // Call all registered module functions
  void RegisterAll(pybind11::module_ &m) {
    for (const auto &func : pynative_reg_functions_) {
      func(m);
    }
    for (const auto &func : graph_reg_functions_) {
      func(m);
    }
  }

 private:
  ModuleRegistry() = default;
  ~ModuleRegistry() = default;

  // Disable copy and assignment
  ModuleRegistry(const ModuleRegistry &) = delete;
  ModuleRegistry &operator=(const ModuleRegistry &) = delete;

  // Store all registered functions
  std::vector<ModuleRegisterFunction> pynative_reg_functions_;
  std::vector<ModuleRegisterFunction> graph_reg_functions_;
};

#define REG_GRAPH_MODE_OP(op, OpFuncImplClass, KernelClass)                                            \
  MS_CUSTOM_OPS_REGISTER(op, OpFuncImplClass, KernelClass);                                            \
  static void op##_func(pybind11::args args, pybind11::kwargs kwargs) {                                \
    MS_LOG(EXCEPTION) << "Custom [" << #op                                                             \
                      << "] does not support PyNative mode. Please implement PyboostRunner to enable " \
                         "PyNative mode execution, or switch to Graph mode.";                          \
  }                                                                                                    \
  static void op##_register(pybind11::module_ &m) {                                                    \
    if (!pybind11::hasattr(m, #op)) {                                                                  \
      m.def(#op, &op##_func);                                                                          \
    }                                                                                                  \
  }                                                                                                    \
  struct op##_registrar {                                                                              \
    op##_registrar() { ModuleRegistry::Instance().Register(op##_register, false); }                    \
  };                                                                                                   \
  static op##_registrar registrar_instance

#define CONCATENATE_DETAIL(x, y) x##y
#define CONCATENATE(x, y) CONCATENATE_DETAIL(x, y)

#define MS_CUSTOM_OPS_EXTENSION_MODULE(m)                                                   \
  static void CONCATENATE(func_register_, __LINE__)(pybind11::module_ &);                   \
  namespace {                                                                               \
  struct CONCATENATE(func_registrar_, __LINE__) {                                           \
    CONCATENATE(func_registrar_, __LINE__)() {                                              \
      ModuleRegistry::Instance().Register(CONCATENATE(func_register_, __LINE__));           \
    }                                                                                       \
  };                                                                                        \
  static CONCATENATE(func_registrar_, __LINE__) CONCATENATE(registrar_instance_, __LINE__); \
  }                                                                                         \
  static void CONCATENATE(func_register_, __LINE__)(pybind11::module_ & m)

#endif  // MS_CUSTOM_OPS_MODULE_H_
