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
#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_DEVICE_CONTEXT_CONF_OP_PRECISION_CONF_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_DEVICE_CONTEXT_CONF_OP_PRECISION_CONF_H_

#include <memory>
#include <string>

#include "plugin/ascend/res_manager/visible.h"
#include "utils/ms_utils.h"
#include "pybind11/pybind11.h"

namespace py = pybind11;

namespace mindspore {
namespace device {
namespace ascend {
class ASCEND_RES_MANAGER_EXPORT OpPrecisionConf {
 public:
  OpPrecisionConf() = default;
  ~OpPrecisionConf() = default;
  OpPrecisionConf(const OpPrecisionConf &) = delete;
  OpPrecisionConf &operator=(const OpPrecisionConf &) = delete;
  static std::shared_ptr<OpPrecisionConf> GetInstance();
  void set_precision_mode(const std::string &value) { precision_mode_ = value; }
  std::string precision_mode() const;
  void set_op_precision_mode(const std::string &value) { op_precision_mode_ = value; }
  std::string op_precision_mode() const;
  void set_matmul_allow_hf32(const std::string &value) { matmul_allow_hf32_ = value; }
  std::string matmul_allow_hf32() const;
  void set_conv_allow_hf32(const std::string &value) { conv_allow_hf32_ = value; }
  std::string conv_allow_hf32() const;
  bool IsPrecisionModeConfigured() const { return !precision_mode_.empty(); }
  bool IsOpPrecisionModeConfigured() const { return !op_precision_mode_.empty(); }
  bool IsMatmulAllowHf32Configured() const { return !matmul_allow_hf32_.empty(); }
  bool IsConvAllowHf32Configured() const { return !conv_allow_hf32_.empty(); }

 private:
  static std::shared_ptr<OpPrecisionConf> inst_context_;
  std::string precision_mode_{""};
  std::string op_precision_mode_{""};
  std::string matmul_allow_hf32_{""};
  std::string conv_allow_hf32_{""};
};

ASCEND_RES_MANAGER_EXPORT void RegOpPrecisionConf(py::module *m);
}  // namespace ascend
}  // namespace device
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_DEVICE_CONTEXT_CONF_OP_PRECISION_CONF_H_
