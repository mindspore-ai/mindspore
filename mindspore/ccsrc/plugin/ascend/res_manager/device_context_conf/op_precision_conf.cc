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
#include "plugin/ascend/res_manager/device_context_conf/op_precision_conf.h"
#include "utils/log_adapter.h"
#include "utils/ms_context.h"

namespace mindspore {
namespace device {
namespace ascend {
std::shared_ptr<OpPrecisionConf> OpPrecisionConf::inst_context_ = nullptr;

std::shared_ptr<OpPrecisionConf> OpPrecisionConf::GetInstance() {
  static std::once_flag inst_context_init_flag_ = {};
  std::call_once(inst_context_init_flag_, [&]() {
    if (inst_context_ == nullptr) {
      MS_LOG(DEBUG) << "Create new mindspore OpPrecisionConf";
      inst_context_ = std::make_shared<OpPrecisionConf>();
    }
  });
  MS_EXCEPTION_IF_NULL(inst_context_);
  return inst_context_;
}

std::string OpPrecisionConf::precision_mode() const {
  if (!precision_mode_.empty()) {
    return precision_mode_;
  }
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  auto precision_mode = ms_context->get_param<std::string>(MS_CTX_PRECISION_MODE);
  return precision_mode;
}

std::string OpPrecisionConf::op_precision_mode() const {
  if (!op_precision_mode_.empty()) {
    return op_precision_mode_;
  }
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  auto op_precision_mode = ms_context->get_param<std::string>(MS_CTX_OP_PRECISION_MODE);
  return op_precision_mode;
}

std::string OpPrecisionConf::matmul_allow_hf32() const {
  if (!matmul_allow_hf32_.empty()) {
    return matmul_allow_hf32_;
  }
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  auto allow_matmul_hf32 = ms_context->get_param<std::string>(MS_CTX_MATMUL_ALLOW_HF32);
  return allow_matmul_hf32;
}

std::string OpPrecisionConf::conv_allow_hf32() const {
  if (!conv_allow_hf32_.empty()) {
    return conv_allow_hf32_;
  }
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  auto conv_allow_hf32 = ms_context->get_param<std::string>(MS_CTX_CONV_ALLOW_HF32);
  return conv_allow_hf32;
}

std::string OpPrecisionConf::cube_math_type() const { return cube_math_type_; }

void RegOpPrecisionConf(py::module *m) {
  (void)py::class_<OpPrecisionConf, std::shared_ptr<OpPrecisionConf>>(*m, "AscendOpPrecisionConf")
    .def_static("get_instance", &OpPrecisionConf::GetInstance, "Get OpPrecisionConf instance.")
    .def("set_precision_mode", &OpPrecisionConf::set_precision_mode, "Set Precision Mode.")
    .def("precision_mode", &OpPrecisionConf::precision_mode, "Get Precision Mode.")
    .def("set_op_precision_mode", &OpPrecisionConf::set_op_precision_mode, "Set Op Precision Mode.")
    .def("op_precision_mode", &OpPrecisionConf::op_precision_mode, "Get Op Precision Mode.")
    .def("set_matmul_allow_hf32", &OpPrecisionConf::set_matmul_allow_hf32, "Matmul Op allow fp32 to hf32.")
    .def("matmul_allow_hf32", &OpPrecisionConf::matmul_allow_hf32, "Get Matmul Op allow fp32 to hf32 Conf.")
    .def("set_conv_allow_hf32", &OpPrecisionConf::set_conv_allow_hf32, "Conv Op allow fp32 tp hf32.")
    .def("conv_allow_hf32", &OpPrecisionConf::conv_allow_hf32, "Get Conv Op allow fp32 tp hf32 Conf.")
    .def("set_cube_math_type", &OpPrecisionConf::set_cube_math_type, "Set Cube Math Type.")
    .def("cube_math_type", &OpPrecisionConf::cube_math_type, "Get Cube Math Type.")
    .def("is_precision_mode_configured", &OpPrecisionConf::IsPrecisionModeConfigured, "Is Precision Mode Configured.")
    .def("is_op_precision_mode_configured", &OpPrecisionConf::IsOpPrecisionModeConfigured,
         "Is Op Precision Mode Configured.")
    .def("is_matmul_allow_hf32_configured", &OpPrecisionConf::IsMatmulAllowHf32Configured,
         "Is Matmul Allow Hf32 Configured.")
    .def("is_conv_allow_hf32_configured", &OpPrecisionConf::IsConvAllowHf32Configured, "Is Conv Allow Hf32 Configured.")
    .def("is_cube_math_type_configured", &OpPrecisionConf::IsCubeMathTypeConfigured, "Is Cube Math Type Configured.");
}
}  // namespace ascend
}  // namespace device
}  // namespace mindspore
