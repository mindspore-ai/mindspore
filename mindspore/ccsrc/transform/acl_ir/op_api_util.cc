/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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
#include "transform/acl_ir/op_api_util.h"
#include <dlfcn.h>
#include <unordered_map>
#include <unordered_set>
#include "acl/error_codes/rt_error_codes.h"
#include "transform/acl_ir/acl_helper.h"
#include "include/backend/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "include/common/utils/utils.h"
#include "ops/math_op_name.h"
#include "ops/nn_op_name.h"
#include "mindspore/core/ops/array_ops.h"
#include "utils/ms_context.h"
#include "transform/symbol/acl_base_symbol.h"
#include "transform/symbol/acl_compiler_symbol.h"
#include "transform/symbol/symbol_utils.h"
#include "plugin/device/ascend/kernel/internal/internal_kernel_build.h"

namespace mindspore::transform {
namespace {
typedef aclError (*AclrtCtxSetSysParamOpt)(aclSysParamOpt, int64_t);
typedef HcclResult (*HcclSetConfigFunc)(HcclConfig, HcclConfigValue);

static const char k910BKey[] = "Ascend910B";
static const char k310BKey[] = "Ascend310B";
static const char k910CKey[] = "Ascend910C";

static const std::unordered_map<std::string, aclCubeMathType> kCubeMathType = {
  {"force_fp16", FORCE_FP16},
  {"allow_fp32_to_fp16", ALLOW_FP32_DOWN_PRECISION},
  {"allow_mix_precision", ALLOW_FP32_DOWN_PRECISION},
  {"must_keep_origin_dtype", KEEP_DTYPE},
  {"allow_fp32_to_bf16", ALLOW_FP32_DOWN_PRECISION},
  {"allow_mix_precision_fp16", ALLOW_FP32_DOWN_PRECISION},
  {"allow_mix_precision_bf16", ALLOW_FP32_DOWN_PRECISION}};

static const std::unordered_map<uint8_t, aclCubeMathType> kSelectMoreMathType = {
  {0b01, KEEP_DTYPE}, {0b00, FORCE_FP16}, {0b11, FORCE_HF32}, {0b10, ALLOW_FP32_DOWN_PRECISION}};

std::mutex set_opt_mutex;

aclError SetCompileopt(aclCompileOpt opt, const char *value) { return CALL_ASCEND_API(aclSetCompileopt, opt, value); }

void *GetAclFunc(const std::string &lib_path, const std::string &func_name) {
  static auto ascend_path = mindspore::transform::GetAscendPath();
  auto load_path = ascend_path + "/lib64/" + lib_path;

  auto handler = dlopen(load_path.c_str(), RTLD_LAZY);
  if (handler == nullptr) {
    MS_LOG(INFO) << "Dlopen " << load_path << " failed!" << dlerror();
    return nullptr;
  }

  auto func = dlsym(handler, func_name.c_str());
  if (func == nullptr) {
    MS_LOG(INFO) << "Dlsym " << func_name << " from " << load_path << " failed!" << dlerror();
  }
  return func;
}
}  // namespace

aclCubeMathType OpApiUtil::GetCubeMathType(bool use_hf32) {
  static std::string precision_mode = "not_inited";
  if (precision_mode == "not_inited") {
    auto ms_context = MsContext::GetInstance();
    MS_EXCEPTION_IF_NULL(ms_context);
    precision_mode = ms_context->get_param<std::string>(MS_CTX_PRECISION_MODE);
  }

  if (!precision_mode.empty() && kCubeMathType.count(precision_mode) != 0) {
    return kCubeMathType.at(precision_mode);
  }
  uint8_t select_mode = (static_cast<uint8_t>(use_hf32) << 1) + AclUtil::KeepOriginDType();
  if (kSelectMoreMathType.count(select_mode) != 0) {
    return kSelectMoreMathType.at(select_mode);
  }
  return AclUtil::KeepOriginDType() ? KEEP_DTYPE : ALLOW_FP32_DOWN_PRECISION;
}

void OpApiUtil::GetValidKernelBuildInfo(const AnfNodePtr &node, std::vector<std::string> *input_formats,
                                        std::vector<std::string> *output_formats,
                                        std::vector<std::string> *input_reshape_types,
                                        std::vector<std::string> *output_reshape_types, const KernelType &kernel_type) {
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(input_formats);
  MS_EXCEPTION_IF_NULL(output_formats);
  MS_EXCEPTION_IF_NULL(input_reshape_types);
  MS_EXCEPTION_IF_NULL(output_reshape_types);

  input_formats->clear();
  output_formats->clear();
  input_reshape_types->clear();
  output_reshape_types->clear();

  size_t input_num = common::AnfAlgo::GetInputTensorNum(node);
  size_t output_num = AnfUtils::GetOutputTensorNum(node);

  input_formats->assign(input_num, kOpFormat_DEFAULT);
  output_formats->assign(output_num, kOpFormat_DEFAULT);

  input_reshape_types->assign(input_num, "");
  output_reshape_types->assign(output_num, "");

  if (kernel_type == INTERNAL_KERNEL || IsOneOfPrimitiveCNode(node, {prim::kPrimReshapeExt, prim::kPrimReshape})) {
    kernel::GetValidKernelBuildInfoWithInternalFormat(node, input_formats, output_formats);
    return;
  }

  std::vector<size_t> special_inputs;
  for (size_t i = 0; i < input_num; ++i) {
    auto kernel_with_index = common::AnfAlgo::GetPrevNodeOutput(node, i);
    std::string input_format = AnfAlgo::GetOutputFormat(kernel_with_index.first, kernel_with_index.second);
    if (!AclHelper::CheckDefaultSupportFormat(input_format)) {
      (void)special_inputs.emplace_back(i);
    }
  }
  if (!special_inputs.empty()) {
    common::AnfAlgo::SetNodeAttr(kAttrAclSpecialInputFormat, MakeValue(special_inputs), node);
  }
}

uint8_t AclUtil::KeepOriginDType() {
  static std::string version = "";
  static uint8_t need_keep_dtype = 0;
  if (version.empty()) {
    const char *soc_name_c = CALL_ASCEND_API(aclrtGetSocName);
    if (soc_name_c != nullptr) {
      version = soc_name_c;
    }
    if (version.find(k910BKey) != std::string::npos || version.find(k310BKey) != std::string::npos ||
        version.find(k910CKey) != std::string::npos) {
      need_keep_dtype = 1;
    }
  }
  return need_keep_dtype;
}

void AclUtil::SetDeterministic() {
  std::lock_guard<std::mutex> lock(set_opt_mutex);
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  bool is_deterministic = ms_context->get_param<std::string>(MS_CTX_DETERMINISTIC) == "ON" ? true : false;
  // Set acl
  auto ret = SetCompileopt(aclCompileOpt::ACL_OP_DETERMINISTIC, is_deterministic ? "1" : "0");
  if (ret != ACL_SUCCESS) {
    MS_LOG(EXCEPTION) << "Acl set deterministic mode failed! mode is " << is_deterministic << " and error flag is "
                      << ret;
  }
  // Set acl sys
  const std::string rt_sys_opt_lib = "libacl_op_compiler.so";
  const std::string rt_sys_opt_name = "aclrtCtxSetSysParamOpt";
  auto rt_sys_opt = GetAclFunc(rt_sys_opt_lib, rt_sys_opt_name);
  if (rt_sys_opt == nullptr) {
    MS_LOG(EXCEPTION) << "Get 'aclrtCtxSetSysParamOpt' from " << rt_sys_opt_lib << " failed!";
  }
  auto rt_sys_opt_func = reinterpret_cast<AclrtCtxSetSysParamOpt>(rt_sys_opt);
  ret = rt_sys_opt_func(aclSysParamOpt::ACL_OPT_DETERMINISTIC, is_deterministic ? 1 : 0);
  if (ret != ACL_SUCCESS) {
    MS_LOG(EXCEPTION) << "Acl sys set deterministic mode failed! mode is " << is_deterministic << " and error flag is "
                      << ret;
  }
  // Set hccl
  const std::string hccl_lib = "libhccl.so";
  const std::string hccl_set_config_name = "HcclSetConfig";
  auto hccl_set_config = GetAclFunc(hccl_lib, hccl_set_config_name);
  if (hccl_set_config == nullptr) {
    MS_LOG(EXCEPTION) << "Get 'HcclSetConfig' from " << hccl_lib << " failed!";
  }
  auto hccl_set_config_func = reinterpret_cast<HcclSetConfigFunc>(hccl_set_config);
  HcclConfigValue config = {is_deterministic ? 1 : 0};
  auto hccl_ret = hccl_set_config_func(HcclConfig::HCCL_DETERMINISTIC, config);
  if (hccl_ret != HCCL_SUCCESS) {
    MS_LOG(EXCEPTION) << "Hccl set deterministic mode failed! mode is " << is_deterministic << " and error flag is "
                      << ret;
  }
}

aclError AclUtil::SetCompileMode(const int64_t is_dynamic) {
  std::lock_guard<std::mutex> lock(set_opt_mutex);
  static int64_t last_mode = -1;
  if (is_dynamic != last_mode) {
    std::string mode = is_dynamic ? "disable" : "enable";
    auto set_compile_flag = SetCompileopt(aclCompileOpt::ACL_OP_JIT_COMPILE, mode.c_str());
    last_mode = is_dynamic;
    return set_compile_flag;
  }

  return ACL_SUCCESS;
}

aclError AclUtil::SetPrecisionMode(const std::string &mode) {
  std::lock_guard<std::mutex> lock(set_opt_mutex);

  static int8_t is_global_precision = -1;
  if (is_global_precision == -1) {
    auto ms_context = MsContext::GetInstance();
    MS_EXCEPTION_IF_NULL(ms_context);
    auto precision_mode = ms_context->get_param<std::string>(MS_CTX_PRECISION_MODE);
    if (!precision_mode.empty()) {
      is_global_precision = 1;
    } else {
      is_global_precision = 0;
    }
  }
  if (is_global_precision == 1) {
    return ACL_SUCCESS;
  }

  static std::string last_mode = (AclUtil::KeepOriginDType() == 1) ? "must_keep_origin_dtype" : "allow_fp32_to_fp16";
  if (last_mode != mode) {
    auto ret = SetCompileopt(aclCompileOpt::ACL_PRECISION_MODE, mode.c_str());
    last_mode = mode;
    return ret;
  }
  return ACL_SUCCESS;
}

void AclUtil::SetOpPrecisionMode() {
  std::lock_guard<std::mutex> lock(set_opt_mutex);
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  auto op_precision_mode = ms_context->get_param<std::string>(MS_CTX_OP_PRECISION_MODE);
  if (op_precision_mode.empty()) {
    return;
  }
  MS_LOG(DEBUG) << "Set ACL_OP_PRECISION_MODE: " << op_precision_mode;
  auto ret = SetCompileopt(aclCompileOpt::ACL_OP_PRECISION_MODE, op_precision_mode.c_str());
  if (ret != ACL_SUCCESS) {
    MS_LOG(EXCEPTION) << "Acl set op precision mode failed! error flag is " << ret;
  }
}
}  // namespace mindspore::transform
