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
#include <unordered_map>
#include "runtime/dev.h"
#include "acl/error_codes/rt_error_codes.h"
#include "transform/acl_ir/acl_helper.h"
#include "include/backend/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "include/common/utils/utils.h"

namespace mindspore::transform {
namespace {
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

uint8_t KeepOriginDType() {
  static std::string version = "";
  static uint8_t need_keep_dtype = 0;
  if (version.empty()) {
    const int kSocVersionLen = 50;
    char soc_version[kSocVersionLen] = {0};
    auto ret = rtGetSocVersion(soc_version, kSocVersionLen);
    if (ret != RT_ERROR_NONE) {
      MS_LOG(EXCEPTION) << "GetSocVersion failed.";
    }
    version = soc_version;
    if (version.find(k910BKey) != std::string::npos || version.find(k310BKey) != std::string::npos ||
        version.find(k910CKey) != std::string::npos) {
      need_keep_dtype = 1;
    }
  }
  return need_keep_dtype;
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
  uint8_t select_mode = (static_cast<uint8_t>(use_hf32) << 1) + KeepOriginDType();
  if (kSelectMoreMathType.count(select_mode) != 0) {
    return kSelectMoreMathType.at(select_mode);
  }
  return KeepOriginDType() ? KEEP_DTYPE : ALLOW_FP32_DOWN_PRECISION;
}

void OpApiUtil::GetValidKernelBuildInfo(const AnfNodePtr &node, std::vector<std::string> *input_formats,
                                        std::vector<std::string> *output_formats,
                                        std::vector<std::string> *input_reshape_types,
                                        std::vector<std::string> *output_reshape_types) {
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
}  // namespace mindspore::transform
