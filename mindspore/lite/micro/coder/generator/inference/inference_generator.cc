/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "coder/generator/inference/inference_generator.h"
#include <vector>
#include <string>
#include "coder/generator/component/common_component.h"
#include "coder/generator/component/parallel_component.h"
#include "coder/generator/component/weight_component.h"
#include "coder/generator/component/const_blocks/license.h"
#include "coder/generator/component/component.h"

namespace mindspore::lite::micro {
int InferenceGenerator::CodeNetHFile() {
  std::string net_include_file;
  net_include_file = net_src_file_path_ + net_inc_hfile_;
  std::ofstream ofs(net_include_file);
  MS_CHECK_TRUE(!ofs.bad(), "filed to open file");
  MS_LOG(INFO) << "write " << net_include_file;
  ofs << g_hwLicense;
  if (config_->support_parallel()) {
    ofs << "#include \"thread_pool.h\"\n";
  }
  ofs << kExternCpp;
  CodeInputState(ofs);
  CodeCopyOutputsState(ofs);
  if (is_get_quant_args_) {
    CodeGraphQuantArgsState(ofs);
  }
  if (config_->support_parallel()) {
    CodeSetGlobalThreadPoolState(ofs);
  }
  if (config_->target() != kARM32M) {
    CodeInitWeightState(ofs);
  }
  CodeManageResourceState(ofs);
  CodeInferenceState(ofs);
  ofs << kEndExternCpp;
  return RET_OK;
}

int InferenceGenerator::CodeNetCFile() {
  std::string net_impl_file = net_src_file_path_ + net_src_cfile_;
  std::ofstream ofs(net_impl_file);
  MS_CHECK_TRUE(!ofs.bad(), "filed to open file");
  MS_LOG(INFO) << "write " << net_impl_file;
  ofs << g_hwLicense << "\n"
      << "#include \"" << net_weight_hfile_ << "\"\n"
      << "#include \"" << net_inc_hfile_ << "\"\n\n";
  if (config_->debug_mode()) {
    ofs << "#include \"" << kDebugUtils << "\"\n";
  }
  if (config_->support_parallel()) {
    CodeSetGlobalThreadPoolImplement(ofs);
  }
  CodeInputImplement(ofs, ctx_);
  CodeCopyOutputsImplement(ofs, ctx_);
  CodeInitResourceImplement(ofs, ctx_);
  CodeFreeResourceImplement(ofs, ctx_);
  if (is_get_quant_args_) {
    CodeGraphQuantArgsImplement(ofs, ctx_);
  }
  CodeNetRunFunc(ofs);
  ofs.close();
  return RET_OK;
}
}  // namespace mindspore::lite::micro
