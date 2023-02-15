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
#include <string>
#include "coder/generator/component/common_component.h"
#include "coder/generator/component/component.h"
#include "coder/opcoders/parallel.h"

namespace mindspore::lite::micro {
void InferenceGenerator::CodeNetExecuteFunc(std::ofstream &ofs) {
  ofs << "void Execute" << ctx_->GetCurModelIndex() << "(bool train_mode) {\n";
  if (config_->support_parallel()) {
    ofs << "  " << gThreadNum << " = GetCurrentThreadNum();\n";
    ofs << "  SetSpinCountMaxValue();\n";
  }
  for (const auto &block : ctx_->code_blocks()) {
    ofs << "  {\n" << block << "  }\n";
  }

  for (const auto &block : ctx_->after_inference_code_blocks()) {
    ofs << block << "\n";
  }
  if (config_->support_parallel()) {
    ofs << "  SetSpinCountMinValue();\n";
  }
  ofs << "}\n";
}

int InferenceGenerator::CodeNetHFile() {
  std::string net_include_file = net_src_file_path_ + net_inc_hfile_;
  std::ofstream ofs(net_include_file);
  MS_CHECK_TRUE(!ofs.bad(), "filed to open file");
  MS_LOG(INFO) << "write " << net_include_file;
  CodeCommonNetH(ofs);
  CodeCopyOutputsState(ofs, ctx_->GetCurModelIndex());
  ofs << kEndExternCpp;
  ofs.close();
  return RET_OK;
}

int InferenceGenerator::CodeNetCFile() {
  std::string net_impl_file = net_src_file_path_ + net_src_cfile_;
  std::ofstream ofs(net_impl_file);
  MS_CHECK_TRUE(!ofs.bad(), "filed to open file");
  MS_LOG(INFO) << "write " << net_impl_file;
  CodeCommonNetC(ofs);
  CodeCopyOutputsImplement(ofs, ctx_);
  CodeNetExecuteFunc(ofs);
  ofs.close();
  return RET_OK;
}
}  // namespace mindspore::lite::micro
