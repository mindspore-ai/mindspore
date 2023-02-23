/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include "coder/train/train_generator.h"
#include <string>
#include "coder/generator/component/train_component.h"
#include "coder/opcoders/parallel.h"
#include "coder/generator/component/component.h"
#include "tools/common/string_util.h"

namespace mindspore::lite::micro {
void TrainGenerator::CodeTrainAndEvalFunc(std::ofstream &ofs) {
  size_t i = 0;
  size_t code_blocks_size = code_blocks_with_flag_.size();
  while (i < code_blocks_size) {
    bool is_train_only = code_blocks_with_flag_.at(i).second;
    if (!is_train_only) {
      ofs << "  {\n" << code_blocks_with_flag_.at(i).first << "  }\n";
      i++;
      continue;
    }

    size_t j = i;
    while (j < code_blocks_size && code_blocks_with_flag_.at(j).second) {  // is loss or grad op
      j++;
    }
    ofs << "  if (train_mode) {\n";
    for (; i < j; i++) {
      auto code_block = code_blocks_with_flag_.at(i).first;
      (void)FindAndReplaceAll(&code_block, "    ", "      ");
      ofs << "    {\n" << code_block << "    }\n";
    }
    ofs << "  }\n";
  }
}

void TrainGenerator::CodeNetExecuteFunc(std::ofstream &ofs) {
  ofs << "void Execute" << ctx_->GetCurModelIndex() << "(bool train_mode) {\n";
  if (config_->support_parallel()) {
    ofs << "  " << gThreadNum << " = GetCurrentThreadNum();\n";
    ofs << "  SetSpinCountMaxValue();\n";
  }

  CodeTrainAndEvalFunc(ofs);

  if (config_->support_parallel()) {
    ofs << "  SetSpinCountMinValue();\n";
  }

  ofs << "}\n";
}

int TrainGenerator::CodeNetHFile() {
  std::string net_include_file = model_dir_ + net_inc_hfile_;
  std::ofstream ofs(net_include_file);
  MS_CHECK_TRUE(!ofs.bad(), "filed to open file");
  MS_LOG(INFO) << "write " << net_include_file;
  CodeCommonNetH(ofs);
  CodeCopyTrainOutputsState(ofs, ctx_->GetCurModelIndex());
  ofs << kEndExternCpp;
  ofs.close();
  return RET_OK;
}

int TrainGenerator::CodeNetCFile() {
  std::string net_impl_file = net_src_file_path_ + net_src_cfile_;
  std::ofstream ofs(net_impl_file);
  MS_CHECK_TRUE(!ofs.bad(), "filed to open file");
  MS_LOG(INFO) << "write " << net_impl_file;
  CodeCommonNetC(ofs);
  CodeCopyTrainOutputsImplement(ofs, ctx_);
  CodeNetExecuteFunc(ofs);
  ofs.close();
  return RET_OK;
}
}  // namespace mindspore::lite::micro
