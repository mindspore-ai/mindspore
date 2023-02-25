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

#ifndef MINDSPORE_LITE_TOOLS_CONVERTER_MICRO_CODER_TRAIN_TRAIN_SESSION_H_
#define MINDSPORE_LITE_TOOLS_CONVERTER_MICRO_CODER_TRAIN_TRAIN_SESSION_H_

#include <map>
#include <utility>
#include <string>
#include <vector>
#include "tools/converter/micro/coder/session.h"
namespace mindspore::lite::micro {
class CoderTrainSession : public CoderSession {
 public:
  int Build() override;

  int Run(const std::string &model_name) override;

  int GenerateCode() override;

 private:
  int DoCode() override;
  int UpdateCodeBlocksWithFlag();
  int PassArgsToContext(const std::string &model_name) override;
  void FindEvalCoders(OperatorCoder *coder);
  int CompileTrainCoders();
  int CompileEvalCoders(const std::map<std::string, std::vector<Tensor *>> &eval_outputs_map);

 private:
  std::vector<std::pair<std::string, bool>> code_blocks_with_flag_;  // <code block, is op only in train mode>
  std::vector<OperatorCoder *> train_op_coders_;
  std::vector<OperatorCoder *> eval_op_coders_;
};
}  // namespace mindspore::lite::micro
#endif  // MINDSPORE_LITE_TOOLS_CONVERTER_MICRO_CODER_TRAIN_TRAIN_SESSION_H_
