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

#ifndef MINDSPORE_LITE_TOOLS_CONVERTER_MICRO_CODER_TRAIN_TRAIN_GENERATOR_H_
#define MINDSPORE_LITE_TOOLS_CONVERTER_MICRO_CODER_TRAIN_TRAIN_GENERATOR_H_

#include <utility>
#include <memory>
#include <string>
#include <vector>
#include "tools/converter/micro/coder/generator/generator.h"

namespace mindspore::lite::micro {
class TrainGenerator : public Generator {
 public:
  TrainGenerator(std::unique_ptr<CoderContext> ctx, std::vector<std::pair<std::string, bool>> code_blocks_with_flag)
      : Generator(std::move(ctx)), code_blocks_with_flag_(std::move(code_blocks_with_flag)) {}
  ~TrainGenerator() override = default;

 private:
  void CodeTrainAndEvalFunc(std::ofstream &ofs);
  void CodeNetExecuteFunc(std::ofstream &ofs) override;
  int CodeNetHFile() override;
  int CodeNetCFile() override;

 private:
  std::vector<std::pair<std::string, bool>> code_blocks_with_flag_;  // <code block, is op only in train mode>
};
}  // namespace mindspore::lite::micro
#endif  // MINDSPORE_LITE_TOOLS_CONVERTER_MICRO_CODER_TRAIN_TRAIN_GENERATOR_H_
