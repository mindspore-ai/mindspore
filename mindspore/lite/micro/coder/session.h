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
#ifndef MINDSPORE_LITE_MICRO_CODER_SESSION_H_
#define MINDSPORE_LITE_MICRO_CODER_SESSION_H_

#include <map>
#include <string>
#include <vector>
#include <memory>
#include "schema/inner/model_generated.h"
#include "coder/graph.h"
#include "coder/context.h"
#include "coder/config.h"
#include "coder/allocator/allocator.h"
#include "coder/opcoders/op_coder.h"

namespace mindspore::lite::micro {
class CoderSession {
 public:
  CoderSession();

  ~CoderSession();

  int Init(const std::string &model_path);

  int Build();

  int Run();

  int GenerateCode();

 private:
  int InitOpcodersInputsAndOutputs();
  int InitTensorsRef();
  int CreateOpCoders();
  int InitCodeGraph();
  int CompileGraph();
  int InferShape();
  void EndCode();

  std::unique_ptr<CoderGraph> coder_graph_{nullptr};
  std::unique_ptr<CoderContext> context_{nullptr};
  MemoryAllocator *allocator_{nullptr};
  std::vector<std::unique_ptr<OperatorCoder>> op_coders_;
};

std::shared_ptr<CoderSession> CreateCoderSession();

}  // namespace mindspore::lite::micro
#endif  // MINDSPORE_LITE_MICRO_CODER_SESSION_H_
