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

#ifndef MINDSPORE_LITE_SRC_EXTENDRT_POST_PROCESS_MODEL_MANAGER_H_
#define MINDSPORE_LITE_SRC_EXTENDRT_POST_PROCESS_MODEL_MANAGER_H_

#include <functional>
#include <set>
#include <string>
#include <memory>
#include <utility>
#include "src/extendrt/cxx_api/llm_engine/ms_llm_config.h"
#include "extendrt/cxx_api/model/model_impl.h"

namespace mindspore {
namespace llm {
class PostProcessModelManager {
 public:
  PostProcessModelManager() {}
  ~PostProcessModelManager();

  PostProcessModelManager(const PostProcessModelManager &) = delete;
  PostProcessModelManager &operator=(const PostProcessModelManager &) = delete;

  static PostProcessModelManager &GetInstance() {
    static PostProcessModelManager instance;
    return instance;
  }

  std::shared_ptr<mindspore::ModelImpl> GetModel(GenerateParameters *param);

 private:
  std::shared_ptr<mindspore::ModelImpl> CreateArgMaxModel(GenerateParameters *param);
  std::shared_ptr<mindspore::ModelImpl> CreateTopKTopPModel(GenerateParameters *param);

  FuncGraphPtr CreateArgMaxFuncGraph();
  FuncGraphPtr CreatTopKTopPFuncGraph();

  std::shared_ptr<mindspore::Context> CreateContext();

  std::shared_ptr<mindspore::ModelImpl> argmax_model_;
  std::shared_ptr<mindspore::ModelImpl> topk_and_topp_model_;
};
}  // namespace llm
}  // namespace mindspore

#endif  // MINDSPORE_LITE_SRC_EXTENDRT_POST_PROCESS_MODEL_MANAGER_H_
