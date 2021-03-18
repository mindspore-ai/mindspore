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

#ifndef MINDSPORE_LITE_MICRO_CODER_GENERATOR_INFERENCE_GENERATOR_H_
#define MINDSPORE_LITE_MICRO_CODER_GENERATOR_INFERENCE_GENERATOR_H_

#include <utility>
#include <memory>
#include "coder/generator/generator.h"

namespace mindspore::lite::micro {
class InferenceGenerator : public Generator {
 public:
  explicit InferenceGenerator(std::unique_ptr<CoderContext> ctx) : Generator(std::move(ctx)) {}
  ~InferenceGenerator() override = default;

 private:
  int CodeNetHFile() override;
  int CodeNetCFile() override;
};
}  // namespace mindspore::lite::micro
#endif  // MINDSPORE_LITE_MICRO_CODER_GENERATOR_INFERENCE_GENERATOR_H_
