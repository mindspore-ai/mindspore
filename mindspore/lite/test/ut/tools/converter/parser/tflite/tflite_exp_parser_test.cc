/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
#include <iostream>
#include "mindspore/core/utils/log_adapter.h"
#include "common/common_test.h"
#include "tools/converter/converter_flags.h"
#include "schema/inner/model_generated.h"
#include "tools/converter/parser/tflite/tflite_converter.h"
#include "tools/converter/parser/tflite/tflite_exp_parser.h"
#include "src/kernel_registry.h"
#include "src/lite_kernel.h"

namespace mindspore {

class TestTfliteExpParser : public mindspore::Common {
 public:
  TestTfliteExpParser() {}
};

TEST_F(TestTfliteExpParser, ExpParser) {
  lite::converter::Flags flags;
  flags.modelFile = "./test_data/Exp.tflite";
  flags.fmk = lite::converter::FmkType_TFLITE;
  lite::TfliteConverter converter;
  schema::MetaGraphT *fb_graph = nullptr;
  fb_graph = converter.Convert(&flags);
  const auto &nodes = fb_graph->nodes;
  nodes.back();
}
}  // namespace mindspore
