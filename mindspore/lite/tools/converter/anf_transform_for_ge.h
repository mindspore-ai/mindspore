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

#ifndef MINDSPORE_LITE_TOOLS_CONVERTER_ANF_TRANSFORM_FOR_GE_H_
#define MINDSPORE_LITE_TOOLS_CONVERTER_ANF_TRANSFORM_FOR_GE_H_

#include <memory>
#include <vector>
#include <set>
#include "include/backend/optimizer/optimizer.h"
#include "include/backend/optimizer/pass.h"
#include "schema/inner/model_generated.h"
#include "tools/common/meta_graph_serializer.h"
#include "ir/anf.h"
#include "tools/converter/quantizer/quantizer.h"
#include "tools/converter/converter_context.h"

namespace mindspore {
namespace lite {
class AnfTransformForGe {
 public:
  AnfTransformForGe();
  virtual ~AnfTransformForGe();
  STATUS Transform(const FuncGraphPtr &old_graph, const std::shared_ptr<ConverterPara> &param);

 private:
  static int RunGeFusionPass(const FuncGraphPtr &old_graph, const std::shared_ptr<ConverterPara> &param);
  FuncGraphManagerPtr manager_ = nullptr;
};
}  // namespace lite
}  // namespace mindspore

#endif
