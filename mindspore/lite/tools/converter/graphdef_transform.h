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

#ifndef MINDSPORE_LITE_TOOLS_CONVERTER_GRAPHDEF_TRANSFORM_H
#define MINDSPORE_LITE_TOOLS_CONVERTER_GRAPHDEF_TRANSFORM_H

#include <memory>
#include <vector>
#include "tools/converter/optimizer.h"
#include "tools/converter/quantizer/quantizer.h"
#include "schema/inner/model_generated.h"
#include "tools/common/storage.h"
#include "tools/converter/converter_flags.h"

namespace mindspore {
namespace lite {
/*
 * transform GraphDef by fusion legacy_optimizer and quantizer
 * */

class GraphDefTransform {
 public:
  GraphDefTransform();
  virtual ~GraphDefTransform();
  virtual int Transform(const converter::Flags &ctx);
  void SetGraphDef(schema::MetaGraphT *dst_def);
  inline schema::MetaGraphT *GetOutput() const { return graph_defT_; }

 protected:
  std::vector<schema::CNodeT *> GetGraphNodes();
  schema::MetaGraphT *graph_defT_ = nullptr;
  Optimizer *optimizer = nullptr;
};
}  // namespace lite
}  // namespace mindspore

#endif
