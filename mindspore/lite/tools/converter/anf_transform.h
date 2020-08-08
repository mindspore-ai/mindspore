/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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

#ifndef MS_ANF_TRANSFORM_H
#define MS_ANF_TRANSFORM_H

#include "schema/inner/model_generated.h"
#include "tools/common/storage.h"
#include "tools/converter/converter_flags.h"
#include "ir/anf.h"


namespace mindspore {
namespace lite {
class AnfTransform {
 public:
  AnfTransform();
  virtual ~AnfTransform();
  FuncGraphPtr Transform(const FuncGraphPtr &old_graph);
  void SetGraphDef(schema::MetaGraphT *dstDef);
  inline schema::MetaGraphT *GetOutput() { return graphDefT; }

 protected:
  schema::MetaGraphT *graphDefT = nullptr;
};
}  // namespace lite
}  // namespace mindspore

#endif

