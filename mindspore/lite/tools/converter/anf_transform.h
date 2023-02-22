/**
 * Copyright 2019-2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_LITE_TOOLS_CONVERTER_ANF_TRANSFORM_H_
#define MINDSPORE_LITE_TOOLS_CONVERTER_ANF_TRANSFORM_H_

#include <memory>
#include <vector>
#include <set>
#include "backend/common/optimizer/optimizer.h"
#include "backend/common/optimizer/pass.h"
#include "schema/inner/model_generated.h"
#include "tools/common/meta_graph_serializer.h"
#include "ir/anf.h"
#include "tools/converter/quantizer/quantizer.h"
#include "tools/converter/converter_context.h"

namespace mindspore {
namespace lite {
class AnfTransform {
 public:
  AnfTransform();
  virtual ~AnfTransform();
  STATUS Transform(const FuncGraphPtr &old_graph, const std::shared_ptr<ConverterPara> &param);

 private:
  STATUS TransformFuncGraph(const FuncGraphPtr &old_graph, const std::shared_ptr<ConverterPara> &param);

  static int RunPass(const FuncGraphPtr &old_graph, const std::shared_ptr<ConverterPara> &param);

  static int RunFusionPass(const FuncGraphPtr &old_graph, const std::shared_ptr<ConverterPara> &param);

  static int RunGraphPass(const FuncGraphPtr &old_graph, const std::shared_ptr<ConverterPara> &param);

  static int RunConvertPass(const FuncGraphPtr &old_graph, const std::shared_ptr<ConverterPara> &param);

  static int RunInt64CastInt32Pass(const FuncGraphPtr &old_graph, const std::shared_ptr<ConverterPara> &param);

  static int RunConstFoldPass(const FuncGraphPtr &olde_graph, const std::shared_ptr<ConverterPara> &param);

  static int RunParallelPass(const FuncGraphPtr &old_graph, const std::shared_ptr<ConverterPara> &param);

  static int DoQuantize(const FuncGraphPtr &old_graph, const std::shared_ptr<ConverterPara> &param);

  static int DoFormatForMindIR(const FuncGraphPtr &old_graph, const std::shared_ptr<ConverterPara> &param);

  static bool StoreBuiltinPass(const std::shared_ptr<ConverterPara> &param);

  static int RunFormatTrans(const FuncGraphPtr &old_graph);

  static STATUS MarkTrainInputOp(const FuncGraphPtr &func_graph, const CNodePtr &cnode);

  static STATUS MarkTrainWeightSharingOp(const FuncGraphPtr &func_graph, const CNodePtr &cnode);

  static STATUS MarkTrainOp(const FuncGraphPtr &func_graph);

  static bool CheckExternalExtension(const std::shared_ptr<ConverterPara> &param);

  static STATUS ProcOnlineTransform(const FuncGraphPtr &old_graph, const std::shared_ptr<ConverterPara> &param);

  FuncGraphManagerPtr manager_ = nullptr;
};
}  // namespace lite
}  // namespace mindspore

#endif
