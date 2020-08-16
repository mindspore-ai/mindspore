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

#ifndef MINDSPORE_LITE_SRC_PASS_COMMON_GLLO_UTILS_H_
#define MINDSPORE_LITE_SRC_PASS_COMMON_GLLO_UTILS_H_

#include <memory>
#include "src/ir/primitive_t_value.h"
#include "ir/anf.h"
#include "ir/func_graph.h"
#include "src/common/utils.h"
#include "backend/optimizer/common/pattern_engine.h"
#include "schema/inner/model_generated.h"
#include "src/param_value_lite.h"

using PrimitiveTValuePtr = std::shared_ptr<mindspore::lite::PrimitiveTValue>;
namespace mindspore {
namespace opt {
bool IsRealCNodeKernel(const AnfNodePtr &node);

bool IsGraphKernel(const AnfNodePtr &node);

void CheckIfFuncGraphIsNull(const FuncGraphPtr &graph);

void CheckIfAnfNodeIsNull(const AnfNodePtr &node);

void CheckIfCNodeIsNull(const CNodePtr &node);

void CheckIfVarIsNull(const VarPtr &var);

void CheckInputSize(const CNodePtr &node, int size);

void CheckIfNodeIsParam(const AnfNodePtr &node);

void CheckLeastInputSize(const CNodePtr &node, int size);

AnfNodePtr AddNewBiasNode(float *bias_data, const FuncGraphPtr &func_graph, int kernel_num,
                          const ParamValueLitePtr &weight_tensor);

schema::PrimitiveType GetCNodeType(const BaseRef &node);

bool IsParamNode(const BaseRef &n);

bool IsConvNode(const BaseRef &n);

bool CheckIsAllInputsParam(const AnfNodePtr &node);

size_t GetOutputTensorNum(const AnfNodePtr &node);

bool IsMultiOutputTensors(const FuncGraphPtr &graph, const AnfNodePtr &node);
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_LITE_SRC_PASS_COMMON_GLLO_UTILS_H_
