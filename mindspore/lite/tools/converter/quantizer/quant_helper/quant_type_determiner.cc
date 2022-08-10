/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include "tools/converter/quantizer/quant_helper/quant_type_determiner.h"
#include <utility>
#include <set>
#include "tools/optimizer/common/gllo_utils.h"
#include "tools/converter/quantizer/quantize_util.h"
#include "src/litert/kernel_exec.h"
#include "src/litert/kernel_registry.h"
#include "src/common/ops/anf_utils.h"
#include "tools/optimizer/common/format_utils.h"
#include "tools/common/node_util.h"

namespace mindspore::lite::quant {
namespace {
static const std::set<PrimitivePtr> fp32_output_operator = {prim::kPrimDetectionPostProcess};
}
bool QuantTypeDeterminer::DetermineQuantAll(const CNodePtr &cnode) {
  MS_ASSERT(cnode != nullptr);
  if (opt::IsSpecialType(cnode)) {
    return false;
  }
  auto primT = GetPrimitiveT(cnode->input(kPrimIndex));
  if (primT == nullptr) {
    MS_LOG(WARNING) << cnode->fullname_with_scope() << " primitive is nullptr.";
    return false;
  }
  // Check if there is an int8 operator.
  kernel::KernelKey desc{kernel::kCPU, kNumberTypeInt8, NHWC, primT->value.type, ""};
  if (!KernelRegistry::GetInstance()->SupportKernel(desc)) {
    return false;
  }

  // Check Quant Type.
  auto quant_holder = GetCNodeQuantHolder(cnode);
  if (quant_holder == nullptr) {
    MS_LOG(INFO) << cnode->fullname_with_scope() << " quant holder is nullptr.";
    return false;
  }

  // Get CNode QuantType directly.
  if (quant_holder->quant_type() != schema::QuantType_QUANT_NONE) {
    return quant_holder->quant_type() == schema::QuantType_QUANT_ALL;
  }
  // Check input quant params, quantization parameters exist for all activations.
  for (size_t i = 1; i < cnode->size(); ++i) {
    auto input = cnode->input(i);
    if (input->isa<CNode>() && !quant_holder->CheckInit(i - kPrimOffset, true)) {
      return false;
    }
  }
  // Check output quant params.
  if (CheckNodeInSet(cnode, fp32_output_operator) && !quant_holder->IsOutputQuantParamsInited()) {
    return false;
  }
  return true;
}

bool QuantTypeDeterminer::DetermineQuantWeight(const CNodePtr &cnode) {
  auto quant_holder = GetCNodeQuantHolder(cnode);
  if (quant_holder == nullptr) {
    MS_LOG(INFO) << cnode->fullname_with_scope() << " quant holder is nullptr.";
    return false;
  }

  // Get CNode QuantType directly.
  if (quant_holder->quant_type() != schema::QuantType_QUANT_NONE) {
    return quant_holder->quant_type() == schema::QuantType_QUANT_WEIGHT;
  }

  // Weight quantization, the output does not contain quantization information.
  if (quant_holder->IsOutputQuantParamsInited()) {
    return false;
  }

  bool quant_flag = false;
  for (size_t i = 1; i < cnode->size(); i++) {
    auto input = cnode->input(i);
    if (IsGraphInput(input)) {
      continue;
    }
    // non-constants(CNode) don't include quantization parameters
    if (input->isa<mindspore::CNode>()) {
      if (quant_holder->CheckInit(i - kPrimOffset, true)) {
        return false;
      }
    } else {
      // Constants have quantization parameters
      if (quant_holder->CheckInit(i - kPrimOffset, true)) {
        quant_flag = true;
        continue;
      }
    }
  }
  return quant_flag;
}

int QuantTypeDeterminer::Determine() {
  CHECK_NULL_RETURN(func_graph_);
  auto nodes = func_graph_->GetOrderedCnodes();
  for (auto const &cnode : nodes) {
    auto quant_holder = GetCNodeQuantHolder(cnode);
    if (quant_holder == nullptr) {
      MS_LOG(INFO) << cnode->fullname_with_scope() << " quant holder is nullptr.";
      continue;
    }
    if (!quant_holder->IsInputQuantParamsInited() && !quant_holder->IsOutputQuantParamsInited()) {  // Check FP32.
      if (opt::CheckPrimitiveType(cnode, prim::kPrimQuantDTypeCast)) {
        continue;
      }
      MS_LOG(INFO) << cnode->fullname_with_scope() << " Remove unused quant info";
      quant_holder->ClearQuantParams();
    } else if (DetermineQuantWeight(cnode)) {
      MS_LOG(INFO) << cnode->fullname_with_scope() << " set QuantType_QUANT_WEIGHT";
      quant_holder->set_quant_type(schema::QuantType_QUANT_WEIGHT);
    } else if (DetermineQuantAll(cnode)) {
      MS_LOG(INFO) << cnode->fullname_with_scope() << " set QuantType_QUANT_ALL";
      quant_holder->set_quant_type(schema::QuantType_QUANT_ALL);
    } else {
      MS_LOG(INFO) << cnode->fullname_with_scope() << " default quant type: QuantType_QUANT_NONE";
    }
  }
  return RET_OK;
}
}  // namespace mindspore::lite::quant
