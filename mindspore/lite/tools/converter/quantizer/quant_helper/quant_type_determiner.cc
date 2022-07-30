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
#include "tools/optimizer/common/gllo_utils.h"
#include "tools/converter/quantizer/quantize_util.h"
#include "src/litert/kernel_exec.h"
#include "src/litert/kernel_registry.h"
#include "src/common/ops/anf_utils.h"

namespace mindspore::lite::quant {
std::pair<size_t, size_t> QuantTypeDeterminer::GetQuantParamsNum(const QuantParamHolderPtr &quant_holder) {
  // update input quant params num
  auto input_inited_quant_params = 0;
  auto input_tensors = quant_holder->get_input_quant_params();
  for (auto input : input_tensors) {
    bool is_quant_params_inited = !std::any_of(
      input.begin(), input.end(), [](const schema::QuantParamT &quant_param) { return !quant_param.inited; });
    if (is_quant_params_inited) {
      input_inited_quant_params++;
    }
  }
  auto output_inited_quant_params = 0;
  auto output_tensors = quant_holder->get_output_quant_params();
  for (auto output : output_tensors) {
    bool is_quant_params_inited = !std::any_of(
      output.begin(), output.end(), [](const schema::QuantParamT &quant_param) { return !quant_param.inited; });
    if (is_quant_params_inited) {
      output_inited_quant_params++;
    }
  }
  return {input_inited_quant_params, output_inited_quant_params};
}

bool QuantTypeDeterminer::DetermineQuantAll(const CNodePtr &cnode) {
  MS_ASSERT(node != nullptr);
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

  // GetCNodeQuantType
  if (quant_holder->quant_type() != schema::QuantType_QUANT_NONE) {
    return quant_holder->quant_type() == schema::QuantType_QUANT_ALL;
  }

  if (!quant_holder->IsInputQuantParamsInited() || !quant_holder->IsOutputQuantParamsInited()) {
    return false;
  }

  auto in_out_quant_params = GetQuantParamsNum(quant_holder);
  // Check quant param size is same as tensor size.
  auto input_size = (cnode->size() - kPrimOffset);
  auto output_size = opt::GetOutputSize(cnode);
  if (in_out_quant_params.first == input_size && in_out_quant_params.second == output_size) {
    quant_holder->set_quant_type(schema::QuantType_QUANT_ALL);
    return true;
  }
  return false;
}

bool QuantTypeDeterminer::DetermineQuantWeight(const CNodePtr &cnode) {
  auto quant_holder = GetCNodeQuantHolder(cnode);
  if (quant_holder == nullptr) {
    MS_LOG(INFO) << cnode->fullname_with_scope() << " quant holder is nullptr.";
    return false;
  }

  // GetCNodeQuantType
  if (quant_holder->quant_type() != schema::QuantType_QUANT_NONE) {
    return quant_holder->quant_type() == schema::QuantType_QUANT_WEIGHT;
  }

  // Weight quantization, the output does not contain quantization information.
  if (quant_holder->IsOutputQuantParamsInited()) {
    return false;
  }

  for (size_t i = 1; i < cnode->size(); i++) {
    auto input = cnode->input(i);
    // non-constants(CNode) don't include quantization parameters
    if (input->isa<mindspore::CNode>()) {
      if (quant_holder->CheckInit(i, true)) {
        return false;
      }
    } else {
      // Constants have quantization parameters
      if (quant_holder->CheckInit(i, true)) {
        return true;
      }
    }
  }
  return false;
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
    if (DetermineQuantWeight(cnode)) {
      MS_LOG(INFO) << cnode->fullname_with_scope() << " set QuantType_QUANT_WEIGHT";
      quant_holder->set_quant_type(schema::QuantType_QUANT_WEIGHT);
    } else if (DetermineQuantAll(cnode)) {
      MS_LOG(INFO) << cnode->fullname_with_scope() << " set QuantType_QUANT_ALL";
      quant_holder->set_quant_type(schema::QuantType_QUANT_ALL);
    } else {
      MS_LOG(INFO) << cnode->fullname_with_scope() << " Remove unused quant info";
      quant_holder->ClearQuantParams();
    }
  }
  return RET_OK;
}
}  // namespace mindspore::lite::quant
