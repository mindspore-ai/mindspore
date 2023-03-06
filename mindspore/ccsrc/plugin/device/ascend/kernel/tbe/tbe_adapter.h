/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_TBE_TBE_ADAPTER_H
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_TBE_TBE_ADAPTER_H

#include <string>
#include <map>
#include <memory>
#include <vector>
#include <algorithm>
#include <unordered_set>
#include "nlohmann/json.hpp"
#include "base/base.h"
#include "kernel/oplib/opinfo.h"
#include "kernel/kernel_fusion.h"
#include "backend/common/session/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
// Note: This file is mainly used to adapt the ME front-end operator description and
//       the TBE back-end operator implementation difference
namespace mindspore {
namespace kernel {
enum kCreaterType : int { SINGLE_BUILD = 0, OP_SELECT_FORMAT, CHECK_SUPPORTED, OP_PRE_COMPILE };
namespace tbe {
const std::map<std::string, std::string> opTypeAdapter = {{"ReLUV2", "ReluV2"},
                                                          {"ReLU6", "Relu6"},
                                                          {"ReLU6Grad", "Relu6Grad"},
                                                          {"ReLUGrad", "ReluGrad"},
                                                          {"ReLU", "Relu"},
                                                          {"Pad", "PadD"},
                                                          {"Gather", "GatherV2"},
                                                          {"SparseGatherV2", "GatherV2"},
                                                          {"SparseApplyFtrl", "SparseApplyFtrlD"},
                                                          {"Concat", "ConcatD"},
                                                          {"DepthwiseConv2dNative", "DepthwiseConv2D"},
                                                          {"FastGeLU", "FastGelu"},
                                                          {"FastGeLUGrad", "FastGeluGrad"},
                                                          {"GeLU", "Gelu"},
                                                          {"GeLUGrad", "GeluGrad"},
                                                          {"PReLU", "PRelu"},
                                                          {"PReLUGrad", "PReluGrad"},
                                                          {"SeLU", "Selu"},
                                                          {"TransposeNOD", "Transpose"},
                                                          {"ParallelResizeBilinear", "SyncResizeBilinearV2"},
                                                          {"ParallelResizeBilinearGrad", "SyncResizeBilinearV2Grad"},
                                                          {"ResizeBilinearGrad", "ResizeBilinearV2Grad"},
                                                          {"Split", "SplitD"},
                                                          {"HSwish", "HardSwish"},
                                                          {"HSwishGrad", "HardSwishGrad"},
                                                          {"CeLU", "CeluV2"},
                                                          {"ArgminV2", "ArgMin"},
                                                          {"IndexAdd", "InplaceIndexAdd"},
                                                          {"InplaceUpdateV2", "InplaceUpdate"},
                                                          {"CumSum", "Cumsum"},
                                                          {"UnsortedSegmentSumD", "UnsortedSegmentSum"},
                                                          {"KLDivLossGrad", "KlDivLossGrad"}};

enum FusionDataType { kFusionNormal = 0, kFusionAddN, kFusionReLUGradV2, kFusionAdd };
using FAttrsPass = void (*)(const AnfNodePtr &anf_node, const std::vector<std::shared_ptr<OpAttr>> &op_info_attrs,
                            nlohmann::json *attrs_json);
using FPreAttrsPass = void (*)(const AnfNodePtr &anf_node, std::vector<OpAttrPtr> *op_info_attrs,
                               nlohmann::json *attrs_json);
class TbeAdapter {
 public:
  TbeAdapter() = default;
  ~TbeAdapter() = default;
  template <typename T>
  static void InputOrderPass(const std::shared_ptr<AnfNode> &anf_node, std::vector<T> const &inputs_list,
                             std::vector<T> *inputs_json) {
    MS_EXCEPTION_IF_NULL(inputs_json);
    if (DynamicInputAdjusted(anf_node, inputs_list, inputs_json)) {
      return;
    }
    auto op_name = common::AnfAlgo::GetCNodeName(anf_node);
    if (input_order_adjusted_ops_.find(op_name) == input_order_adjusted_ops_.end()) {
      (void)std::copy(inputs_list.begin(), inputs_list.end(), std::back_inserter((*inputs_json)));
    } else {
      if (op_name == kMinimumGradOpName || op_name == kMaximumGradOpName) {
        if (inputs_list.size() < kIndex3) {
          MS_LOG(EXCEPTION) << "Op " << op_name << " should have at least " << kIndex3 << " inputs, but got "
                            << inputs_list.size();
        }
        inputs_json->push_back(inputs_list[kIndex2]);
        inputs_json->push_back(inputs_list[kIndex0]);
        inputs_json->push_back(inputs_list[kIndex1]);
        for (size_t i = 3; i < inputs_list.size(); ++i) {
          inputs_json->push_back(inputs_list[i]);
        }
      } else if (op_name == kApplyCenteredRMSPropOpName) {
        // Parameter order of ApplyCenteredRMSProp's TBE implementation is different from python API, so map
        // TBE parameter to correspond python API parameter by latter's index using hardcode
        if (inputs_list.size() < kIndex9) {
          MS_LOG(EXCEPTION) << "Op " << op_name << " should have at least " << kIndex9 << " inputs, but got "
                            << inputs_list.size();
        }
        inputs_json->push_back(inputs_list[kIndex0]);
        inputs_json->push_back(inputs_list[kIndex1]);
        inputs_json->push_back(inputs_list[kIndex2]);
        inputs_json->push_back(inputs_list[kIndex3]);
        inputs_json->push_back(inputs_list[kIndex5]);
        inputs_json->push_back(inputs_list[kIndex6]);
        inputs_json->push_back(inputs_list[kIndex7]);
        inputs_json->push_back(inputs_list[kIndex8]);
        inputs_json->push_back(inputs_list[kIndex4]);
      } else if (op_name == kStridedSliceGradOpName) {
        for (size_t i = 1; i < inputs_list.size(); ++i) {
          inputs_json->push_back(inputs_list[i]);
        }
        inputs_json->push_back(inputs_list[kIndex0]);
      } else {
        if (inputs_list.size() < kIndex2) {
          MS_LOG(EXCEPTION) << "Op " << op_name << " should have at least " << kIndex2 << " inputs, but got "
                            << inputs_list.size();
        }
        inputs_json->push_back(inputs_list[kIndex1]);
        inputs_json->push_back(inputs_list[kIndex0]);
        for (size_t i = 2; i < inputs_list.size(); ++i) {
          inputs_json->push_back(inputs_list[i]);
        }
      }
    }
  }

  template <typename T>
  static bool DynamicInputAdjusted(const std::shared_ptr<AnfNode> &anf_node, std::vector<T> const &inputs_list,
                                   std::vector<T> *inputs_json) {
    if (!common::AnfAlgo::IsDynamicShape(anf_node)) {
      return false;
    }
    auto op_name = common::AnfAlgo::GetCNodeName(anf_node);
    if (op_name == kConv2DBackpropInputOpName) {
      // process dynamic Conv2DBackpropInput, tbe kernel input is x, input_size and dout
      inputs_json->push_back(inputs_list[kIndex2]);
      inputs_json->push_back(inputs_list[kIndex1]);
      inputs_json->push_back(inputs_list[kIndex0]);
      return true;
    }
    if (op_name == kConv2DBackpropFilterOpName) {
      // process dynamic Conv2DBackpropFilter, tbe kernel input is filter_size, x and dout
      inputs_json->push_back(inputs_list[kIndex1]);
      inputs_json->push_back(inputs_list[kIndex2]);
      inputs_json->push_back(inputs_list[kIndex0]);
      return true;
    }
    if (op_name == kStridedSliceGradOpName) {
      inputs_json->push_back(inputs_list[kIndex1]);
      inputs_json->push_back(inputs_list[kIndex2]);
      inputs_json->push_back(inputs_list[kIndex3]);
      inputs_json->push_back(inputs_list[kIndex4]);
      inputs_json->push_back(inputs_list[kIndex0]);
      return true;
    }
    return false;
  }
  static void FusionDescJsonPass(const AnfNodePtr &node, nlohmann::json *output_desc,
                                 const std::map<const AnfNodePtr, tbe::FusionDataType> &spec_data_input);
  static std::string GetRealOpType(const std::string &origin_type);
  static std::string FormatPass(const std::string &format, const size_t &origin_shape_size);
  static bool GetSpecDataInput(const FusionScopeInfo &fusion_scope_info,
                               std::map<const AnfNodePtr, tbe::FusionDataType> *spec_data_input);
  static bool IsPlaceHolderInput(const AnfNodePtr &node, const OpIOInfoPtr &input_ptr);
  static void CastAttrJsonPrePass(const AnfNodePtr &anf_node, std::vector<OpAttrPtr> *op_info_attrs,
                                  nlohmann::json *attrs_json);
  static void CastAttrJsonPost(const AnfNodePtr &anf_node, nlohmann::json *attrs_json);
  static void LayerNormAttrJsonPost(const AnfNodePtr &anf_node, nlohmann::json *attrs_json);

 private:
  static bool IsSpecialFusionComputeNode(const std::vector<mindspore::AnfNodePtr> &compute_nodes);
  static bool GetSpecInputLayers(const std::string &op_name, const std::vector<mindspore::AnfNodePtr> &reorder_layer,
                                 std::map<const AnfNodePtr, FusionDataType> *spec_data_input);

  static std::unordered_set<std::string> input_order_adjusted_ops_;
};
}  // namespace tbe
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_TBE_TBE_ADAPTER_H
