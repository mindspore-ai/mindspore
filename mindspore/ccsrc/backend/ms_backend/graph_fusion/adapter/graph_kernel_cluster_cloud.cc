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
#include "backend/ms_backend/graph_fusion/adapter/graph_kernel_cluster_cloud.h"
#include <set>
#include <functional>
#include <unordered_map>
#include <string>
#include <memory>
#include <vector>
#include "primitive/math_ops.h"
#include "primitive/array_ops.h"
#include "include/utils/anfalgo.h"
#include "utils/anf_utils.h"
#include "utils/ms_context.h"
#include "include/runtime/hardware_abstract/kernel_base/graph_fusion/graph_kernel_flags.h"
#include "include/runtime/hardware_abstract/kernel_base/graph_fusion/graph_kernel/graph_kernel_callback.h"
#include "backend/ms_backend/graph_fusion/core/graph_kernel_utils.h"
#include "backend/ms_backend/graph_fusion/core/value_depend_op_utils.h"
#include "backend/ms_backend/graph_fusion/graph_kernel_helper.h"
#include "include/runtime/hardware_abstract/kernel_base/graph_fusion/graph_kernel/graph_kernel_comm_info_manager.h"
#include "primitive/other_ops.h"  // collective communication operations
#include "primitive/framework_ops.h"
#include "primitive/auto_generate/gen_ops_primitive_a.h"
#include "primitive/auto_generate/gen_ops_primitive_b.h"
#include "primitive/auto_generate/gen_ops_primitive_c.h"
#include "primitive/auto_generate/gen_ops_primitive_d.h"
#include "primitive/auto_generate/gen_ops_primitive_e.h"
#include "primitive/auto_generate/gen_ops_primitive_f.h"
#include "primitive/auto_generate/gen_ops_primitive_g.h"
#include "primitive/auto_generate/gen_ops_primitive_i.h"
#include "primitive/auto_generate/gen_ops_primitive_l.h"
#include "primitive/auto_generate/gen_ops_primitive_m.h"
#include "primitive/auto_generate/gen_ops_primitive_n.h"
#include "primitive/auto_generate/gen_ops_primitive_o.h"
#include "primitive/auto_generate/gen_ops_primitive_p.h"
#include "primitive/auto_generate/gen_ops_primitive_r.h"
#include "primitive/auto_generate/gen_ops_primitive_s.h"
#include "primitive/auto_generate/gen_ops_primitive_t.h"

namespace mindspore::graphkernel {
namespace {
// The value of max dimension size is due to two constraints:
// 1. current implementation does not support stride between two rows greater than UINT16_MAX
// 2. even if the row stride in the original input shape does not exceed UINT16_MAX, after address
// alignment, it can potentially exceed UINT16_MAX.
// The current value of kMaxDimSize guarantees that after address alignment, row stride is within
// a reasonable range.
constexpr int64_t kMaxDimSize = UINT16_MAX - UINT8_MAX;
constexpr int64_t kMinDimSize = 512;

bool IsFloatType(TypeId type_id) {
  switch (type_id) {
    case kNumberTypeFloat16:
    case kNumberTypeFloat32:
    case kNumberTypeBFloat16:
      return true;
    default:
      return false;
  }
}

bool IsFloatIntType(TypeId type_id) { return IsFloatType(type_id) || type_id == kNumberTypeInt32; }

class DvmSupportChecker {
 public:
  static DvmSupportChecker &Instance() {
    static DvmSupportChecker instance;
    return instance;
  }

  bool Check(const AnfNodePtr &node) {
    if (!CheckFormat(node)) {
      return false;
    }
    auto prim = GetCNodePrimitive(node);
    MS_EXCEPTION_IF_NULL(prim);
    std::string op_name = prim->name();

    auto it = check_func_.find(op_name);
    if (it != check_func_.end()) {
      const auto &funcs = it->second;
      for (const auto &func : funcs) {
        if (!func(node)) {
          return false;
        }
      }
      return true;
    } else {
      auto node_output_type = GetNodeOutputType(node);
      return IsFloatType(node_output_type) && InputCheck(node, {});
    }
  }

 private:
  DvmSupportChecker() {
    auto input_same_type = [](const AnfNodePtr &node) {
      auto cb = Callback::Instance();
      MS_EXCEPTION_IF_NULL(cb);
      auto input_num = common::AnfAlgo::GetInputTensorNum(node);
      if (input_num > 1) {
        auto base_type = cb->GetInputType(node, 0);
        for (size_t i = 1; i < input_num; ++i) {
          auto input_type = cb->GetInputType(node, i);
          if (input_type != base_type) {
            return false;
          }
        }
      }
      return true;
    };
    auto input_check_all = [](const AnfNodePtr &node) { return InputCheck(node, {}); };
    auto input_check_first = [](const AnfNodePtr &node) { return InputCheck(node, {1}); };
    auto cast_check = [](const AnfNodePtr &node) {
      auto node_output_type = GetNodeOutputType(node);
      auto cb = Callback::Instance();
      MS_EXCEPTION_IF_NULL(cb);
      static std::set<TypeId> supported_types{kNumberTypeFloat16, kNumberTypeFloat32, kNumberTypeBool, kNumberTypeInt32,
                                              kNumberTypeBFloat16};
      auto node_input_type = cb->GetInputType(node, 0);
      return !(supported_types.find(node_input_type) == supported_types.end() ||
               supported_types.find(node_output_type) == supported_types.end());
    };
    auto int_op_check = [](const AnfNodePtr &node) {
      auto node_output_type = GetNodeOutputType(node);
      return IsFloatIntType(node_output_type);
    };
    auto compare_check = [](const AnfNodePtr &node) {
      auto cb = Callback::Instance();
      MS_EXCEPTION_IF_NULL(cb);
      auto node_input_type = cb->GetInputType(node, 0);
      return IsFloatIntType(node_input_type);
    };
    auto transpose_op_check = [](const AnfNodePtr &node) {
      auto node_output_type = GetNodeOutputType(node);
      return node_output_type == kNumberTypeFloat16 || node_output_type == kNumberTypeFloat32;
    };
    auto collective_comm_op_check = [](const AnfNodePtr &node) {
      auto cb = Callback::Instance();
      auto node_input_type = cb->GetInputType(node, 0);
      // only support fp16 and fp32 at present
      if (node_input_type != kNumberTypeFloat16 && node_input_type != kNumberTypeFloat32) {
        return false;
      }
      const auto &node_input_shape = cb->GetInputShape(node, 0);
      auto input_size = std::accumulate(node_input_shape.begin(), node_input_shape.end(), static_cast<int64_t>(1),
                                        std::multiplies<int64_t>());
      if (input_size == 1) {
        return false;
      }
      const std::string &device_target = MsContext::GetInstance()->get_param<std::string>(MS_CTX_DEVICE_TARGET);
      auto comm_info = GraphKernelCommInfoManager::Instance().GetCommInfo(device_target);
      if (comm_info == nullptr) {
        return false;
      }
      if (comm_info->IsTargetCommOp(node)) {
        return true;
      }
      return false;
    };
    // cast op
    check_func_["Cast"] = {cast_check};
    // reducesum op
    check_func_["ReduceSum"] = {DvmSupportChecker::DvmReduceSumSupported, input_check_first};
    // cmp op
    check_func_["Equal"] = {compare_check, input_same_type};
    check_func_["NotEqual"] = {compare_check, input_same_type};
    check_func_["Greater"] = {compare_check, input_same_type};
    check_func_["GreaterEqual"] = {compare_check, input_same_type};
    check_func_["Less"] = {compare_check, input_same_type};
    check_func_["LessEqual"] = {compare_check, input_same_type};
    check_func_["IsFinite"] = {compare_check, [](const AnfNodePtr &node) {
                                 return Callback::Instance()->GetInputType(node, 0) != kNumberTypeInt32;
                               }};
    // select op
    check_func_["Select"] = {DvmSupportChecker::DvmSelectSupported,
                             [](const AnfNodePtr &node) { return InputCheck(node, {2, 3}); }};
    // int op
    check_func_["Add"] = {int_op_check, input_check_all};
    check_func_["Sub"] = {int_op_check, input_check_all};
    check_func_["Mul"] = {FloatIntMixTypeCheck};
    check_func_["Maximum"] = {int_op_check, input_check_all};
    check_func_["Minimum"] = {int_op_check, input_check_all};
    check_func_["Neg"] = {int_op_check, input_check_all};
    check_func_["Abs"] = {int_op_check, input_check_all};
    check_func_["Assign"] = {int_op_check, input_check_all};
    check_func_["BroadcastTo"] = {int_op_check, input_check_first};
    // slice op
    check_func_["Slice"] = {DvmSupportChecker::DvmSliceSupported, input_check_first};
    check_func_["StridedSlice"] = {DvmSupportChecker::DvmSliceSupported, input_check_first};
    // matmul op
    check_func_["MatMul"] = {DvmSupportChecker::DvmMatMulSupported, input_check_all};
    check_func_["BatchMatMul"] = {DvmSupportChecker::DvmMatMulSupported, input_check_all};
    check_func_[ops::kNameGroupedMatmul] = {DvmSupportChecker::DvmGroupedMatmulSupported};
    // transpose op
    check_func_["Transpose"] = {transpose_op_check, input_check_all};
    // collective comm op
    check_func_["AllReduce"] = {collective_comm_op_check};
    check_func_["Reshape"] = {DvmSupportChecker::DvmReshapeSupported};
  }

  static TypeId GetNodeOutputType(const AnfNodePtr &node) {
    auto cb = Callback::Instance();
    MS_EXCEPTION_IF_NULL(cb);
    return cb->GetOutputType(node, 0);
  }

  static bool InputCheck(const AnfNodePtr &node, const std::vector<size_t> &inputs_to_check) {
    auto cb = Callback::Instance();
    MS_EXCEPTION_IF_NULL(cb);
    auto node_output_type = GetNodeOutputType(node);
    auto cnode = node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);

    size_t input_num = cnode->size() - 1;
    std::vector<size_t> inputs;
    if (inputs_to_check.empty()) {
      for (size_t i = 1; i <= input_num; ++i) {
        inputs.push_back(i);
      }
    } else {
      inputs = inputs_to_check;
    }
    for (size_t idx : inputs) {
      auto input_node = cnode->input(idx);
      MS_EXCEPTION_IF_NULL(input_node);
      auto input_abstract = input_node->abstract();
      if (input_abstract->isa<abstract::AbstractTensor>() && cb->GetInputType(node, idx - 1) != node_output_type) {
        return false;
      }
    }
    return true;
  }

  static bool MixTypeCheck(const AnfNodePtr &node, const std::function<bool(TypeId)> &type_check,
                           const std::vector<size_t> &inputs_to_check) {
    MS_EXCEPTION_IF_NULL(node);
    auto cb = Callback::Instance();
    MS_EXCEPTION_IF_NULL(cb);
    auto output_type = cb->GetOutputType(node, 0);
    if (!type_check(output_type)) {
      return false;
    }
    std::vector<size_t> inputs(inputs_to_check);
    size_t input_num = common::AnfAlgo::GetInputTensorNum(node);
    if (inputs_to_check.empty()) {
      for (size_t i = 1; i <= input_num; ++i) {
        inputs.push_back(i);
      }
    }
    for (auto idx : inputs) {
      auto input_type = cb->GetInputType(node, idx - 1);
      if (!type_check(input_type)) {
        return false;
      }
    }
    return true;
  }

  static bool FloatIntMixTypeCheck(const AnfNodePtr &node) { return MixTypeCheck(node, IsFloatIntType, {}); }

  static bool CheckFormat(const AnfNodePtr &node) {
    if (common::AnfAlgo::IsDynamicRankNode(node)) {
      MS_LOG(DEBUG) << "skip dynamic rank";
      return false;
    }
    if (common::AnfAlgo::IsDynamicShape(node) && !CheckDefaultFormat(node)) {
      // dvm kernel infer shape use inputs device shape, but the output abstract shape inferred from device shape is
      // not unique if some shape value are not a multiple of 16
      MS_LOG(DEBUG) << "skip node: " << node->fullname_with_scope()
                    << " because only default format is supported in dynamic shape";
      return false;
    }
    auto cb = Callback::Instance();
    MS_EXCEPTION_IF_NULL(cb);
    auto input_num = AnfUtils::GetInputTensorNum(node);
    if (input_num > 0) {
      bool has_special_format = false;
      auto base_format = cb->GetInputFormat(node, 0);
      for (size_t i = 0; i < input_num; ++i) {
        auto input_format = cb->GetInputFormat(node, i);
        if (!has_special_format &&
            (input_format.find("FRACTAL") != std::string::npos || input_format.find("C0") != std::string::npos)) {
          has_special_format = true;
        }
        if (has_special_format && input_format != base_format) {
          // mixed special format and default format is not supported, because extra Reshape/TransData is needed
          return false;
        }
      }
    }
    return true;
  }

  static bool DvmSliceSupported(const AnfNodePtr &node) {
    constexpr size_t kMaxRank = 4;
    if (common::AnfAlgo::IsDynamicRankNode(node)) {
      return false;
    }
    auto node_output_type = GetNodeOutputType(node);
    auto cnode = node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    auto output_shape = GetShape(node);
    auto input_shape = GetShape(cnode->input(kIndex1));
    auto rank = output_shape.size();
    for (size_t i = kIndex3; i < rank; i++) {
      if (input_shape[rank - 1 - i] != output_shape[rank - 1 - i]) {
        return false;
      }
    }
    if (input_shape.size() > kMaxRank) {
      return false;
    }
    if (IsPrimitiveCNode(node, prim::kPrimStridedSlice)) {
      auto step_node = cnode->input(kIndex4)->cast<ValueNodePtr>();
      if (step_node == nullptr) {
        return false;
      }
      auto step_value = step_node->value();
      MS_EXCEPTION_IF_NULL(step_value);
      auto step_vector = GetValue<std::vector<int64_t>>(step_value);

      if (std::any_of(step_vector.begin(), step_vector.end(), [](int i) { return i != 1; })) {
        return false;
      }
    }
    return IsFloatIntType(node_output_type);
  }

  static bool DvmMatMulSupported(const AnfNodePtr &node) {
    auto node_output_type = GetNodeOutputType(node);
    if (common::AnfAlgo::IsDynamicShape(node)) {
      return false;
    }
    if (node_output_type != kNumberTypeFloat16 && node_output_type != kNumberTypeBFloat16) {
      return false;
    }
    auto cnode = node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    auto a_shape = GetShape(cnode->input(kIndex1));
    auto b_shape = GetShape(cnode->input(kIndex2));
    auto c_shape = GetShape(node);
    if (a_shape.back() > kMaxDimSize || b_shape.back() > kMaxDimSize) {
      return false;
    }
    if (IsPrimitiveCNode(node, prim::kPrimMatMul) && c_shape.back() <= kMinDimSize &&
        c_shape[c_shape.size() - kSizeTwo] <= kMinDimSize) {
      return false;
    }
    if (IsPrimitiveCNode(node, prim::kPrimBatchMatMul) && c_shape.size() > kSizeFour) {
      return false;
    }
    return true;
  }

  static bool DvmGroupedMatmulSupported(const AnfNodePtr &node) {
    constexpr int64_t kGroupTypeK = 2;
    constexpr int64_t kGroupTypeM = 0;
    constexpr int64_t KSplitNumType3 = 3;
    auto prim = GetCNodePrimitive(node);
    MS_EXCEPTION_IF_NULL(prim);
    auto split_item = GetValue<int64_t>(prim->GetAttr("split_item"));
    auto group_type = GetValue<int64_t>(prim->GetAttr("group_type"));
    if (split_item != KSplitNumType3 || (group_type != kGroupTypeM && group_type != kGroupTypeK)) {
      return false;
    }
    auto node_output_type = GetNodeOutputType(node);
    if (node_output_type != kNumberTypeFloat16 && node_output_type != kNumberTypeBFloat16) {
      return false;
    }
    auto cnode = node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    for (size_t i = kIndex4; i < kIndex8; i++) {
      auto input_node = cnode->input(i);
      MS_EXCEPTION_IF_NULL(input_node);
      if (input_node->isa<ValueNode>() && input_node->cast<ValueNodePtr>()->value()->isa<None>()) {
        continue;
      }
      if (GetShape(input_node) == ShapeVector{0}) {
        continue;
      }
      return false;
    }
    auto a_shape = GetShape(cnode->input(kIndex1));
    auto b_shape = GetShape(cnode->input(kIndex2));
    if (a_shape.back() > kMaxDimSize || b_shape.back() > kMaxDimSize) {
      return false;
    }
    return true;
  }

  static bool DvmReduceSumSupported(const AnfNodePtr &node) {
    auto node_output_type = GetNodeOutputType(node);
    auto prim = GetCNodePrimitive(node);
    MS_EXCEPTION_IF_NULL(prim);
    auto skip_mode_attr = prim->GetAttr(kAttrSkipMode);
    MS_EXCEPTION_IF_NULL(skip_mode_attr);
    auto skip_mode = GetValue<bool>(skip_mode_attr);
    if (skip_mode) {
      auto cnode = node->cast<CNodePtr>();
      MS_EXCEPTION_IF_NULL(cnode);
      auto axis = cnode->input(kIndex2);
      auto axis_abs = axis->abstract();
      if (!axis_abs->isa<abstract::AbstractSequence>() ||
          axis_abs->cast<abstract::AbstractSequencePtr>()->dynamic_len() ||
          axis_abs->cast<abstract::AbstractSequencePtr>()->size() == 0) {
        return false;
      }
    }
    return IsFloatType(node_output_type);
  }

  static bool DvmCubeReshapeSupported(const AnfNodePtr &reshape_node) {
    auto node = reshape_node->cast<CNodePtr>()->input(kIndex1);
    AnfNodePtr cube_op = nullptr;
    if (IsPrimitiveCNode(node, prim::kPrimMatMul) || IsPrimitiveCNode(node, prim::kPrimBatchMatMul)) {
      cube_op = node;
    } else if (IsPrimitiveCNode(node, prim::kPrimTupleGetItem)) {
      auto cnode = node->cast<CNodePtr>()->input(kIndex1);
      if (IsPrimitiveCNode(cnode, prim::kPrimGroupedMatmul)) {
        cube_op = cnode;
      }
    }
    return cube_op && StaticShapeCluster::CanClusterableOp(cube_op, StaticShapeCluster::GetClusterOps());
  }

  static bool DvmReshapeSupported(const AnfNodePtr &node) { return DvmSupportChecker::DvmCubeReshapeSupported(node); }

  static bool DvmSelectSupported(const AnfNodePtr &node) {
    auto node_output_type = GetNodeOutputType(node);
    auto cb = Callback::Instance();
    if (cb->GetInputType(node, 0) != kNumberTypeBool) {
      return false;
    }
    return IsFloatType(node_output_type);
  }

  std::unordered_map<std::string, std::vector<std::function<bool(const AnfNodePtr &)>>> check_func_;
};

bool DvmSupported(const AnfNodePtr &node) { return DvmSupportChecker::Instance().Check(node); }

const std::vector<OpWithLevel> clusterable_ops_with_level = {
  // all target
  {kAllTarget, OpLevel_0, prim::kPrimAbs},
  {kAllTarget, OpLevel_0, prim::kPrimAdd},
  {kAllTarget, OpLevel_0, prim::kPrimCast},
  {kAllTarget, OpLevel_0, prim::kPrimEqual},
  {kAllTarget, OpLevel_0, prim::kPrimExp},
  {kAllTarget, OpLevel_0, prim::kPrimLog},
  {kAllTarget, OpLevel_0, prim::kPrimMaximum},
  {kAllTarget, OpLevel_0, prim::kPrimMinimum},
  {kAllTarget, OpLevel_0, prim::kPrimMul},
  {kAllTarget, OpLevel_0, prim::kPrimNeg},
  {kAllTarget, OpLevel_0, prim::kPrimPow},
  {kAllTarget, OpLevel_0, prim::kPrimRealDiv},
  {kAllTarget, OpLevel_0, prim::kPrimReciprocal},
  {kAllTarget, OpLevel_1, prim::kPrimReduceSum},
  {kAllTarget, OpLevel_1, prim::kPrimReshape},
  {kAllTarget, OpLevel_0, prim::kPrimRound},
  {kAllTarget, OpLevel_0, prim::kPrimRsqrt},
  {kAllTarget, OpLevel_0, prim::kPrimSqrt},
  {kAllTarget, OpLevel_0, prim::kPrimSub},
  {kAllTarget, OpLevel_0, prim::kPrimTanh},
  {kAllTarget, OpLevel_1, prim::kPrimTranspose},
  // ascend
  {kAscendDevice, OpLevel_1, prim::kPrimMatMul},
  {kAscendDevice, OpLevel_1, prim::kPrimTransData},
  {kAscendDevice, OpLevel_1, prim::kPrimBatchMatMul},
  // gpu
  {kGPUDevice, OpLevel_0, prim::kPrimACos},
  {kGPUDevice, OpLevel_0, prim::kPrimAcosh},
  {kGPUDevice, OpLevel_2, prim::kPrimArgMax},
  {kGPUDevice, OpLevel_2, prim::kPrimArgmin},
  {kGPUDevice, OpLevel_0, prim::kPrimAsin},
  {kGPUDevice, OpLevel_0, prim::kPrimAsinh},
  {kGPUDevice, OpLevel_0, prim::kPrimAssign},
  {kGPUDevice, OpLevel_0, prim::kPrimAtan},
  {kGPUDevice, OpLevel_0, prim::kPrimAtan2},
  {kGPUDevice, OpLevel_0, prim::kPrimCos},
  {kGPUDevice, OpLevel_0, prim::kPrimDiv},
  {kGPUDevice, OpLevel_0, prim::kPrimErf},
  {kGPUDevice, OpLevel_0, prim::kPrimExpm1},
  {kGPUDevice, OpLevel_0, prim::kPrimFloor},
  {kGPUDevice, OpLevel_0, prim::kPrimFloorDiv},
  {kGPUDevice, OpLevel_0, prim::kPrimFloorMod},
  {kGPUDevice, OpLevel_0, prim::kPrimGreater},
  {kGPUDevice, OpLevel_0, prim::kPrimGreaterEqual},
  {kGPUDevice, OpLevel_0, prim::kPrimIsFinite},
  {kGPUDevice, OpLevel_0, prim::kPrimIsInf},
  {kGPUDevice, OpLevel_0, prim::kPrimIsNan},
  {kGPUDevice, OpLevel_0, prim::kPrimLess},
  {kGPUDevice, OpLevel_0, prim::kPrimLessEqual},
  {kGPUDevice, OpLevel_0, prim::kPrimLogicalAnd},
  {kGPUDevice, OpLevel_0, prim::kPrimLogicalOr},
  {kGPUDevice, OpLevel_0, prim::kPrimLogicalNot},
  {kGPUDevice, OpLevel_0, prim::kPrimMod},
  {kGPUDevice, OpLevel_0, prim::kPrimNotEqual},
  {kGPUDevice, OpLevel_1, prim::kPrimReduceMax},
  {kGPUDevice, OpLevel_1, prim::kPrimReduceMin},
  {kGPUDevice, OpLevel_0, prim::kPrimSelect},
  {kGPUDevice, OpLevel_0, prim::kPrimSign},
  {kGPUDevice, OpLevel_0, prim::kPrimSin},
  {kGPUDevice, OpLevel_0, prim::kPrimStridedSlice},
  {kGPUDevice, OpLevel_1, prim::kPrimCumSum},
  {kGPUDevice, OpLevel_1, prim::kPrimOneHot},
  // cpu
  {kCPUDevice, OpLevel_0, prim::kPrimLogicalNot},
  {kCPUDevice, OpLevel_0, prim::kPrimMod},
  {kCPUDevice, OpLevel_1, prim::kPrimReduceMax},
  {kCPUDevice, OpLevel_0, prim::kPrimSelect},
  {kCPUDevice, OpLevel_0, prim::kPrimLess},
  {kCPUDevice, OpLevel_0, prim::kPrimLessEqual},
};

const std::vector<OpWithLevel> clusterable_ops_with_level_v2 = {
  // cpu
  {kCPUDevice, OpLevel_0, prim::kPrimNotEqual},
  {kCPUDevice, OpLevel_0, prim::kPrimGreaterEqual},
  {kCPUDevice, OpLevel_0, prim::kPrimGreater},
  {kCPUDevice, OpLevel_0, prim::kPrimFloor},
  {kCPUDevice, OpLevel_0, prim::kPrimIsNan},
  {kCPUDevice, OpLevel_0, prim::kPrimAssign},
  {kCPUDevice, OpLevel_0, prim::kPrimBroadcastTo},
  {kCPUDevice, OpLevel_0, prim::kPrimTile},
  {kCPUDevice, OpLevel_0, prim::kPrimLogicalAnd},
  {kCPUDevice, OpLevel_0, prim::kPrimCos},
  {kCPUDevice, OpLevel_0, prim::kPrimSin},
  {kCPUDevice, OpLevel_0, prim::kPrimACos},
  {kCPUDevice, OpLevel_0, prim::kPrimAsin},
  {kCPUDevice, OpLevel_0, prim::kPrimTanh},
  {kCPUDevice, OpLevel_0, prim::kPrimAtan2},
  {kCPUDevice, OpLevel_0, prim::kPrimMinimum},
  {kCPUDevice, OpLevel_0, prim::kPrimMaximum},
  {kCPUDevice, OpLevel_0, prim::kPrimReduceAll},
  {kCPUDevice, OpLevel_0, prim::kPrimStridedSlice},
  // gpu
  {kGPUDevice, OpLevel_0, prim::kPrimNotEqual},
  {kGPUDevice, OpLevel_0, prim::kPrimSelect},
  {kGPUDevice, OpLevel_0, prim::kPrimTile},
  {kGPUDevice, OpLevel_0, prim::kPrimLogicalAnd},
  {kGPUDevice, OpLevel_0, prim::kPrimCos},
  {kGPUDevice, OpLevel_0, prim::kPrimSin},
  {kGPUDevice, OpLevel_0, prim::kPrimMinimum},
  {kGPUDevice, OpLevel_0, prim::kPrimMaximum},
  {kGPUDevice, OpLevel_0, prim::kPrimAssign},
};

const std::vector<std::string> disable_cluster_op_list_v2 = {"OneHot", "CumSum",      "Transpose",   "BatchMatMul",
                                                             "MatMul", "BroadcastTo", "StridedSlice"};

// note: inplace op can not be fused by default, as view + inplace case may have precision error
const std::vector<OpWithLevel> clusterable_ops_with_level_dvm = {
  {kAscendDevice, OpLevel_0, prim::kPrimAbs},          {kAscendDevice, OpLevel_0, prim::kPrimAdd},
  {kAscendDevice, OpLevel_0, prim::kPrimBroadcastTo},  {kAscendDevice, OpLevel_0, prim::kPrimCast},
  {kAscendDevice, OpLevel_0, prim::kPrimExp},          {kAscendDevice, OpLevel_0, prim::kPrimLog},
  {kAscendDevice, OpLevel_0, prim::kPrimMaximum},      {kAscendDevice, OpLevel_0, prim::kPrimMinimum},
  {kAscendDevice, OpLevel_0, prim::kPrimMul},          {kAscendDevice, OpLevel_0, prim::kPrimNeg},
  {kAscendDevice, OpLevel_0, prim::kPrimPow},          {kAscendDevice, OpLevel_0, prim::kPrimDiv},
  {kAscendDevice, OpLevel_0, prim::kPrimRealDiv},      {kAscendDevice, OpLevel_0, prim::kPrimReciprocal},
  {kAscendDevice, OpLevel_0, prim::kPrimRsqrt},        {kAscendDevice, OpLevel_0, prim::kPrimSqrt},
  {kAscendDevice, OpLevel_0, prim::kPrimSub},          {kAscendDevice, OpLevel_0, prim::kPrimEqual},
  {kAscendDevice, OpLevel_0, prim::kPrimNotEqual},     {kAscendDevice, OpLevel_0, prim::kPrimGreater},
  {kAscendDevice, OpLevel_0, prim::kPrimGreaterEqual}, {kAscendDevice, OpLevel_0, prim::kPrimLess},
  {kAscendDevice, OpLevel_0, prim::kPrimLessEqual},    {kAscendDevice, OpLevel_0, prim::kPrimLogicalAnd},
  {kAscendDevice, OpLevel_0, prim::kPrimLogicalOr},    {kAscendDevice, OpLevel_0, prim::kPrimLogicalNot},
  {kAscendDevice, OpLevel_0, prim::kPrimSelect},       {kAscendDevice, OpLevel_0, prim::kPrimAssign},
  {kAscendDevice, OpLevel_0, prim::kPrimReduceSum},    {kAscendDevice, OpLevel_0, prim::kPrimIsFinite},
  {kAscendDevice, OpLevel_1, prim::kPrimReshape},      {kAscendDevice, OpLevel_0, prim::kPrimTranspose},
  {kAscendDevice, OpLevel_0, prim::kPrimFloor},        {kAscendDevice, OpLevel_0, prim::kPrimCeil},
  {kAscendDevice, OpLevel_0, prim::kPrimTrunc},        {kAscendDevice, OpLevel_1, prim::kPrimMatMul},
  {kAscendDevice, OpLevel_1, prim::kPrimBatchMatMul},  {kAscendDevice, OpLevel_1, prim::kPrimGroupedMatmul},
  {kAscendDevice, OpLevel_2, prim::kPrimTensorMove},
};

bool IsComplexDataType(const AnfNodePtr &node) {
  auto cb = Callback::Instance();
  MS_EXCEPTION_IF_NULL(cb);
  auto node_output_type = cb->GetOutputType(node, 0);
  if (node_output_type == kNumberTypeComplex64 || node_output_type == kNumberTypeComplex128) {
    return true;
  }
  if (IsPrimitiveCNode(node, prim::kPrimCast)) {
    auto node_input_type = cb->GetInputType(node, 0);
    if ((node_input_type == kNumberTypeComplex64) || (node_input_type == kNumberTypeComplex128)) {
      return true;
    }
  }
  return false;
}
}  // namespace

std::vector<PrimitivePtr> StaticShapeCluster::GetClusterOps() {
  const auto &flags = GraphKernelFlags::GetInstance();
  std::vector<std::string> disable_cluster_ops = flags.disable_cluster_ops;
  auto cb = Callback::Instance();

  std::vector<OpWithLevel> clusterable_ops;
  if (flags.kernel_generator == "AKG_V2") {
    clusterable_ops = clusterable_ops_with_level;
    clusterable_ops.insert(clusterable_ops.end(), clusterable_ops_with_level_v2.begin(),
                           clusterable_ops_with_level_v2.end());
    if (cb->GetTargetFromContext() == kCPUDevice &&
        std::find(flags.enable_cluster_ops.begin(), flags.enable_cluster_ops.end(), "Reshape") ==
          flags.enable_cluster_ops.end()) {
      disable_cluster_ops.push_back("Reshape");
    }
    if (cb->GetTargetFromContext() == kGPUDevice) {
      for (const std::string &item : disable_cluster_op_list_v2) {
        if (std::find(flags.enable_cluster_ops.begin(), flags.enable_cluster_ops.end(), item) ==
            flags.enable_cluster_ops.end()) {
          disable_cluster_ops.push_back(item);
        }
      }
    }
  } else if (flags.kernel_generator == "DVM") {
    clusterable_ops = clusterable_ops_with_level_dvm;
  } else {
    clusterable_ops = clusterable_ops_with_level;
  }
  auto ops = GkUtils::GetValidOps(clusterable_ops, flags.fusion_ops_level, flags.enable_cluster_ops_only,
                                  flags.enable_cluster_ops, disable_cluster_ops);
  return GkUtils::FilterExcludedOps(ops);
}

std::vector<PrimitivePtr> StaticShapeCluster::GetClusterableOpList() { return StaticShapeCluster::GetClusterOps(); }

bool SkipHostInputNode(const AnfNodePtr &node, bool is_dvm) {
  if (is_dvm && GraphKernelFlags::GetInstance().IsEnableKernelPacket()) {
    auto cnode = node->cast<CNodePtr>();
    return cnode != nullptr &&
           std::any_of(cnode->inputs().begin() + 1, cnode->inputs().end(), AnfAlgo::IsKernelSelectBackoffOp);
  }
  return false;
}

bool StaticShapeCluster::CanClusterableOp(const AnfNodePtr &node, const std::vector<PrimitivePtr> &op_list) {
  if (AnfUtils::IsGraphKernel(node)) {
    auto sub_graph = GetCNodeFuncGraph(node);
    if (auto type = sub_graph->get_attr("composite_type")) {
      if (GetValue<std::string>(type) == "inplace_assign_builder") {
        return false;
      }
    }
    return true;
  }
  if (GkUtils::IsKeepBasicNode(node)) {
    return false;
  }
  bool is_dvm = (GraphKernelFlags::GetInstance().kernel_generator == "DVM");
  if (!is_dvm && common::AnfAlgo::IsDynamicShape(node)) {
    return false;
  }
  bool node_in_oplist = std::any_of(op_list.begin(), op_list.end(),
                                    [&node](const PrimitivePtr &prim) { return IsPrimitiveCNode(node, prim); });
  if (!node_in_oplist) {
    return false;
  }

  auto cb = Callback::Instance();
  MS_EXCEPTION_IF_NULL(cb);
  // if node's output type is complex64 or complex128, cannot be added to the cluster list.
  if (IsComplexDataType(node)) {
    return false;
  }

  if (is_dvm && !DvmSupported(node)) {
    return false;
  }

  if (IsPrimitiveCNode(node, prim::kPrimReshape)) {
    auto output_format = cb->GetOutputFormat(node, 0);
    if (output_format != kOpFormat_DEFAULT) {
      auto primitive = GetCNodePrimitive(node);
      MS_EXCEPTION_IF_NULL(primitive);
      primitive = primitive->Clone();
      // format attr used by ReshapeOp::InferFormat
      primitive->AddAttr("format", MakeValue(output_format));
      auto cnode = node->cast<CNodePtr>();
      MS_EXCEPTION_IF_NULL(cnode);
      cnode->set_input(kAnfPrimitiveIndex, NewValueNode(primitive));
    }
  }
  if (!ValueDependOpUtils::IsConstInput(node)) {
    return false;
  }
  if (SkipHostInputNode(node, is_dvm)) {
    // this node can be fused with input host ops by kernelpacket
    return false;
  }
  if (GkUtils::InplaceWithViewInputs(node)) {
    return false;
  }
  if (is_dvm) {
    GkUtils::CheckOpLevel(node, clusterable_ops_with_level_dvm, OpLevel_1);
  }
  return !GkUtils::IsShapeZero(node);
}

bool StaticShapeCluster::IsClusterableOp(const AnfNodePtr &node) {
  return StaticShapeCluster::CanClusterableOp(node, op_list_);
}

std::vector<PrimitivePtr> DynamicShapeCluster::GetClusterableOpList() {
  std::vector<PrimitivePtr> dyn_clusterable_ops_list = {
    prim::kPrimAdd, prim::kPrimCast, prim::kPrimMul,  prim::kPrimRealDiv,   prim::kPrimSub,
    prim::kPrimAbs, prim::kPrimExp,  prim::kPrimLog,  prim::kPrimMaximum,   prim::kPrimMinimum,
    prim::kPrimNeg, prim::kPrimPow,  prim::kPrimSqrt, prim::kPrimTranspose, prim::kPrimReduceSum};
  return dyn_clusterable_ops_list;
}

bool DynamicShapeCluster::IsClusterableOp(const AnfNodePtr &node) {
  bool node_in_oplist = std::any_of(op_list_.begin(), op_list_.end(),
                                    [&node](const PrimitivePtr &prim) { return IsPrimitiveCNode(node, prim); });
  if (!node_in_oplist || common::AnfAlgo::IsDynamicRankNode(node)) {
    return false;
  }
  if (GkUtils::IsKeepBasicNode(node)) {
    return false;
  }
  if (!ValueDependOpUtils::IsConstInput(node)) {
    return false;
  }
  return true;
}

bool DynamicShapeCluster::Run(const FuncGraphPtr &func_graph) {
  auto mng = func_graph->manager();
  MS_EXCEPTION_IF_NULL(mng);
  Init(func_graph);
  bool changed = Process(func_graph);
  if (changed) {
    mng->RemoveRoots();
    mng->KeepRoots({func_graph});
  }
  Clean();
  return changed;
}
}  // namespace mindspore::graphkernel
