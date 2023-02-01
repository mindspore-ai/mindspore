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
#include "plugin/device/ascend/optimizer/ir_fission/topk_split.h"

#include <string>
#include <utility>
#include <vector>
#include <memory>
#include <set>
#include "utils/hash_set.h"
#include "backend/common/optimizer/const_input_to_attr.h"
#include "kernel/kernel_build_info.h"
#include "include/common/utils/utils.h"
#include "backend/common/session/kernel_graph.h"
#include "backend/common/session/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "runtime/device/kernel_info.h"
#include "utils/ms_context.h"
#include "plugin/device/ascend/optimizer/optimizer_factory.h"
#include "common/util/platform_info.h"

namespace mindspore::opt {
namespace {
constexpr size_t kMultiply2 = 2;
constexpr size_t kTopkIndexK = 1;
constexpr auto kAttrSorted = "sorted";
constexpr auto r_assist_const = "assist_const";
constexpr auto r_new_value = "new_value";
constexpr auto r_topk = "r_topk";
constexpr auto m_topk = "m_topk";
constexpr auto x1 = "x1";
constexpr auto x2 = "x2";

tensor::TensorPtr ConstructAssistTensor(size_t assist_len, bool is_segment_sort = false, bool is_int32 = false) {
  // create tensor
  int64_t shape_len = is_segment_sort ? SizeToLong(assist_len) : SizeToLong(assist_len * kMultiply2);
  std::vector<int64_t> assist_shape{shape_len};
  auto dtype = is_int32 ? kInt32 : kFloat16;
  TensorTypePtr tensor_type = std::make_shared<TensorType>(dtype);
  tensor::DeviceInfo device_info{kOpFormat_DEFAULT, tensor_type};
  tensor::TensorPtr assist_tensor = std::make_shared<tensor::Tensor>(dtype->type_id(), assist_shape);
  assist_tensor->set_device_info(device_info);

  // set value of tensor
  auto data_ptr = assist_tensor->data_c();
  MS_EXCEPTION_IF_NULL(data_ptr);
  if (is_int32) {
    auto data = static_cast<int32_t *>(data_ptr);
    for (int32_t i = 0; i < SizeToInt(assist_len); ++i) {
      *data = i;
      ++data;
    }
  } else {
    auto data = static_cast<float16 *>(data_ptr);
    for (size_t i = 0; i < assist_len; ++i) {
      *data = float16(static_cast<float>(i));
      ++data;
    }
    if (!is_segment_sort) {
      for (size_t i = 0; i < assist_len; ++i) {
        auto gap = static_cast<int>(i) - static_cast<int>(float16(static_cast<float>(i)));
        *data = float16(static_cast<float>(gap));
        ++data;
      }
    }
  }

  return assist_tensor;
}

tensor::TensorPtr CreateAssistTensor(const std::vector<int64_t> &input_shape, int32_t k_num,
                                     const fe::PlatformInfo &platform_info, const fe::OptionalInfo &optional_info) {
  bool is_lhisi = optional_info.soc_version.find("Hi3796CV300CS") != std::string::npos ||
                  optional_info.soc_version.find("Hi3796CV300ES") != std::string::npos ||
                  optional_info.soc_version.find("SD3403") != std::string::npos;
  constexpr int64_t kLhisiMaxLastSize = 3000;
  constexpr int64_t kHisiMaxLastSize = 5000;
  constexpr int64_t kLhisiMaxKNum = 2048;
  constexpr int64_t kHisiMaxKNum = 4096;
  constexpr size_t kSmallSceneAssistLen = 4096;
  constexpr size_t kLargeSceneAssistLen = 2048;
  int64_t max_last_size = is_lhisi ? kLhisiMaxLastSize : kHisiMaxLastSize;
  int64_t max_k_num = is_lhisi ? kLhisiMaxKNum : kHisiMaxKNum;
  if (input_shape.back() > max_last_size || k_num > max_k_num) {
    if (platform_info.str_info.short_soc_version == "Ascend910B" ||
        platform_info.str_info.short_soc_version == "Ascend310B") {
      return ConstructAssistTensor(kLargeSceneAssistLen, true, true);
    } else {
      return ConstructAssistTensor(kLargeSceneAssistLen, true);
    }
  }
  return ConstructAssistTensor(kSmallSceneAssistLen);
}

ValueNodePtr CreateAssistNode(const std::vector<int64_t> &input_shape, int32_t k_num,
                              const fe::PlatformInfo &platform_info, const fe::OptionalInfo &optional_info) {
  tensor::TensorPtr assist_tensor = CreateAssistTensor(input_shape, k_num, platform_info, optional_info);
  MS_EXCEPTION_IF_NULL(assist_tensor);
  auto assist_const = std::make_shared<ValueNode>(assist_tensor);
  auto assist_abstract = assist_tensor->ToAbstract();
  assist_const->set_abstract(assist_abstract);
  auto assist_kernel_info = std::make_shared<device::KernelInfo>();
  assist_const->set_kernel_info(assist_kernel_info);
  kernel::KernelBuildInfo::KernelBuildInfoBuilder builder1;
  builder1.SetOutputsFormat({kOpFormat_DEFAULT});
  builder1.SetOutputsDeviceType({common::AnfAlgo::GetOutputInferDataType(assist_const, 0)});
  builder1.SetOutputsKernelObjectType({kernel::KernelObjectType::TENSOR});
  AnfAlgo::SetSelectKernelBuildInfo(builder1.Build(), assist_const.get());
  return assist_const;
}

kernel::KernelBuildInfoPtr CreateKernelBuildInfo() {
  kernel::KernelBuildInfo::KernelBuildInfoBuilder builder;
  builder.SetKernelType(TBE_KERNEL);
  builder.SetFusionType(kernel::kPatternOpaque);
  builder.SetProcessor(kernel::AICORE);
  builder.SetInputsFormat({kOpFormat_DEFAULT, kOpFormat_DEFAULT});
  builder.SetOutputsFormat({kOpFormat_DEFAULT, kOpFormat_DEFAULT});
  builder.SetInputsDeviceType({kNumberTypeFloat16, kNumberTypeFloat16});
  builder.SetOutputsDeviceType({kNumberTypeFloat16, kNumberTypeInt32});
  builder.SetInputsKernelObjectType({kernel::KernelObjectType::TENSOR});
  builder.SetOutputsKernelObjectType({kernel::KernelObjectType::TENSOR});
  return builder.Build();
}

bool CheckInputNamesSize(const CNodePtr &cnode) {
  auto input_names_vec = common::AnfAlgo::GetNodeAttr<std::vector<std::string>>(cnode, kAttrInputNames);
  if (input_names_vec.size() < kTopkIndexK + 1) {
    MS_LOG(INFO) << "The input k of topk has been converted to attr";
    return false;
  }
  return true;
}

bool CheckInputShape(const AnfNodePtr &node) {
  auto shape = common::AnfAlgo::GetPrevNodeOutputInferShape(node, 0);
  if (shape.empty()) {
    MS_LOG(INFO) << "The input shape of topk to split must not be empty";
    return false;
  }
  auto last_dim = shape.back();
  const int64_t kMaxFloat16 = 65500;
  if (last_dim > kMaxFloat16) {
    MS_LOG(INFO) << "The last dim is more than " << kMaxFloat16 << ", switch to aicpu ops.";
    return false;
  }
  return true;
}

bool CheckInputType(const AnfNodePtr &node) {
  auto dtype = common::AnfAlgo::GetPrevNodeOutputInferDataType(node, 0);
  const std::set<TypeId> aicore_supported_types = {kNumberTypeFloat16, kNumberTypeFloat32, kNumberTypeFloat};
  if (aicore_supported_types.find(dtype) == aicore_supported_types.end()) {
    MS_LOG(INFO) << "The input data type of topk to split must be float";
    return false;
  }
  return true;
}

bool CheckFusion(const CNodePtr &node) {
  if (common::AnfAlgo::HasNodeAttr(kAttrSorted, node) && !common::AnfAlgo::GetNodeAttr<bool>(node, kAttrSorted)) {
    return false;
  }
  if (!CheckInputNamesSize(node)) {
    return false;
  }
  if (!CheckInputShape(node)) {
    return false;
  }
  if (!CheckInputType(node)) {
    return false;
  }
  return true;
}
struct State {
  int k_num;
};
using StatePtr = std::shared_ptr<State>;
}  // namespace

bool TopKSplit::CheckMatchedDAG(const PatternMap &, const FuncGraphPtr &graph, const AnfNodePtr &node) const {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(node);
  if (common::AnfAlgo::IsDynamicShape(node)) {
    return false;
  }
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  if (!CheckFusion(cnode)) {
    return false;
  }
  // Convert the tensor input to scalar and convert it to attr
  auto input_k = cnode->input(kTopkIndexK + 1);
  MS_EXCEPTION_IF_NULL(input_k);
  if (!IsValueNode<tensor::Tensor>(input_k)) {
    return false;
  }
  fe::PlatformInfo platform_info;
  fe::OptionalInfo optional_info;
  if (fe::PlatformInfoManager::Instance().GetPlatformInfoWithOutSocVersion(platform_info, optional_info) != 0) {
    MS_LOG(WARNING) << "Get platform info failed, quit fusion.";
    return false;
  }
  return true;
}

class BuildAssistConst {
 public:
  explicit BuildAssistConst(StatePtr s_) : s(std::move(s_)) {}
  AnfNodePtr operator()(const PatternMap &m) {
    fe::PlatformInfo platform_info;
    fe::OptionalInfo optional_info;
    if (fe::PlatformInfoManager::Instance().GetPlatformInfoWithOutSocVersion(platform_info, optional_info) != 0) {
      MS_LOG(EXCEPTION) << "Get platform info failed in BuildAssistConst.";
    }
    auto cnode = m.Get(m_topk)->cast<CNodePtr>();
    auto input_shape = common::AnfAlgo::GetPrevNodeOutputInferShape(cnode, 0);
    auto input_k = cnode->input(kTopkIndexK + 1);
    ValuePtr value = GetValueNode(input_k);
    MS_EXCEPTION_IF_NULL(value);
    auto tensor = value->cast<tensor::TensorPtr>();
    MS_EXCEPTION_IF_NULL(tensor);
    auto *data = static_cast<int32_t *>(tensor->data_c());
    MS_EXCEPTION_IF_NULL(data);
    int32_t k_num = *data;
    s->k_num = k_num;
    auto assist_const = CreateAssistNode(input_shape, k_num, platform_info, optional_info);
    return assist_const;
  }

 private:
  StatePtr s;
};

class BuildNewValue {
 public:
  explicit BuildNewValue(StatePtr s_) : s(std::move(s_)) {}
  AnfNodePtr operator()(const PatternMap &) { return std::make_shared<ValueNode>(MakeValue(s->k_num)); }

 private:
  StatePtr s;
};

AnfNodePtr BuildTopk(const PatternMap &m, const AnfNodePtr &default_node) {
  auto g = default_node->func_graph();
  auto kernel_graph = g->cast<KernelGraphPtr>();
  auto cnode = m.Get(m_topk)->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  MS_EXCEPTION_IF_NULL(kernel_graph);

  std::vector<AnfNodePtr> new_inputs{NewValueNode(std::make_shared<Primitive>(kTopKOpName))};
  (void)new_inputs.insert(new_inputs.cend(), cnode->inputs().cbegin() + 1, cnode->inputs().cend());
  CNodePtr new_cnode = NewCNode(new_inputs, g);
  MS_EXCEPTION_IF_NULL(new_cnode);
  new_cnode->set_abstract(cnode->abstract());
  new_cnode->set_scope(cnode->scope());
  common::AnfAlgo::CopyNodeAttrs(cnode, new_cnode);
  CheckCNodeInputSize(new_cnode, kTopkInputTensorNum);

  new_cnode->set_input(kTopkIndexK + 1, m.Get(r_new_value));
  mindspore::HashSet<size_t> attr_index{kTopkIndexK};
  new_cnode = ConstInputToAttr(new_cnode, attr_index);
  new_cnode->add_input(m.Get(r_assist_const));

  if (!CheckAICoreSupportedSpec(new_cnode, CreateKernelBuildInfo())) {
    MS_LOG(INFO) << "Split topk failed, check to aicpu.";
    return nullptr;
  }
  if (kernel_graph != nullptr) {
    MS_LOG(INFO) << "Split topk success. use tbe aicore.";
    kernel_graph->AddValueNodeToGraph(m.Get(r_assist_const)->cast<ValueNodePtr>());
  }

  return new_cnode;
}

void TopKSplit::DefineSrcPattern(SrcPattern *src_pattern) {
  (void)(*src_pattern).AddVar(x1).AddVar(x2).AddCNode(m_topk, {std::make_shared<Primitive>(kTopKOpName), x1, x2});
}

void TopKSplit::DefineDstPattern(DstPattern *dst_pattern) {
  StatePtr s = std::make_shared<State>();
  (void)(*dst_pattern)
    .AddValueNode(r_assist_const, BuildAssistConst(s))
    .AddValueNode(r_new_value, BuildNewValue(s))
    .AddCNode(r_topk, {std::make_shared<Primitive>(kTopKOpName), x1, r_assist_const}, BuildTopk);
}

MS_PASS_FACTORY_REG(PatternToPatternPass, topk_split_fission, TopKSplit, kIRFusionFissionPass);
}  // namespace mindspore::opt
