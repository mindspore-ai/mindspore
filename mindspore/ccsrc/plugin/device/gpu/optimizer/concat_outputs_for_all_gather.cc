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

#include "plugin/device/gpu/optimizer/concat_outputs_for_all_gather.h"
#include <string>
#include <tuple>
#include <utility>
#include <algorithm>
#include "backend/common/session/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"

namespace mindspore::opt {
namespace {
using OutputInfo =
  std::tuple<std::vector<TypeId>, std::vector<ShapeVector>, std::vector<std::string>, std::vector<TypeId>>;
OutputInfo GetNodeOutputInfo(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  std::vector<TypeId> output_infer_dtype;
  std::vector<ShapeVector> output_infer_shape;
  std::vector<std::string> output_format;
  std::vector<TypeId> output_device_dtype;
  auto type_ptr = node->Type();
  auto shape_ptr = node->Shape();
  size_t output_num = AnfAlgo::GetOutputTensorNum(node);
  auto kernel_info = dynamic_cast<device::KernelInfo *>(node->kernel_info());
  MS_EXCEPTION_IF_NULL(kernel_info);
  auto build_info = kernel_info->select_kernel_build_info();
  MS_EXCEPTION_IF_NULL(build_info);
  for (size_t i = 0; i < output_num; i++) {
    output_infer_dtype.emplace_back(common::AnfAlgo::GetOutputInferDataType(type_ptr, i));
    output_infer_shape.emplace_back(common::AnfAlgo::GetOutputInferShape(node, shape_ptr, i));
    output_format.emplace_back(build_info->GetOutputFormat(i));
    output_device_dtype.emplace_back(build_info->GetOutputDeviceType(i));
  }

  return {output_infer_dtype, output_infer_shape, output_format, output_device_dtype};
}

kernel::KernelBuildInfoPtr GenerateKernelBuildInfo(const AnfNodePtr &concat, const OutputInfo &allgather_output_info,
                                                   size_t allgather_input_num, size_t allgather_input_idx) {
  MS_EXCEPTION_IF_NULL(concat);
  std::vector<std::string> inputs_device_format;
  std::vector<std::string> outputs_device_format;
  std::vector<TypeId> inputs_device_type;
  std::vector<TypeId> outputs_device_type;
  kernel::KernelBuildInfo::KernelBuildInfoBuilder builder;
  size_t concat_input_num = common::AnfAlgo::GetInputTensorNum(concat);
  for (size_t i = 0; i < concat_input_num; ++i) {
    size_t input_index = allgather_input_idx + i * allgather_input_num;
    inputs_device_format.emplace_back(std::get<kIndex2>(allgather_output_info)[input_index]);
    inputs_device_type.emplace_back(std::get<kIndex3>(allgather_output_info)[input_index]);
  }
  // Current only support default format & float16
  auto cmp_format = inputs_device_format.begin();
  auto format_iter = std::find_if(inputs_device_format.begin(), inputs_device_format.end(),
                                  [&](const auto &format) { return format != (*cmp_format); });
  if (format_iter != inputs_device_format.end()) {
    MS_LOG(EXCEPTION) << "Input format is not same, value: " << (*format_iter) << ", need format: " << (*cmp_format);
  }
  auto cmp_dtype = inputs_device_type.begin();
  auto dtype_iter = std::find_if(inputs_device_type.begin(), inputs_device_type.end(),
                                 [&](const auto &dtype) { return dtype != (*cmp_dtype); });
  if (dtype_iter != inputs_device_type.end()) {
    MS_LOG(EXCEPTION) << "Input dtype is not same, value: " << TypeIdLabel(*dtype_iter)
                      << ", need dtype: " << TypeIdLabel(*cmp_dtype);
  }
  outputs_device_format.emplace_back(*cmp_format);
  outputs_device_type.emplace_back(*cmp_dtype);

  builder.SetFusionType(kernel::kPatternOpaque);
  builder.SetInputsFormat(inputs_device_format);
  builder.SetOutputsFormat(outputs_device_format);
  builder.SetInputsDeviceType(inputs_device_type);
  builder.SetOutputsDeviceType(outputs_device_type);
  return builder.Build();
}

AnfNodePtr InsertConcatForOutput(const FuncGraphPtr &func_graph, const AnfNodePtr &node, const OutputInfo &output_info,
                                 const std::vector<AnfNodePtr> &new_tuple_getitems, int64_t rank_size) {
  MS_EXCEPTION_IF_NULL(func_graph);
  std::vector<AnfNodePtr> make_tuple_inputs{NewValueNode(std::make_shared<Primitive>(prim::kPrimMakeTuple->name()))};
  size_t inputs_size = common::AnfAlgo::GetInputTensorNum(node);
  for (size_t i = 0; i < inputs_size; ++i) {
    std::vector<AnfNodePtr> concat_inputs{NewValueNode(std::make_shared<Primitive>(prim::kPrimConcat->name()))};
    for (size_t j = 0, idx = i; j < LongToSize(rank_size); ++j, idx += inputs_size) {
      concat_inputs.push_back(new_tuple_getitems[idx]);
    }
    auto concat = func_graph->NewCNode(concat_inputs);
    MS_EXCEPTION_IF_NULL(concat);
    MS_EXCEPTION_IF_NULL(new_tuple_getitems[i]);
    const std::vector<TypeId> &dtypes = {std::get<0>(output_info)[i]};
    auto shape = std::get<1>(output_info)[i];
    shape[0] *= LongToSize(rank_size);
    common::AnfAlgo::SetOutputInferTypeAndShape(dtypes, {shape}, concat.get());

    common::AnfAlgo::SetNodeAttr(kAttrAxis, MakeValue(static_cast<int64_t>(0)), concat);
    common::AnfAlgo::SetNodeAttr(kAttrInputNums, MakeValue(rank_size), concat);
    std::vector<int64_t> dyn_input_size{rank_size};
    common::AnfAlgo::SetNodeAttr(kAttrDynInputSizes, MakeValue(dyn_input_size), concat);
    auto kernel_build_info = GenerateKernelBuildInfo(concat, output_info, inputs_size, i);
    AnfAlgo::SetSelectKernelBuildInfo(kernel_build_info, concat.get());
    make_tuple_inputs.push_back(concat);
  }

  auto make_tuple = func_graph->NewCNode(make_tuple_inputs);
  return make_tuple;
}
}  // namespace

const BaseRef ConcatOutputsForAllGather::DefinePattern() const {
  VarPtr Xs = std::make_shared<SeqVar>();
  auto prim = std::make_shared<Primitive>(kAllGatherOpName);
  return VectorRef({prim, Xs});
}

const AnfNodePtr ConcatOutputsForAllGather::Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                                    const EquivPtr &) const {
  MS_EXCEPTION_IF_NULL(node);
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  if (!common::AnfAlgo::HasNodeAttr(kAttrFusion, cnode) || !common::AnfAlgo::HasNodeAttr(kAttrRankSize, cnode)) {
    return nullptr;
  }
  auto fusion = common::AnfAlgo::GetNodeAttr<int64_t>(cnode, kAttrFusion);
  if (fusion <= 0) {
    return nullptr;
  }
  if (common::AnfAlgo::HasNodeAttr("fused", cnode) || common::AnfAlgo::GetInputTensorNum(node) == 1) {
    return nullptr;
  }
  common::AnfAlgo::SetNodeAttr("fused", MakeValue(true), node);
  auto rank_size = common::AnfAlgo::GetNodeAttr<int64_t>(node, kAttrRankSize);
  std::vector<AnfNodePtr> new_outputs;
  OutputInfo output_info = GetNodeOutputInfo(node);
  size_t output_num = AnfAlgo::GetOutputTensorNum(node);
  for (size_t i = 0; i < output_num; ++i) {
    int64_t temp = SizeToLong(i);
    auto idx = NewValueNode(temp);
    MS_EXCEPTION_IF_NULL(idx);
    auto imm = std::make_shared<Int64Imm>(temp);
    auto abstract_scalar = std::make_shared<abstract::AbstractScalar>(imm);
    idx->set_abstract(abstract_scalar);
    auto tuple_getitem = func_graph->NewCNode({NewValueNode(prim::kPrimTupleGetItem), node, idx});
    MS_EXCEPTION_IF_NULL(tuple_getitem);
    auto shape = AnfAlgo::GetOutputDetailShape(node, i);
    common::AnfAlgo::SetOutputTypeAndDetailShape({std::get<0>(output_info)[i]}, {shape}, tuple_getitem.get());
    new_outputs.emplace_back(std::move(tuple_getitem));
  }
  return InsertConcatForOutput(func_graph, node, output_info, new_outputs, rank_size);
}
}  // namespace mindspore::opt
