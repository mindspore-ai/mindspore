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
#include "kernel/akg/akg_kernel_attrs_process.h"

#include <algorithm>
#include "session/anf_runtime_algorithm.h"
#include "pre_activate/common/helper.h"

namespace mindspore {
namespace kernel {
void SetAkgAttrsForFour2Five(const AnfNodePtr &anf_node) {
  MS_EXCEPTION_IF_NULL(anf_node);
  // The x and output are akg op input and output param.
  std::vector<std::string> input_names = {"x"};
  std::vector<std::string> output_names = {"output"};
  AnfAlgo::SetNodeAttr("input_names", MakeValue(input_names), anf_node);
  AnfAlgo::SetNodeAttr("output_names", MakeValue(output_names), anf_node);

  TypeId dst_type_id = AnfAlgo::GetOutputDeviceDataType(anf_node, 0);
  std::string dst_type;
  if (dst_type_id == kFloat32->type_id()) {
    dst_type = "float32";
  } else if (dst_type_id == kFloat16->type_id()) {
    dst_type = "float16";
  }
  AnfAlgo::SetNodeAttr("dst_type", MakeValue(dst_type), anf_node);
}

void SetAkgAttrsForFive2Four(const AnfNodePtr &anf_node) {
  MS_EXCEPTION_IF_NULL(anf_node);
  std::vector<std::string> input_names = {"x"};
  std::vector<std::string> output_names = {"output"};
  AnfAlgo::SetNodeAttr("input_names", MakeValue(input_names), anf_node);
  AnfAlgo::SetNodeAttr("output_names", MakeValue(output_names), anf_node);
  std::vector<size_t> origin_shape = AnfAlgo::GetOutputInferShape(anf_node, 0);
  if (origin_shape.size() != kShape4dDims) {
    MS_LOG(EXCEPTION) << "The dim of origin_shape is not equal to 4, but it's dim is " << origin_shape.size() << ".";
  }
  std::vector<int> shape_transform;
  (void)std::transform(origin_shape.begin(), origin_shape.end(), std::back_inserter(shape_transform),
                       [](const int &origin_shape) { return static_cast<int>(origin_shape); });
  AnfAlgo::SetNodeAttr("shape4d", MakeValue(shape_transform), anf_node);
  AnfAlgo::SetNodeAttr("output_format", MakeValue(kOpFormat_NCHW), anf_node);

  TypeId dst_type_id = AnfAlgo::GetOutputDeviceDataType(anf_node, 0);
  std::string dst_type;
  if (dst_type_id == kFloat32->type_id()) {
    dst_type = "float32";
  } else if (dst_type_id == kFloat16->type_id()) {
    dst_type = "float16";
  }
  AnfAlgo::SetNodeAttr("dstType", MakeValue(dst_type), anf_node);
}

void SetAkgAttrsForCast(const AnfNodePtr &anf_node) {
  MS_EXCEPTION_IF_NULL(anf_node);
  // The x and output are akg op input and output param.
  std::vector<std::string> input_names = {"x", "dst_type"};
  std::vector<std::string> output_names = {"output"};
  AnfAlgo::SetNodeAttr(kAttrInputNames, MakeValue(input_names), anf_node);
  AnfAlgo::SetNodeAttr(kAttrOutputNames, MakeValue(output_names), anf_node);

  std::string dst_type;
  TypeId output_type = AnfAlgo::GetOutputDeviceDataType(anf_node, 0);
  if (output_type == kFloat32->type_id()) {
    dst_type = "float32";
  } else if (output_type == kFloat16->type_id()) {
    dst_type = "float16";
  } else if (output_type == kInt32->type_id()) {
    dst_type = "int32";
  } else {
    MS_LOG(WARNING) << "Unknown cast_to type: " << TypeIdToType(output_type)->ToString();
  }
  AnfAlgo::SetNodeAttr("dst_type", MakeValue(dst_type), anf_node);
}

void SetAkgAttrsForBNGrad1(const AnfNodePtr &anf_node) {
  MS_EXCEPTION_IF_NULL(anf_node);
  std::vector<std::string> input_names{"dy", "data", "mean"};
  std::vector<std::string> output_names{"dgamma_red_hw", "dbeta_red_hw", "data_minus_mean"};
  AnfAlgo::SetNodeAttr(kAttrInputNames, MakeValue(input_names), anf_node);
  AnfAlgo::SetNodeAttr(kAttrOutputNames, MakeValue(output_names), anf_node);
}

void SetAkgAttrsForBNGrad2(const AnfNodePtr &anf_node) {
  const size_t kBNGrad2InputSize = 5;
  MS_EXCEPTION_IF_NULL(anf_node);
  std::vector<std::string> input_names{"dgamma_red_hw", "dbeta_red_hw", "variance", "gamma"};
  std::vector<std::string> output_names{"bn_scale", "bn_bias", "rs", "dgamma_dx", "dbeta_dx"};
  AnfAlgo::SetNodeAttr(kAttrInputNames, MakeValue(input_names), anf_node);
  AnfAlgo::SetNodeAttr(kAttrOutputNames, MakeValue(output_names), anf_node);
  auto cnode = anf_node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  if (cnode->inputs().size() < kBNGrad2InputSize) {
    MS_LOG(EXCEPTION) << "The inputs size of BNGrad2 is less then " << kBNGrad2InputSize;
  }
  auto input1 = cnode->input(1);
  MS_EXCEPTION_IF_NULL(input1);
  auto tuple_getitem = input1->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(tuple_getitem);
  if (tuple_getitem->inputs().size() < kTupleGetItemInputSize) {
    MS_LOG(EXCEPTION) << "The inputs size of tuple_getitem is less then " << kTupleGetItemInputSize;
  }
  auto bn_grad1 = tuple_getitem->input(kRealInputNodeIndexInTupleGetItem);
  std::vector<size_t> data_shape = AnfAlgo::GetInputDeviceShape(bn_grad1, 0);
  AnfAlgo::SetNodeAttr(kAttrDataShape, MakeValue(opt::Convert2Int(data_shape)), anf_node);
}

void SetAkgAttrsForBNGrad3(const AnfNodePtr &anf_node) {
  MS_EXCEPTION_IF_NULL(anf_node);
  std::vector<std::string> input_names{"dy", "rs", "dgamma_dx", "dbeta_dx", "data_minus_mean"};
  std::vector<std::string> output_names{"dx"};
  AnfAlgo::SetNodeAttr(kAttrInputNames, MakeValue(input_names), anf_node);
  AnfAlgo::SetNodeAttr(kAttrOutputNames, MakeValue(output_names), anf_node);
}

void SetAkgAttrsForFusedBN1(const AnfNodePtr &anf_node) {
  MS_EXCEPTION_IF_NULL(anf_node);
  // Set attr for fused_bn1
  std::vector<std::string> fused_bn1_input_names{"data"};
  std::vector<std::string> fused_bn1_output_names{"mean", "var_part"};
  AnfAlgo::SetNodeAttr(kAttrInputNames, MakeValue(fused_bn1_input_names), anf_node);
  AnfAlgo::SetNodeAttr(kAttrOutputNames, MakeValue(fused_bn1_output_names), anf_node);
}

void SetAkgAttrsForFusedBN2(const AnfNodePtr &anf_node) {
  MS_EXCEPTION_IF_NULL(anf_node);
  // Set attr for fused_bn2
  std::vector<std::string> fused_bn2_input_names{"mean", "var_part", "running_mean", "running_var"};
  std::vector<std::string> fused_bn2_output_names{"variance", "running_mean", "running_variance"};
  AnfAlgo::SetNodeAttr(kAttrInputNames, MakeValue(fused_bn2_input_names), anf_node);
  AnfAlgo::SetNodeAttr(kAttrOutputNames, MakeValue(fused_bn2_output_names), anf_node);
}

void SetAkgAttrsForFusedBN3(const AnfNodePtr &anf_node) {
  MS_EXCEPTION_IF_NULL(anf_node);
  // Set attr for fused_bn3
  std::vector<std::string> fused_bn3_input_names{"data", "mean", "variance", "gamma", "beta"};
  std::vector<std::string> fused_bn3_output_names{"y"};
  AnfAlgo::SetNodeAttr(kAttrInputNames, MakeValue(fused_bn3_input_names), anf_node);
  AnfAlgo::SetNodeAttr(kAttrOutputNames, MakeValue(fused_bn3_output_names), anf_node);
}

void SetAkgAttrsForConvBN1(const AnfNodePtr &anf_node) {
  MS_EXCEPTION_IF_NULL(anf_node);
  std::vector<std::string> conv_bn1_output_names{"data", "var_part", "mean"};
  AnfAlgo::SetNodeAttr(kAttrOutputNames, MakeValue(conv_bn1_output_names), anf_node);
}

void SetAkgAttrsForBN2AddRelu(const AnfNodePtr &anf_node) {
  MS_EXCEPTION_IF_NULL(anf_node);
  std::vector<std::string> bn2_add_relu_input_names{"data",  "var_part", "mean",         "other_branch_data",
                                                    "gamma", "beta",     "running_mean", "running_var"};
  AnfAlgo::SetNodeAttr(kAttrInputNames, MakeValue(bn2_add_relu_input_names), anf_node);
  std::vector<std::string> bn2_add_relu_output_names{"output", "running_mean", "running_variance", "save_inv_variance"};
  AnfAlgo::SetNodeAttr(kAttrOutputNames, MakeValue(bn2_add_relu_output_names), anf_node);
}

void SetAkgAttrsForBN2Relu(const AnfNodePtr &anf_node) {
  MS_EXCEPTION_IF_NULL(anf_node);
  std::vector<std::string> bn2_input_names{"data", "var_part", "mean", "gamma", "beta", "running_mean", "running_var"};
  std::vector<std::string> bn2_output_names{"y", "running_mean", "running_variance", "save_inv_variance"};
  AnfAlgo::SetNodeAttr(kAttrInputNames, MakeValue(bn2_input_names), anf_node);
  AnfAlgo::SetNodeAttr(kAttrOutputNames, MakeValue(bn2_output_names), anf_node);
}
}  // namespace kernel
}  // namespace mindspore
