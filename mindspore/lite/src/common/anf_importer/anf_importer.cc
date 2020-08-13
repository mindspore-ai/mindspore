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

#include <utility>
#include <vector>
#include <string>
#include <memory>
#include "src/common/anf_importer/anf_importer.h"
#include "schema/model_generated.h"
#include "ir/dtype.h"
#include "ir/primitive.h"
#include "src/param_value_lite.h"
#include "frontend/operator/ops.h"
#include "abstract/abstract_value.h"
#include "src/ir/primitive_value.h"
#include "include/errorcode.h"
#include "schema/inner/model_generated.h"
namespace mindspore {
namespace lite {
#if 0
PrimitivePtr SetConv2DAttr(const schema::CNode *cNode) {
  MS_EXCEPTION_IF_NULL(cNode);
  auto attrs = cNode->primitive()->value_as_Conv2D();
  PrimitivePtr prim;
  if (attrs->group() > 1) {
    prim = std::make_shared<Primitive>("DepthwiseConv2D");
    prim->set_instance_name("DepthwiseConv2D");
  } else {
    prim = std::make_shared<Primitive>("Conv2D");
    prim->set_instance_name("Conv2D");
  }

  prim->set_attr("group", MakeValue<int>(attrs->group()));
  prim->set_attr("format", MakeValue<int>(attrs->format()));
  prim->set_attr("pad_mode", MakeValue<int>(attrs->padMode()));
  std::vector<int> pad_list = {attrs->padUp(), attrs->padDown(), attrs->padLeft(), attrs->padRight()};
  prim->set_attr("pad_list", MakeValue<std::vector<int>>(pad_list));
  std::vector<int> dilate = {attrs->dilateH(), attrs->dilateW()};
  prim->set_attr("dilation", MakeValue<std::vector<int>>(dilate));
  std::vector<int> kernel_size = {attrs->kernelH(), attrs->kernelW()};
  prim->set_attr("kernel_size", MakeValue<std::vector<int>>(kernel_size));
  std::vector<int> stride = {1, 1, attrs->strideH(), attrs->strideW()};
  prim->set_attr("stride", MakeValue<std::vector<int>>(stride));
  prim->set_attr("out_channel", MakeValue<int>(attrs->channelOut()));
  prim->set_attr("group", MakeValue<int>(attrs->group()));
  return prim;
}

PrimitivePtr SetActivationAttr(const schema::CNode *cNode) {
  MS_EXCEPTION_IF_NULL(cNode);
  auto attrs = cNode->primitive()->value_as_Activation();
  PrimitivePtr prim;
  if (attrs->type() == schema::ActivationType_RELU) {
    prim = std::make_shared<Primitive>("ReLU");
    prim->set_instance_name("ReLU");
  }
  return prim;
}

PrimitivePtr SetPoolingAttr(const schema::CNode *cNode) {
  MS_EXCEPTION_IF_NULL(cNode);
  auto attrs = cNode->primitive()->value_as_Pooling();
  PrimitivePtr prim;
  if (attrs->poolingMode() == schema::PoolMode_MAX_POOLING) {
    prim = std::make_shared<Primitive>("MaxPool");
    prim->set_instance_name("MaxPool");
  } else if (attrs->poolingMode() == schema::PoolMode_MEAN_POOLING) {
    prim = std::make_shared<Primitive>("MeanPool");
    prim->set_instance_name("MeanPool");
  }

  prim->set_attr("format", MakeValue<int>(attrs->format()));
  prim->set_attr("pad_mode", MakeValue<int>(attrs->padMode()));
  prim->set_attr("ksize", MakeValue<std::vector<int>>(std::vector<int>({1, 1, attrs->windowH(), attrs->windowW()})));
  prim->set_attr("strides", MakeValue<std::vector<int>>(std::vector<int>({1, 1, attrs->strideH(), attrs->strideW()})));
  return prim;
}

PrimitivePtr SetFlattenAttr(const schema::CNode *cNode) {
  MS_EXCEPTION_IF_NULL(cNode);
  auto prim = std::make_shared<Primitive>("Flatten");
  prim->set_instance_name("Flatten");
  return prim;
}

PrimitivePtr SetMatmulAttr(const schema::CNode *cNode) {
  MS_EXCEPTION_IF_NULL(cNode);
  auto attrs = cNode->primitive()->value_as_MatMul();
  auto prim = std::make_shared<Primitive>("Matmul");
  prim->set_instance_name("Matmul");
  prim->set_attr("transpose_a", MakeValue<int>(attrs->transposeA()));
  prim->set_attr("transpose_b", MakeValue<int>(attrs->transposeB()));
  return prim;
}

PrimitivePtr SetMulAttr(const schema::CNode *cNode) {
  MS_EXCEPTION_IF_NULL(cNode);
  //  auto attrs = nodedef->attr_as_Mul();
  auto prim = std::make_shared<Primitive>("Mul");
  prim->set_instance_name("Mul");
  return prim;
}

PrimitivePtr SetSigmoidAttr(const schema::CNode *cNode) {
  MS_EXCEPTION_IF_NULL(cNode);
  auto prim = std::make_shared<Primitive>("Sigmoid");
  prim->set_instance_name("Sigmoid");
  return prim;
}

PrimitivePtr SetReduceAttr(const schema::CNode *cNode) {
  MS_EXCEPTION_IF_NULL(cNode);
  auto prim = std::make_shared<Primitive>("ReduceMean");
  prim->set_instance_name("ReduceMean");
  return prim;
}

PrimitivePtr SetBatchNormAttr(const schema::CNode *cNode) {
  MS_EXCEPTION_IF_NULL(cNode);
  auto attrs = cNode->primitive_as_BatchNorm();
  auto prim = std::make_shared<Primitive>("BatchNorm");
  prim->set_attr("is_training", MakeValue<bool>(attrs->is_training()));
  prim->set_instance_name("BatchNorm");
  return prim;
}

PrimitivePtr SetBiasAddAttr(const schema::CNode *cNode) {
  MS_EXCEPTION_IF_NULL(cNode);
  auto prim = std::make_shared<Primitive>("BiasAdd");
  prim->set_instance_name("BiasAdd");
  return prim;
}

PrimitivePtr SetAddAttr(const schema::CNode *cNode) {
  MS_EXCEPTION_IF_NULL(cNode);
  auto prim = std::make_shared<Primitive>("Add");
  prim->set_instance_name("Add");
  return prim;
}

void MinnieBuildGraph::FbTest(const GraphDef *graph_def) {
  auto node_def = graph_def->subgraphs()->begin()->nodes()->GetAs<OpDef>(3);
  PrimitivePtr prim = ConverterOperatorAttr(node_def);
  if (prim->GetAttr("format")) MS_LOG(INFO) << "find format";
  if (prim->GetAttr("group")) MS_LOG(INFO) << "find group";
}
#endif

int AnfImporter::Import(const schema::QuantType &quantType) {
  ConverterConstTensor();
  auto ret = ConverterCNode();
  if (RET_OK != ret) {
    MS_LOG(ERROR) << "ConverterCNode failed " << ret;
    return ret;
  }
  AddReturnCNode();
  return RET_OK;
}

AnfNodePtr AnfImporter::GetNode(int tensor_id) {
  auto n = nodes_.find(tensor_id);
  if (n == nodes_.end()) {
    return nullptr;
  }
  return n->second;
}

void AnfImporter::AddNode(int tensor_id, AnfNodePtr node) { nodes_[tensor_id] = std::move(node); }
}  // namespace lite
}  // namespace mindspore

