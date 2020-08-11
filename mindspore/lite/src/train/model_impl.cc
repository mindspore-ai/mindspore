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

#include <string>
#include "src/train/model_impl.h"
#include "ir/func_graph.h"
#include "schema/model_generated.h"
#include "src/common/anf_importer/import_from_meta_graph.h"

namespace mindspore::lite::train {

std::shared_ptr<ModelImpl> ModelImpl::Import(const char *model_buf, size_t size) {
  MS_EXCEPTION_IF_NULL(model_buf);
  flatbuffers::Verifier verify((const uint8_t *)model_buf, size);
  if (!schema::VerifyMetaGraphBuffer(verify)) {
    MS_LOG(ERROR) << "The buffer is invalid and fail to create graph.";
    return nullptr;
  }
  // todo hangangqiang remove when copy primitive done
  auto *inner_buf = new char[size];
  memcpy(inner_buf, model_buf, size);
  auto meta_graph = schema::GetMetaGraph(inner_buf);
  auto func_graph_model = std::make_shared<ModelImpl>(meta_graph);
  auto ret = func_graph_model->BuildOps();
  if (0 != ret) {
    MS_LOG(ERROR) << "BuildOps failed";
    return nullptr;
  }
  AnfImporterFromMetaGraph anfImporter(func_graph_model);
  anfImporter.Import();
  return func_graph_model;
}

const lite::Primitive *ModelImpl::GetOp(const std::string &name) const {
  auto iter = ops.find(name);
  if (iter == ops.end()) {
    return nullptr;
  } else {
    return iter->second;
  }
}

void ModelImpl::FreeMetaGraph() { delete this->meta_graph; }

const schema::MetaGraph *ModelImpl::GetMetaGraph() const { return this->meta_graph; }

lite::Primitive *ModelImpl::CopyPrimitive(const schema::Primitive *srcPrim) {
  MS_EXCEPTION_IF_NULL(srcPrim);
  auto op_type = srcPrim->value_type();
  switch (op_type) {
    case schema::PrimitiveType_SoftMax:
      return new lite::SoftMax(const_cast<schema::Primitive *>(srcPrim));
    case schema::PrimitiveType_Activation:
      return new lite::Activation(const_cast<schema::Primitive *>(srcPrim));
    case schema::PrimitiveType_Conv2D:
      return new lite::Conv2D(const_cast<schema::Primitive *>(srcPrim));
    case schema::PrimitiveType_Reduce:
      return new lite::Reduce(const_cast<schema::Primitive *>(srcPrim));
    case schema::PrimitiveType_Pooling:
      return new lite::Pooling(const_cast<schema::Primitive *>(srcPrim));
    case schema::PrimitiveType_DepthwiseConv2D:
      return new lite::DepthwiseConv2D(const_cast<schema::Primitive *>(srcPrim));
    case schema::PrimitiveType_FusedBatchNorm:
      return new lite::FusedBatchNorm(const_cast<schema::Primitive *>(srcPrim));
    case schema::PrimitiveType_CaffeBatchNorm:
      return new lite::CaffeBatchNorm(const_cast<schema::Primitive *>(srcPrim));
    case schema::PrimitiveType_FullConnection:
      return new lite::FullConnection(const_cast<schema::Primitive *>(srcPrim));
    case schema::PrimitiveType_Power:
      return new lite::Power(const_cast<schema::Primitive *>(srcPrim));
    case schema::PrimitiveType_Range:
      return new lite::Range(const_cast<schema::Primitive *>(srcPrim));
    case schema::PrimitiveType_Mul:
      return new lite::Mul(const_cast<schema::Primitive *>(srcPrim));
    case schema::PrimitiveType_Add:
      return new lite::Add(const_cast<schema::Primitive *>(srcPrim));
    case schema::PrimitiveType_Sub:
      return new lite::Sub(const_cast<schema::Primitive *>(srcPrim));
    case schema::PrimitiveType_Div:
      return new lite::Div(const_cast<schema::Primitive *>(srcPrim));
    case schema::PrimitiveType_BiasAdd:
      return new lite::BiasAdd(const_cast<schema::Primitive *>(srcPrim));
    case schema::PrimitiveType_ExpandDims:
      return new lite::ExpandDims(const_cast<schema::Primitive *>(srcPrim));
    case schema::PrimitiveType_ArgMax:
      return new lite::ArgMax(const_cast<schema::Primitive *>(srcPrim));
    case schema::PrimitiveType_ArgMin:
      return new lite::ArgMin(const_cast<schema::Primitive *>(srcPrim));
    case schema::PrimitiveType_Cast:
      return new lite::Cast(const_cast<schema::Primitive *>(srcPrim));
    case schema::PrimitiveType_Reshape:
      return new lite::Reshape(const_cast<schema::Primitive *>(srcPrim));
    case schema::PrimitiveType_Scale:
      return new lite::Scale(const_cast<schema::Primitive *>(srcPrim));
    case schema::PrimitiveType_Eltwise:
      return new lite::Eltwise(const_cast<schema::Primitive *>(srcPrim));
    case schema::PrimitiveType_Ceil:
      return new lite::Ceil(const_cast<schema::Primitive *>(srcPrim));
    case schema::PrimitiveType_Concat:
      return new lite::Concat(const_cast<schema::Primitive *>(srcPrim));
    case schema::PrimitiveType_Fill:
      return new lite::Fill(const_cast<schema::Primitive *>(srcPrim));
    case schema::PrimitiveType_Transpose:
      return new lite::Transpose(const_cast<schema::Primitive *>(srcPrim));
    case schema::PrimitiveType_Slice:
      return new lite::Slice(const_cast<schema::Primitive *>(srcPrim));
    case schema::PrimitiveType_Nchw2Nhwc:
      return new lite::Nchw2Nhwc(const_cast<schema::Primitive *>(srcPrim));
    case schema::PrimitiveType_Nhwc2Nchw:
      return new lite::Nhwc2Nchw(const_cast<schema::Primitive *>(srcPrim));
    case schema::PrimitiveType_MatMul:
      return new lite::MatMul(const_cast<schema::Primitive *>(srcPrim));
    default:
      break;
  }
  return nullptr;
}

int ModelImpl::BuildOps() {
  if (this->meta_graph == nullptr) {
    MS_LOG(ERROR) << "mete_graph is nullptr";
    return -1;
  }
  for (size_t i = 0; i < meta_graph->nodes()->size(); i++) {
    auto cNode = meta_graph->nodes()->GetAs<schema::CNode>(i);
    auto name = cNode->name()->str();
    auto srcPrim = cNode->primitive();
    this->ops[name] = CopyPrimitive(srcPrim);
  }
  return 0;
}
}  // namespace mindspore::lite::train
