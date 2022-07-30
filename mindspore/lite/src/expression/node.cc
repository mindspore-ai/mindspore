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

#include <algorithm>
#include <utility>
#include <functional>
#include "src/expression/node.h"
#include "src/expression/ops.h"
#include "src/expression/export.h"
#include "src/litert/infer_manager.h"
#include "src/common/utils.h"
#include "src/litert/cxx_api/expression/net_impl.h"

namespace mindspore {
namespace lite {
int Node::name_id;

std::vector<EXPR *> Node::construct(const std::vector<EXPR *> &inputs) {
  if (inputs.size() >= expr()->params().size()) {
    expr()->set_params(inputs);
  } else {
    for (std::size_t i = 0; i < inputs.size(); i++) {
      expr()->set_params(i, inputs[i]);
    }
  }
  auto ret = InferShape();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "error infershape for node " << name();
    return {};
  }
  std::vector<EXPR *> res(expr_.size());
  (void)std::transform(expr_.begin(), expr_.end(), res.begin(), [](const EXPR &e) { return const_cast<EXPR *>(&e); });
  return res;
}

std::vector<EXPR *> Node::Grad(EXPR *expr) {
  MS_LOG(ERROR) << name() << " (" << schema::EnumNamePrimitiveType(primitive()) << ") does not have grad defined";
  return {};
}

int Node::CreateTensorFromExpr(const std::vector<EXPR *> &expr, std::vector<Tensor *> *tensors, bool is_input) {
  MS_ASSERT(tensors != nullptr);
  int ret = RET_OK;
  for (auto e : expr) {
    // Tensor -> TensorC
    if (is_input && e->node()->primitive() == schema::PrimitiveType_Depend) {
      continue;
    }
    auto type = (e->node()->primitive() != schema::PrimitiveType_NONE) ? Category::VAR : Category::CONST_TENSOR;
    auto t = std::make_unique<Tensor>(e->data_type(), e->dims(), (mindspore::Format)e->format(), type);
    if (t == nullptr) {
      ret = RET_NULL_PTR;
      break;
    }
    // copy data if any
    if (type == Category::CONST_TENSOR) {
      void *dst = t->MutableData();
      if (dst == nullptr) {
        ret = RET_NULL_PTR;
        break;
      }
      if (e->node()->data() && (e->node()->data()->data().size() > 0)) {
        uint8_t *src = e->node()->data()->data().data();
        memcpy(dst, src, t->Size());
      }
    }
    tensors->push_back(t.release());
  }
  return ret;
}

void Node::FreeAllTensors(std::vector<Tensor *> *tensors) {
  MS_ASSERT(tensors != nullptr);
  for (auto &t : *tensors) {
    delete t;
  }
  tensors->clear();
}

int Node::InferShape() {
  auto ret = RET_OK;
  std::vector<Tensor *> in_tensors;
  std::vector<Tensor *> out_tensors;
  // build in \ out tensors
  ret = CreateTensorFromExpr(expr()->params(), &in_tensors, true);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Failed in create in tensors";
    FreeAllTensors(&in_tensors);
    return RET_ERROR;
  }
  std::vector<EXPR *> expr(expr_.size());
  (void)std::transform(expr_.begin(), expr_.end(), expr.begin(), [](const EXPR &e) { return const_cast<EXPR *>(&e); });
  ret = CreateTensorFromExpr(expr, &out_tensors);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Failed in create out tensors";
    FreeAllTensors(&in_tensors);
    FreeAllTensors(&out_tensors);
    return RET_ERROR;
  }
  // Do infer Shape
  ret = KernelInferShape(in_tensors, out_tensors, OpParam());
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "failed in infer shape for " << name();
    FreeAllTensors(&in_tensors);
    FreeAllTensors(&out_tensors);
    return RET_ERROR;
  }
  // copy infer shape into expr
  for (uint32_t i = 0; i < expr_.size(); i++) {
    auto e = &expr_.at(i);
    auto o = out_tensors.at(i);
    e->set_format((o->format()));
    e->set_data_type(o->data_type());
    e->SetDims(o->shape());
  }
  // cleanup
  FreeAllTensors(&in_tensors);
  FreeAllTensors(&out_tensors);

  return ret;
}

EXPR *Node::CreateWeights(std::vector<int> dims, TypeId data_type, int format, Param::Mode mode, std::string name) {
  auto weights = new (std::nothrow) InputM(dims);
  if (weights == nullptr) {
    MS_LOG(ERROR) << "Cannot allocate weights";
    return nullptr;
  }
  weights->set_name(this->name() + "/" + name);
  int size = std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<int>());
  weights->data()->SetSize(size);
  weights->data()->Fill(mode);
  PushOp(weights);
  return weights->expr();
}

Node *Node::CreateConstTensor(int index, std::vector<int> dims, TypeId data_type, int format, std::string name,
                              const void *data) {
  auto tensor = NN::Input(dims, data_type, format);
  int elem_size = DataTypeSize(data_type);
  tensor->set_name(this->name() + "/" + name);
  int size = std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<int>()) * elem_size;
  tensor->data()->SetSize(size);
  tensor->data()->Copy(reinterpret_cast<const uint8_t *>(data), size);
  expr()->set_params(index, tensor->expr());
  PushOp(tensor);
  return tensor;
}

int Node::MakeEntry(ExportSession *session) {
  std::vector<uint32_t> input_idx;
  std::vector<uint32_t> output_idx;
  std::vector<uint8_t> empty;
  if (primitive() == schema::PrimitiveType_Depend) return RET_OK;
  // create node input
  size_t inputs = InputsNum();
  for (size_t i = 0; i < inputs; i++) {
    EXPR *ex = expr()->GetInput(i);
    if (ex->node()->primitive() == schema::PrimitiveType_Depend) continue;
    uint32_t id = session->GetOutput(ex);
    input_idx.push_back(id);
  }
  size_t outputs = OutputsNum();
  size_t last_id = session->meta_graph()->allTensors.size();
  int type = (primitive() == schema::PrimitiveType_NONE) ? NodeType_ValueNode : NodeType_CNode;
  auto data = (type == NodeType_ValueNode) ? this->data()->data() : empty;
  if (data.empty()) type = NodeType_CNode;  // input is Cnode !!?
  int idx = 0;
  for (size_t i = 0; i < outputs; i++) {
    if (session->IsToDependOnly(expr(i))) continue;
    output_idx.push_back(last_id + idx);
    session->UpdateOutput(expr(i), last_id + idx);
    auto odims = dims(i);
    auto data_type = expr(i)->data_type();
    auto format = expr(i)->format();
    std::string footer = (i > 0) ? ("-" + std::to_string(i)) : "";
    auto otensor = CreateTensor(name() + footer, type, data_type, odims, format, data);
    std::cout << "tensor -" << last_id + idx << ": " << name() + footer << std::endl;
    idx++;
    session->meta_graph()->allTensors.emplace_back(std::move(otensor));
  }
  if (primitive() != schema::PrimitiveType_NONE) {
    if (output_idx.size() == 0) {
      return RET_OK;
    }
    auto cnode = CreateCNode(input_idx, output_idx);

    auto ret = UnPopulate(cnode);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "failed to populate cnode";
      return RET_ERROR;
    }
    session->meta_graph()->nodes.emplace_back(std::move(cnode));
  }

  return RET_OK;
}

std::unique_ptr<schema::CNodeT> Node::CreateCNode(std::vector<uint32_t> inputIndex, std::vector<uint32_t> outputIndex) {
  auto cnode = std::make_unique<schema::CNodeT>();
  cnode->primitive = std::make_unique<schema::PrimitiveT>();
  cnode->primitive->value.type = primitive();
  cnode->name = name();
  cnode->inputIndex = inputIndex;
  cnode->outputIndex = outputIndex;
  return cnode;
}

int Node::UnPopulate(const std::unique_ptr<schema::CNodeT> &cnode) {
  MS_LOG(ERROR) << "Node " << schema::EnumNamePrimitiveType(primitive()) << " cannot be exported";
  return RET_ERROR;
}

std::unique_ptr<mindspore::schema::TensorT> Node::CreateTensor(std::string name, int type, int data_type,
                                                               const std::vector<int32_t> dims, int format,
                                                               const std::vector<uint8_t> &data) {
  auto tensorT = std::make_unique<mindspore::schema::TensorT>();
  tensorT->nodeType = type;
  tensorT->dims = dims;
  tensorT->format = static_cast<schema::Format>(format);
  tensorT->name = name;
  tensorT->refCount = 0;
  tensorT->offset = 0;
  tensorT->dataType = data_type;
  tensorT->data = data;
  tensorT->enableHuffmanCode = false;
  if (tensorT->nodeType == mindspore::lite::NodeType_ValueNode) {
    tensorT->data = data;
  }
  return tensorT;
}

int Node::SetOutputs(int num) {
  EXPR e(this);
  e.SetSize(0);
  for (auto i = expr_.size(); i < static_cast<size_t>(num); i++) {
    expr_.emplace_back(e);
  }
  return RET_OK;
}

Node::~Node() {
  for (auto &op : ops_) {
    delete op;
  }
  ops_.clear();
  if (impl_ != nullptr) {
    impl_->set_node(nullptr);
    auto pnode = impl_->pnode();
    if (pnode != nullptr) {
      impl_->set_pnode(nullptr);
      delete pnode;
    }
  }
  impl_ = nullptr;
}
}  // namespace lite
}  // namespace mindspore
