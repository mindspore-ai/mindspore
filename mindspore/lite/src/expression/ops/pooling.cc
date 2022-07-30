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

#include "src/expression/ops/pooling.h"
#include "src/expression/ops.h"
#include "src/expression/import.h"
#include "src/litert/cxx_api/expression/node_impl.h"
#include "src/expression/ops/transpose.h"

namespace mindspore {
namespace lite {
PoolingM::PoolingM(const PoolingConfig &cfg) : Node() {
  auto op_param = calloc(1, sizeof(PoolingParameter));
  if (op_param == nullptr) {
    MS_LOG(ERROR) << " cannot allocate PoolingParameter";
    return;
  }
  SetOpParam(op_param);
  PoolingParameter *pool_param = reinterpret_cast<PoolingParameter *>(OpParam());

  pool_param->window_h_ = cfg.kernel_size_[0];
  pool_param->window_w_ = cfg.kernel_size_[1];
  pool_param->stride_h_ = cfg.stride_[0];
  pool_param->stride_w_ = cfg.stride_[1];
  auto pad_mode = GetMode(cfg.pad_mode_);
  if (pad_mode == -1) {
    MS_LOG(ERROR) << "bad pad mode";
    return;
  }
  pool_param->pad_mode_ = static_cast<PadMode>(pad_mode + Pad_pad);
  pool_param->round_mode_ = RoundMode_Floor;
  pool_param->act_type_ = ActType_No;
}

std::vector<EXPR *> PoolingM::construct(const std::vector<EXPR *> &inputs) {
  auto in = inputs;
  auto x = in.front();
  if (x->format() != NHWC && x->dims().size() == C4NUM) {
    x = TransposeM::TransposeCHW2HWC(x);
    x->node()->set_name(name() + "/" + x->node()->name());
    PushOp(x->node());
    in.at(0) = x;
  }
  auto y = Node::construct(in);
  return y;
}

int PoolingM::GetMode(std::string mode) {
  const std::vector<std::string> list = {"same", "valid"};
  auto itr = std::find(list.begin(), list.end(), mode);
  if (itr == list.end()) {
    MS_LOG(ERROR) << "illegal mode" << mode;
    return -1;
  }
  return std::distance(list.begin(), itr);
}

void PoolingM::UpdateRoundMode(const PoolingParameter *param, schema::RoundMode *round_mode) {
  switch (param->round_mode_) {
    case RoundMode_Floor:
      *round_mode = schema::RoundMode_FLOOR;
      break;
    case RoundMode_Ceil:
      *round_mode = schema::RoundMode_CEIL;
      break;
    default:
      *round_mode = schema::RoundMode_FLOOR;
      break;
  }
}

template <typename T>
int PoolingM::UnPopulate(const std::unique_ptr<schema::CNodeT> &cnode) {
  auto param = reinterpret_cast<const PoolingParameter *>(OpParam());
  auto prim = new (std::nothrow) T;
  if (prim == nullptr) {
    MS_LOG(ERROR) << "Cannot allocate primitive";
    return RET_ERROR;
  }
  prim->kernel_size = {param->window_h_, param->window_w_};
  prim->strides = {param->stride_h_, param->stride_w_};
  prim->pad = {param->pad_u_, param->pad_d_, param->pad_l_, param->pad_r_};
  prim->pad_mode = static_cast<schema::PadMode>(param->pad_mode_);
  UpdateRoundMode(param, &prim->round_mode);
  prim->global = param->global_;
  prim->activation_type = schema::ActivationType_NO_ACTIVATION;
  cnode->primitive->value.value = prim;
  return RET_OK;
}

template <typename T>
int PoolingM::UnPopulateGrad(const std::unique_ptr<schema::CNodeT> &cnode) {
  auto param = reinterpret_cast<const PoolingParameter *>(OpParam());
  auto prim = new (std::nothrow) T;
  if (prim == nullptr) {
    MS_LOG(ERROR) << "Cannot allocate primitive";
    return RET_ERROR;
  }
  prim->kernel_size = {param->window_h_, param->window_w_};
  prim->strides = {param->stride_h_, param->stride_w_};
  prim->pad_mode = static_cast<schema::PadMode>(param->pad_mode_);
  cnode->primitive->value.value = prim;
  return RET_OK;
}

// Max pooling Definition
MaxPoolM::MaxPoolM(const PoolingConfig &cfg) : PoolingM(cfg) {
  auto param = reinterpret_cast<PoolingParameter *>(OpParam());
  param->pool_mode_ = PoolMode_MaxPool;
  set_primitive(schema::PrimitiveType_MaxPoolFusion);
  set_name(UniqueName("MaxPool"));
}

int MaxPoolM::UnPopulate(const std::unique_ptr<schema::CNodeT> &cnode) {
  return PoolingM::UnPopulate<schema::MaxPoolFusionT>(cnode);
}

std::vector<EXPR *> MaxPoolM::Grad(EXPR *yt) {
  auto in = yt;
  if (yt->format() != NHWC && yt->dims().size() == C4NUM) {
    in = TransposeM::TransposeCHW2HWC(yt);
    in->node()->set_name(kGradName + "/" + name() + "/" + in->node()->name());
    PushOp(in->node());
  }
  auto pool_grad = new (std::nothrow) MaxPoolGradM(this);
  PushOp(pool_grad);
  return (*pool_grad)({input(0), output(0), in});
}

static ImportReg maxPoolReg(schema::PrimitiveType_MaxPoolFusion, ReturnNode<MaxPoolM>);

// Avg pooling Definition
AvgPoolM::AvgPoolM(const PoolingConfig &cfg) : PoolingM(cfg) {
  auto param = reinterpret_cast<PoolingParameter *>(OpParam());
  param->pool_mode_ = PoolMode_AvgPool;
  set_primitive(schema::PrimitiveType_AvgPoolFusion);
  set_name(UniqueName("AvgPool"));
}

int AvgPoolM::UnPopulate(const std::unique_ptr<schema::CNodeT> &cnode) {
  return PoolingM::UnPopulate<schema::AvgPoolFusionT>(cnode);
}

std::vector<EXPR *> AvgPoolM::Grad(EXPR *yt) {
  auto in = yt;
  if (yt->format() != NHWC && yt->dims().size() == C4NUM) {
    in = TransposeM::TransposeCHW2HWC(yt);
    in->node()->set_name(kGradName + "/" + name() + "/" + in->node()->name());
    PushOp(in->node());
  }
  auto pool_grad = new (std::nothrow) AvgPoolGradM(this);
  PushOp(pool_grad);
  return (*pool_grad)({input(0), output(0), in});
}

static ImportReg avgPoolReg(schema::PrimitiveType_AvgPoolFusion, ReturnNode<AvgPoolM>);

// Max Pool Grad Definition
MaxPoolGradM::MaxPoolGradM(MaxPoolM *node) {
  Node();
  CloneOpParam<PoolingParameter>(node->OpParam());
  set_primitive(schema::PrimitiveType_MaxPoolGrad);
  set_name(kGradName + "/" + node->name() + "/MaxPoolGrad");
}

int MaxPoolGradM::UnPopulate(const std::unique_ptr<schema::CNodeT> &cnode) {
  return PoolingM::UnPopulateGrad<schema::MaxPoolGradT>(cnode);
}

// Avg Pool Grad Definition
AvgPoolGradM::AvgPoolGradM(AvgPoolM *node) {
  Node();
  CloneOpParam<PoolingParameter>(node->OpParam());
  set_primitive(schema::PrimitiveType_AvgPoolGrad);
  set_name(kGradName + "/" + node->name() + "/AvgPoolGrad");
}

int AvgPoolGradM::UnPopulate(const std::unique_ptr<schema::CNodeT> &cnode) {
  return PoolingM::UnPopulateGrad<schema::AvgPoolGradT>(cnode);
}

namespace NN {
Node *MaxPool2D(const PoolingConfig &cfg) {
  auto c = new (std::nothrow) MaxPoolM(cfg);
  if (c == nullptr) {
    MS_LOG(ERROR) << "Cannot allocate max pool object";
    return nullptr;
  }
  return c;
}

Node *AvgPool2D(const PoolingConfig &cfg) {
  auto c = new (std::nothrow) AvgPoolM(cfg);
  if (c == nullptr) {
    MS_LOG(ERROR) << "Cannot allocate average pool object";
    return nullptr;
  }
  return c;
}
}  // namespace NN
}  // namespace lite
}  // namespace mindspore
