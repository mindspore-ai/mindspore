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

#include "src/expression/ops/conv.h"
#include <memory>
#include "src/expression/ops/biasadd.h"
#include "src/expression/ops/depend.h"
#include "src/expression/ops/transpose.h"
#include "nnacl/conv_parameter.h"
#include "inner/model_generated.h"
#include "src/expression/import.h"
#include "src/expression/ops.h"
#include "src/litert/cxx_api/expression/node_impl.h"

namespace mindspore {
namespace lite {
ConvM::ConvM(const ConvConfig &cfg) : Node() {
  auto op_param = calloc(1, sizeof(ConvParameter));
  if (op_param == nullptr) {
    MS_LOG(ERROR) << " cannot allocate ConvParameter";
    return;
  }
  SetOpParam(op_param);
  ConvParameter *conv_param = reinterpret_cast<ConvParameter *>(OpParam());
  conv_param->input_channel_ = cfg.in_channel_;
  conv_param->output_channel_ = cfg.out_channel_;
  conv_param->kernel_h_ = cfg.kernel_size_[0];
  conv_param->kernel_w_ = cfg.kernel_size_[1];
  conv_param->stride_h_ = cfg.stride_[0];
  conv_param->stride_w_ = cfg.stride_[1];
  auto pad_mode = GetMode(cfg.pad_mode_);
  if (pad_mode == -1) {
    MS_LOG(ERROR) << "bad pad mode";
    return;
  }
  conv_param->pad_mode_ = static_cast<PadMode>(pad_mode);
  conv_param->pad_u_ = cfg.padding_[C0NUM];
  conv_param->pad_d_ = cfg.padding_[C1NUM];
  conv_param->pad_l_ = cfg.padding_[C2NUM];
  conv_param->pad_r_ = cfg.padding_[C3NUM];
  conv_param->dilation_h_ = cfg.dilation_[C0NUM];
  conv_param->dilation_w_ = cfg.dilation_[C1NUM];
  conv_param->group_ = cfg.group_;
  conv_param->out_format_ = NHWC;
  conv_param->act_type_ = ActType_No;
  expr()->SetSize(C2NUM);
  set_primitive(schema::PrimitiveType_Conv2DFusion);
  set_name(UniqueName("Conv"));
  Param::Mode mode = Param::String2Enum(cfg.weight_init_);
  std::vector<int> dims = {conv_param->output_channel_, conv_param->kernel_h_, conv_param->kernel_w_,
                           conv_param->input_channel_ / conv_param->group_};
  auto w = CreateWeights(dims, kNumberTypeFloat32, KHWC, mode, "weights");
  expr()->set_params(C1NUM, w);
  if (cfg.has_bias) {
    bias_ = new (std::nothrow) BiasAddM(KHWC);
    if (bias_ == nullptr) {
      MS_LOG(ERROR) << "Cannot allocate bias";
      return;
    }
    bias_->update_name(name());
    std::vector<int> dim_bias = {conv_param->output_channel_};
    wbias_ = CreateWeights(dim_bias, kNumberTypeFloat32, KHWC, Param::Mode::ZEROS, "weights");
    AddLearn(wbias_->node());
    PushOp(bias_);
  }
  SetLearn();
}

std::vector<EXPR *> ConvM::construct(const std::vector<EXPR *> &inputs) {
  auto in = inputs;
  auto x = in.front();
  if (x->format() != NHWC && x->dims().size() == C4NUM) {
    x = TransposeM::TransposeCHW2HWC(x);
    x->node()->set_name(name() + "/" + x->node()->name());
    PushOp(x->node());
    in.at(0) = x;
  }
  auto y = Node::construct(in);
  if (bias_ != nullptr) {
    y = (*bias_)({y.front(), wbias_});
  }
  return y;
}

void ConvM::SetLearn() { AddLearn(input(C1NUM)->node()); }

int ConvM::GetMode(std::string mode) {
  const std::vector<std::string> list = {"pad", "same", "valid"};
  auto itr = std::find(list.begin(), list.end(), mode);
  if (itr == list.end()) {
    MS_LOG(ERROR) << "illegal mode" << mode;
    return -1;
  }
  return std::distance(list.begin(), itr);
}

std::vector<EXPR *> ConvM::Grad(EXPR *yt) {
  // Generate Input Grad
  EXPR *in = yt;
  if (yt->format() != NHWC && yt->dims().size() == C4NUM) {
    in = TransposeM::TransposeCHW2HWC(yt);
    in->node()->set_name(kGradName + "/" + name() + "/" + in->node()->name());
    PushOp(in->node());
  }
  auto inGrad = new (std::nothrow) ConvInputGradM(this);
  if (inGrad == nullptr) {
    MS_LOG(ERROR) << "Cannot allocate convolution input grad";
    return {};
  }
  PushOp(inGrad);
  auto ig = (*inGrad)({in, input(1), inGrad->input(2)});
  // Execution Control Flow !
  auto depend = NN::Depend();
  if (depend == nullptr) {
    MS_LOG(ERROR) << "Cannot allocate depend";
    return {};
  }
  PushOp(depend);
  depend->update_name(name());
  auto de = (*depend)({inGrad->expr()});
  // Generate Filter Grad
  auto filterGrad = new (std::nothrow) ConvFilterGradM(this);
  if (filterGrad == nullptr) {
    MS_LOG(ERROR) << "Cannot allocate convolution filter grad";
    return {};
  }
  PushOp(filterGrad);
  filterGrad->update_name(name());
  auto fg = (*filterGrad)({in, input(0), filterGrad->input(2), de[0]});
  std::vector<EXPR *> res = {ig[0], fg[0]};
  return res;
}

int ConvM::UnPopulate(const std::unique_ptr<schema::CNodeT> &cnode) {
  auto conv_param = reinterpret_cast<const ConvParameter *>(OpParam());
  auto prim = new (std::nothrow) schema::Conv2DFusionT;
  if (prim == nullptr) {
    MS_LOG(ERROR) << "Cannot allocate primitive";
    return RET_ERROR;
  }
  prim->activation_type = static_cast<schema::ActivationType>(conv_param->act_type_);
  prim->format = static_cast<schema::Format>(conv_param->out_format_);
  prim->stride = {conv_param->stride_h_, conv_param->stride_w_};
  prim->kernel_size = {conv_param->kernel_h_, conv_param->kernel_w_};
  prim->dilation = {conv_param->dilation_h_, conv_param->dilation_w_};
  prim->out_channel = conv_param->output_channel_;
  prim->in_channel = conv_param->input_channel_;
  prim->group = conv_param->group_;
  prim->pad_mode = static_cast<schema::PadMode>(conv_param->pad_mode_);
  prim->pad_list = {conv_param->pad_u_, conv_param->pad_d_, conv_param->pad_l_, conv_param->pad_r_};
  prim->mode = 1;
  cnode->primitive->value.value = prim;
  return RET_OK;
}

ConvInputGradM::ConvInputGradM(ConvM *conv_node) : Node() {
  CloneOpParam<ConvParameter>(conv_node->OpParam());
  set_primitive(schema::PrimitiveType_Conv2DBackpropInputFusion);
  set_name(kGradName + "/conv2DBackpropInput");
  expr()->SetSize(C3NUM);
  auto const x = conv_node->input(0);
  CreateConstTensor(C2NUM, {static_cast<int32_t>(x->dims().size())}, kNumberTypeInt32, KHWC, "shape", x->dims().data());
}

int ConvInputGradM::UnPopulate(const std::unique_ptr<schema::CNodeT> &cnode) {
  auto conv_param = reinterpret_cast<const ConvParameter *>(OpParam());
  auto prim = new (std::nothrow) schema::Conv2DBackpropInputFusionT;
  if (prim == nullptr) {
    MS_LOG(ERROR) << "Cannot allocate primitive";
    return RET_ERROR;
  }
  prim->activation_type = static_cast<schema::ActivationType>(conv_param->act_type_);
  prim->format = static_cast<schema::Format>(conv_param->out_format_);
  prim->stride = {conv_param->stride_h_, conv_param->stride_w_};
  prim->kernel_size = {conv_param->kernel_h_, conv_param->kernel_w_};
  prim->dilation = {conv_param->dilation_h_, conv_param->dilation_w_};
  prim->out_channel = conv_param->output_channel_;
  prim->in_channel = conv_param->input_channel_;
  prim->group = conv_param->group_;
  prim->pad_mode = static_cast<schema::PadMode>(conv_param->pad_mode_);
  prim->pad_list = {conv_param->pad_u_, conv_param->pad_d_, conv_param->pad_l_, conv_param->pad_r_};
  cnode->primitive->value.value = prim;
  return RET_OK;
}

ConvFilterGradM::ConvFilterGradM(ConvM *conv_node) : Node() {
  CloneOpParam<ConvParameter>(conv_node->OpParam());
  set_primitive(schema::PrimitiveType_Conv2DBackpropFilterFusion);
  set_name(kGradName + "/conv2DBackpropFilter");
  expr()->SetSize(C4NUM);
  auto w = conv_node->input(1);
  CreateConstTensor(C2NUM, {static_cast<int32_t>(w->dims().size())}, kNumberTypeInt32, KHWC, "shape", w->dims().data());
}
int ConvFilterGradM::UnPopulate(const std::unique_ptr<schema::CNodeT> &cnode) {
  auto conv_param = reinterpret_cast<const ConvParameter *>(OpParam());
  auto prim = new (std::nothrow) schema::Conv2DBackpropFilterFusionT;
  if (prim == nullptr) {
    MS_LOG(ERROR) << "Cannot allocate primitive";
    return RET_ERROR;
  }
  prim->activation_type = static_cast<schema::ActivationType>(conv_param->act_type_);
  prim->format = static_cast<schema::Format>(conv_param->out_format_);
  prim->stride = {conv_param->stride_h_, conv_param->stride_w_};
  prim->kernel_size = {conv_param->kernel_h_, conv_param->kernel_w_};
  prim->dilation = {conv_param->dilation_h_, conv_param->dilation_w_};
  prim->out_channel = conv_param->output_channel_;
  prim->in_channel = conv_param->input_channel_;
  prim->group = conv_param->group_;
  prim->pad_mode = static_cast<schema::PadMode>(conv_param->pad_mode_);
  prim->pad_list = {conv_param->pad_u_, conv_param->pad_d_, conv_param->pad_l_, conv_param->pad_r_};
  cnode->primitive->value.value = prim;
  return RET_OK;
}

static ImportReg reg(schema::PrimitiveType_Conv2DFusion, ReturnNode<ConvM>);

namespace NN {
Node *Conv2D(const ConvConfig &cfg) {
  auto c = new (std::nothrow) ConvM(cfg);
  if (c == nullptr) {
    MS_LOG(ERROR) << "Cannot allocate Convolution object";
    return nullptr;
  }
  return c;
}
}  // namespace NN
}  // namespace lite
}  // namespace mindspore
