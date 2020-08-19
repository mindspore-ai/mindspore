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

#include <memory>
#include <vector>
#include "mindspore/lite/tools/converter/parser/caffe/caffe_proposal_parser.h"

namespace mindspore {
namespace lite {
STATUS CaffeProposalParser::Parse(const caffe::LayerParameter &proto,
                                  const caffe::LayerParameter &weight,
                                  schema::CNodeT *op,
                                  std::vector<schema::TensorT *> *weightVec) {
  std::unique_ptr<schema::ProposalT> attr(new schema::ProposalT());
  const caffe::ProposalParameter proposal_param = proto.proposal_param();

  if (proposal_param.has_feat_stride()) {
    attr->feat_stride = proposal_param.feat_stride();
  }
  if (proposal_param.has_base_size()) {
    attr->base_size = proposal_param.base_size();
  }
  if (proposal_param.has_min_size()) {
    attr->min_size = proposal_param.min_size();
  }
  if (proposal_param.has_pre_nms_topn()) {
    attr->pre_nms_topn = proposal_param.pre_nms_topn();
  }
  if (proposal_param.has_post_nms_topn()) {
    attr->post_nms_topn = proposal_param.post_nms_topn();
  }
  if (proposal_param.has_nms_thresh()) {
    attr->nms_thresh = proposal_param.nms_thresh();
  }
  const int num_ratio = proposal_param.ratio_size();
  attr->ratio.resize(num_ratio);
  for (int i = 0; i < num_ratio; ++i) {
      attr->ratio[i] = proposal_param.ratio(i);
  }
  const int num_scale = proposal_param.scale_size();
  attr->scale.resize(num_scale);
  for (int i = 0; i < num_scale; ++i) {
      attr->scale[i] = proposal_param.scale(i);
  }

  op->primitive = std::make_unique<schema::PrimitiveT>();
  op->primitive->value.value = attr.release();
  op->primitive->value.type = schema::PrimitiveType_Tile;
  return RET_OK;
}

CaffeNodeRegistrar g_caffeProposalParser("Proposal", new CaffeProposalParser());
}  // namespace lite
}  // namespace mindspore
