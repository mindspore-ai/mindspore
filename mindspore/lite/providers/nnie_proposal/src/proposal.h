/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_LITE_PROVIDERS_NNIE_PROPOSAL_SRC_PROPOSAL_H_
#define MINDSPORE_LITE_PROVIDERS_NNIE_PROPOSAL_SRC_PROPOSAL_H_
#include <vector>
#include "include/api/types.h"

#define LOG_TAG1 "Proposal"
#define LOGE(format, ...)                                                                       \
  do {                                                                                          \
    if (1) {                                                                                    \
      fprintf(stderr, "\n[ERROR] " LOG_TAG1 " [" __FILE__ ":%d] %s] ", __LINE__, __FUNCTION__); \
      fprintf(stderr, format, ##__VA_ARGS__);                                                   \
    }                                                                                           \
  } while (0)

#define LOGW(format, ...)                                                                         \
  do {                                                                                            \
    if (1) {                                                                                      \
      fprintf(stderr, "\n[Warning] " LOG_TAG1 " [" __FILE__ ":%d] %s] ", __LINE__, __FUNCTION__); \
      fprintf(stderr, format, ##__VA_ARGS__);                                                     \
    }                                                                                             \
  } while (0)

namespace mindspore {
namespace proposal {
typedef struct {
  uint32_t stride_;
  void *data_;
  uint32_t num_;
  uint32_t width_;
  uint32_t height_;
  uint32_t chn_;
} RpnBoundingBox;

#define MAX(a, b) (((a) > (b)) ? (a) : (b))
#define MIN(a, b) (((a) < (b)) ? (a) : (b))
#define HALF_VAL 0.5f     // the half value
#define COORDI_NUM 4      // coordinate numbers
#define PROPOSAL_WIDTH 6  // the number of proposal values
#define QUANT_BASE 4096   // the base value
#define SCORE_NUM 2       // the num of RPN scores
#define NUM_SCALE_ANCHORS 9
#define NUM_NMS 6000
#define MIN_SIZE 16

typedef struct {
  uint32_t scales_[9];
  uint32_t ratios_[9];
  uint32_t inputs_height_[2];
  uint32_t inputs_width_[2];
  uint32_t inputs_channel_[2];
  uint32_t inputs_stride_;
  uint32_t num_ratio_anchors_;
  uint32_t num_scale_anchors_;
  uint32_t ori_image_height_;
  uint32_t ori_image_width_;
  uint32_t min_size_;
  uint32_t spatial_scale_;
  uint32_t nms_thresh_;
  uint32_t filter_thresh_;
  uint32_t max_roi_num_;
  uint32_t num_before_nms_;
  float *inputs_[2];
  void *rpn_tmp_buf_;
  RpnBoundingBox rpn_bounding_box_;
} ProposalParam;

typedef struct {
  int32_t min_;
  int32_t max_;
} Stack;

int32_t ProposalInit(ProposalParam *param, uint32_t max_roi_num, uint32_t ori_image_height, uint32_t ori_image_width);
int32_t ProposalRun(ProposalParam *param);
void ProposalDeInit(ProposalParam *param);
}  // namespace proposal
}  // namespace mindspore
#endif  // MINDSPORE_LITE_PROVIDERS_NNIE_PROPOSAL_SRC_PROPOSAL_H_
