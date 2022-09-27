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

#include <vector>
#include <cmath>
#include "src/litert/kernel/cpu/base/prior_box.h"
#include "schema/model_generated.h"
#include "src/litert/kernel_registry.h"
#include "include/errorcode.h"

using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_NULL_PTR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_PriorBox;

namespace mindspore::kernel {
namespace {
constexpr int kInputNum = 2;
constexpr int kOutputNum = 1;
}  // namespace
int PriorBoxCPUKernel::Prepare() {
  if (prior_box_param_ == nullptr) {
    MS_LOG(ERROR) << "PriorBoxParameter nullptr";
    return RET_NULL_PTR;
  }

  if (in_tensors_.size() != kInputNum) {
    MS_LOG(ERROR) << "Size of input tensors is wrong.";
    return RET_ERROR;
  }
  if (out_tensors_.size() != kOutputNum) {
    MS_LOG(ERROR) << "Size of output tensors is wrong.";
    return RET_ERROR;
  }

  if (!InferShapeDone()) {
    return RET_OK;
  }
  return ReSize();
}

int PriorBoxCPUKernel::ReSize() { return GeneratePriorBox(); }

int PriorBoxCPUKernel::InitOutput(const std::vector<float> &different_aspect_ratios) {
  for (int i = 0; i < fmap_h_; i++) {
    float cy = i + prior_box_param_->offset;
    for (int j = 0; j < fmap_w_; j++) {
      float cx = j + prior_box_param_->offset;
      for (auto k = 0; k < prior_box_param_->min_sizes_size; k++) {
        float min_size = prior_box_param_->min_sizes[k];
        output_.emplace_back((cx - min_size / step_w_ * 0.5f) / fmap_w_);
        output_.emplace_back((cy - min_size / step_h_ * 0.5f) / fmap_h_);
        output_.emplace_back((cx + min_size / step_w_ * 0.5f) / fmap_w_);
        output_.emplace_back((cy + min_size / step_h_ * 0.5f) / fmap_h_);

        if (prior_box_param_->max_sizes_size > 0) {
          float max_size = prior_box_param_->max_sizes[k];
          MS_CHECK_GT(min_size * max_size, 0, RET_ERROR);
          float prime = sqrt(min_size * max_size);
          output_.emplace_back((cx - prime / step_w_ * 0.5f) / fmap_w_);
          output_.emplace_back((cy - prime / step_h_ * 0.5f) / fmap_h_);
          output_.emplace_back((cx + prime / step_w_ * 0.5f) / fmap_w_);
          output_.emplace_back((cy + prime / step_h_ * 0.5f) / fmap_h_);
        }

        for (auto v : different_aspect_ratios) {
          if (abs(v - 1.0f) < 1e-6) {
            continue;
          }
          MS_CHECK_GT(v, 0, RET_ERROR);
          float as_square_root = sqrt(v);
          float box_w = min_size * as_square_root;
          float box_h = min_size / as_square_root;
          output_.emplace_back((cx - box_w / step_w_ * 0.5f) / fmap_w_);
          output_.emplace_back((cy - box_h / step_h_ * 0.5f) / fmap_h_);
          output_.emplace_back((cx + box_w / step_w_ * 0.5f) / fmap_w_);
          output_.emplace_back((cy + box_h / step_h_ * 0.5f) / fmap_h_);
        }
      }
    }
  }
  return RET_OK;
}

int PriorBoxCPUKernel::GeneratePriorBox() {
  MS_CHECK_TRUE_RET(in_tensors_.at(0) != nullptr, RET_NULL_PTR);
  MS_CHECK_TRUE_RET(in_tensors_.at(1) != nullptr, RET_NULL_PTR);

  fmap_w_ = in_tensors_.at(0)->Width();
  fmap_h_ = in_tensors_.at(0)->Height();

  const int image_w = prior_box_param_->image_size_w > 0 ? prior_box_param_->image_size_w : in_tensors_.at(1)->Width();
  const int image_h = prior_box_param_->image_size_h > 0 ? prior_box_param_->image_size_h : in_tensors_.at(1)->Height();

  MS_CHECK_TRUE_RET(fmap_w_ != 0, RET_ERROR);
  MS_CHECK_TRUE_RET(fmap_h_ != 0, RET_ERROR);
  step_w_ = prior_box_param_->step_w > 0.0f ? prior_box_param_->step_w : static_cast<float>(image_w) / fmap_w_;
  step_h_ = prior_box_param_->step_h > 0.0f ? prior_box_param_->step_h : static_cast<float>(image_h) / fmap_h_;

  std::vector<float> different_aspect_ratios{1.0f};
  auto aspect_ratios = prior_box_param_->aspect_ratios;
  for (auto i = 0; i < prior_box_param_->aspect_ratios_size; i++) {
    float ratio = aspect_ratios[i];
    bool exist = std::any_of(different_aspect_ratios.begin(), different_aspect_ratios.end(),
                             [&](float v) { return abs(ratio - v) < 1e-6; });
    if (!exist) {
      different_aspect_ratios.emplace_back(ratio);
      if (prior_box_param_->flip) {
        MS_CHECK_GT(fabs(ratio), 1e-5, RET_ERROR);
        different_aspect_ratios.emplace_back(1.0f / ratio);
      }
    }
  }

  if (InitOutput(different_aspect_ratios) != RET_OK) {
    MS_LOG(ERROR) << "Init output size error.";
    return RET_ERROR;
  }

  // do clip
  if (prior_box_param_->clip) {
    for (auto item : output_) {
      if (item > 1.0f) {
        item = 1.0f;
      }
      if (item < 0.0f) {
        item = 0.0f;
      }
    }
  }

  // variance
  for (auto i = 0; i < out_tensors_[0]->Height() / COMM_SHAPE_SIZE; i++) {
    for (auto j = 0; j < COMM_SHAPE_SIZE; j++) {
      output_.emplace_back(prior_box_param_->variances[j]);
    }
  }
  return RET_OK;
}

int PriorBoxCPUKernel::PriorBoxImpl(int task_id) {
  auto src = output_.data();
  CHECK_NULL_RETURN(src);
  auto output = out_tensors_.at(0);
  CHECK_NULL_RETURN(output);
  auto output_data = reinterpret_cast<float *>(output->data());
  CHECK_NULL_RETURN(output_data);
  auto ret = PriorBox(src, output_data, output_.size(), task_id, thread_count_);
  return ret;
}

int RunPriorBox(void *cdata, int task_id, float lhs_scale, float rhs_scale) {
  auto prior_box = reinterpret_cast<PriorBoxCPUKernel *>(cdata);
  auto error_code = prior_box->PriorBoxImpl(task_id);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "Resize Run error task_id[" << task_id << "] error_code[" << error_code << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int PriorBoxCPUKernel::Run() {
  int error_code = ParallelLaunch(this->ms_context_, RunPriorBox, this, thread_count_);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "PriorBox run error, error_code[" << error_code << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_PriorBox, LiteKernelCreator<PriorBoxCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeInt8, PrimitiveType_PriorBox, LiteKernelCreator<PriorBoxCPUKernel>)
}  // namespace mindspore::kernel
