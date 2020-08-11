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
#include "src/runtime/kernel/arm/base/prior_box.h"
#include "schema/model_generated.h"
#include "src/kernel_factory.h"
#include "include/errorcode.h"
#include "include/context.h"
#include "src/runtime/runtime_api.h"

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
int PriorBoxCPUKernel::Init() {
  if (prior_box_param_ == nullptr) {
    MS_LOG(ERROR) << "PriorBoxParameter nullptr";
    return RET_NULL_PTR;
  }

  if (context_->infer_shape_interrupt_ && !context_->running_) {
    set_need_reinit();
    return RET_OK;
  }
  MS_ASSERT(in_tensors_.size() == kInputNum);
  MS_ASSERT(out_tensors_.size() == kOutputNum);

  auto ret = GeneratePriorBox();

  return ret;
}

int PriorBoxCPUKernel::GeneratePriorBox() {
  const int fmap_w = in_tensors_[0]->Width();
  const int fmap_h = in_tensors_[0]->Height();

  const int image_w = prior_box_param_->image_size_w > 0 ? prior_box_param_->image_size_w : in_tensors_[1]->Width();
  const int image_h = prior_box_param_->image_size_h > 0 ? prior_box_param_->image_size_h : in_tensors_[1]->Height();

  const float step_w =
    prior_box_param_->step_w > 0.0f ? prior_box_param_->step_w : static_cast<float>(image_w) / fmap_w;
  const float step_h =
    prior_box_param_->step_h > 0.0f ? prior_box_param_->step_h : static_cast<float>(image_h) / fmap_h;

  std::vector<float> different_aspect_ratios{1.0f};
  auto aspect_ratios = prior_box_param_->aspect_ratios;
  MS_ASSERT(aspect_ratios != nullptr);
  for (auto i = 0; i < prior_box_param_->aspect_ratios_size; i++) {
    float ratio = aspect_ratios[i];
    bool exist = std::any_of(different_aspect_ratios.begin(), different_aspect_ratios.end(),
                             [&](float v) { return abs(ratio - v) < 1e-6; });
    if (!exist) {
      different_aspect_ratios.emplace_back(ratio);
      if (prior_box_param_->flip) {
        different_aspect_ratios.emplace_back(1.0f / ratio);
      }
    }
  }

  for (int i = 0; i < fmap_h; i++) {
    float cy = i + prior_box_param_->offset;
    for (int j = 0; j < fmap_w; j++) {
      float cx = j + prior_box_param_->offset;
      for (auto k = 0; k < prior_box_param_->min_sizes_size; k++) {
        float min_size = prior_box_param_->min_sizes[k];
        output_.emplace_back((cx - min_size / step_w * 0.5f) / fmap_w);
        output_.emplace_back((cy - min_size / step_h * 0.5f) / fmap_h);
        output_.emplace_back((cx + min_size / step_w * 0.5f) / fmap_w);
        output_.emplace_back((cy + min_size / step_h * 0.5f) / fmap_h);

        if (prior_box_param_->max_sizes_size > 0) {
          float max_size = prior_box_param_->max_sizes[k];
          float prime = sqrt(min_size * max_size);
          output_.emplace_back((cx - prime / step_w * 0.5f) / fmap_w);
          output_.emplace_back((cy - prime / step_h * 0.5f) / fmap_h);
          output_.emplace_back((cx + prime / step_w * 0.5f) / fmap_w);
          output_.emplace_back((cy + prime / step_h * 0.5f) / fmap_h);
        }

        for (auto v : different_aspect_ratios) {
          if (abs(v - 1.0f) < 1e-6) {
            continue;
          }
          float as_square_root = sqrt(v);
          float box_w = min_size * as_square_root;
          float box_h = min_size / as_square_root;
          output_.emplace_back((cx - box_w / step_w * 0.5f) / fmap_w);
          output_.emplace_back((cy - box_h / step_h * 0.5f) / fmap_h);
          output_.emplace_back((cx + box_w / step_w * 0.5f) / fmap_w);
          output_.emplace_back((cy + box_h / step_h * 0.5f) / fmap_h);
        }
      }
    }
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
  for (auto i = 0; i < out_tensors_[0]->Height() / PRIOR_BOX_VAR_NUM; i++) {
    for (auto j = 0; j < PRIOR_BOX_VAR_NUM; j++) {
      output_.emplace_back(prior_box_param_->variances[j]);
    }
  }
  return RET_OK;
}

int PriorBoxCPUKernel::PriorBoxImpl(int task_id) {
  auto src = output_.data();
  auto output = out_tensors_.at(0);
  if (output == nullptr) {
    return RET_NULL_PTR;
  }
  auto ret = PriorBox(src, reinterpret_cast<float *>(output->Data()), output_.size(), task_id, thread_count_);
  return ret;
}

int RunPriorBox(int task_id, LiteParallelGroupEnv *penv, void *cdata) {
  auto prior_box = reinterpret_cast<PriorBoxCPUKernel *>(cdata);

  auto error_code = prior_box->PriorBoxImpl(task_id);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "Resize Run error task_id[" << task_id << "] error_code[" << error_code << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int PriorBoxCPUKernel::Run() {
  int error_code = LiteBackendParallelLaunch(RunPriorBox, this, thread_count_);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "PriorBox run error, error_code[" << error_code << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

kernel::LiteKernel *CpuPriorBoxKernelCreator(const std::vector<lite::tensor::Tensor *> &inputs,
                                             const std::vector<lite::tensor::Tensor *> &outputs,
                                             OpParameter *opParameter, const Context *ctx,
                                             const kernel::KernelKey &desc, const lite::Primitive *primitive) {
  if (opParameter == nullptr) {
    MS_LOG(ERROR) << "Input opParameter is nullptr!";
    return nullptr;
  }
  if (desc.type != schema::PrimitiveType_PriorBox) {
    MS_LOG(ERROR) << "PriorBox invalid desc type " << desc.type;
    return nullptr;
  }
  auto *kernel = new (std::nothrow) PriorBoxCPUKernel(opParameter, inputs, outputs, ctx, primitive);
  if (kernel == nullptr) {
    MS_LOG(ERROR) << "new PriorBoxCPUKernel fail!";
    return nullptr;
  }
  auto ret = kernel->Init();
  if (ret != RET_OK) {
    delete kernel;
    MS_LOG(ERROR) << "Init kernel failed, name: " << opParameter->name_ << ", type: "
                  << schema::EnumNamePrimitiveType(static_cast<schema::PrimitiveType>(opParameter->type_));
    return nullptr;
  }
  return kernel;
}

REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_PriorBox, CpuPriorBoxKernelCreator)
REG_KERNEL(kCPU, kNumberTypeInt8, PrimitiveType_PriorBox, CpuPriorBoxKernelCreator)
}  // namespace mindspore::kernel
