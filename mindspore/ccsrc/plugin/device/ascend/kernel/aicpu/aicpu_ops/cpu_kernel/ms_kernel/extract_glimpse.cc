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
#include "extract_glimpse.h"
#include <iostream>
#include <random>
#include "context/inc/cpu_kernel_utils.h"
#include "utils/eigen_tensor.h"
#include "utils/kernel_util.h"
using namespace std;
random_device rd;
mt19937 gen(rd());
uniform_real_distribution<float> dis_uniform(0.0f, 255.0f);
normal_distribution<float> dis_normal(10, 0.5);
#define SHED 2048
namespace {
const uint32_t kOutputNum = 1;
const uint32_t kInputNum = 3;
const char *kExtractGlimpse = "ExtractGlimpse";
}  // namespace
namespace aicpu {
uint32_t ExtractGlimpseCpuKernel::Compute(CpuKernelContext &ctx) {
  CUST_KERNEL_HANDLE_ERROR(ctx, NormalCheck(ctx, kInputNum, kOutputNum),
                           "ExtractGlimpse check input and output number failed.");
  CUST_KERNEL_HANDLE_ERROR(ctx, ExtractGlimpseCheck(ctx), "ExtractGlimpse check params failed.");
  Tensor *x = ctx.Input(0);
  Tensor *ss = ctx.Input(1);
  Tensor *offsets = ctx.Input(2);
  Tensor *y = ctx.Output(0);
  AttrValue *centered = ctx.GetAttr("centered");
  AttrValue *normalized = ctx.GetAttr("normalized");
  AttrValue *uniform_noise = ctx.GetAttr("uniform_noise");
  AttrValue *noise = ctx.GetAttr("noise");
  float *x_data = (float *)x->GetData();
  int32_t *ss_data = (int32_t *)ss->GetData();
  float *offsets_data = (float *)offsets->GetData();
  float *y_data = (float *)y->GetData();
  uint64_t offsets_cnt = offsets->GetTensorShape()->GetDimSize(0);
  uint64_t batch_cnt = x->GetTensorShape()->GetDimSize(0);
  CUST_KERNEL_CHECK_FALSE(ctx, offsets_cnt == batch_cnt, KERNEL_STATUS_PARAM_INVALID, "offsets should equal to batches")
  int64_t image_height = x->GetTensorShape()->GetDimSize(1);
  int64_t image_width = x->GetTensorShape()->GetDimSize(2);
  int64_t channels = x->GetTensorShape()->GetDimSize(3);
  uint64_t g_height = ss_data[0];
  uint64_t g_width = ss_data[1];
  uint64_t size1 = image_width * image_height * channels;
  uint64_t size2 = image_width * channels;
  uint64_t size3 = g_height * g_width * channels;
  uint64_t size4 = size3 / g_height;
  int64_t g_size = g_width * g_height;
  if (batch_cnt > SHED) {
    uint32_t min_core = 1;
    uint64_t max_core = std::max(min_core, aicpu::CpuKernelUtils::GetCPUNum(ctx) - 2);
    max_core = min(max_core, (uint64_t)batch_cnt);
    auto fun = [&](size_t st, size_t ed) {
      for (auto i = st; i < ed; i++) {
        float x = offsets_data[i << 1];
        float y = offsets_data[1 + (i << 1)];
        if (normalized->GetBool()) {
          x *= image_height;
          y *= image_width;
        }
        if (centered->GetBool()) {
          x /= 2.0f;
          y /= 2.0f;
          x += image_height / 2.0f;
          y += image_width / 2.0f;
        }
        x -= g_height / 2.0f;
        y -= g_width / 2.0f;
        for (int64_t v = 0; v < g_size; v++) {
          int64_t j = v / g_width;
          int64_t k = v % g_width;
          int64_t a = (int64_t)x + j;
          int64_t b = (int64_t)y + k;
          uint64_t pos_y = i * size3 + j * size4 + k * channels;
          if (a < 0 || a >= image_height || b < 0 || b >= image_width) {
            for (int u = 0; u < channels; u++) {
              if (uniform_noise->GetBool())
                y_data[pos_y + u] = dis_uniform(gen);
              else if (noise->GetString() == "zero")
                y_data[pos_y + u] = 0.0f;
              else if (noise->GetString() == "gaussian")
                y_data[pos_y + u] = max(0.0f, dis_normal(gen));
              else {
                CUST_KERNEL_LOG_ERROR(ctx, "noise type [%s] unsupported.", noise->GetString().c_str());
                return KERNEL_STATUS_PARAM_INVALID;
              }
            }
            continue;
          }
          uint64_t pos_x = i * size1 + a * size2 + b * channels;
          for (int u = 0; u < channels; u++) {
            y_data[pos_y + u] = x_data[pos_x + u];
          }
        }
      }
      return KERNEL_STATUS_OK;
    };
    CUST_KERNEL_HANDLE_ERROR(ctx, CpuKernelUtils::ParallelFor(ctx, batch_cnt, batch_cnt / max_core, fun),
                             "ExtractGlimpse Compute failed.");
  } else {
    for (uint64_t i = 0; i < batch_cnt; i++) {
      float x = offsets_data[i << 1], y = offsets_data[1 + (i << 1)];
      if (normalized->GetBool()) {
        x *= image_height;
        y *= image_width;
      }
      if (centered->GetBool()) {
        x /= 2.0f;
        y /= 2.0f;
        x += image_height / 2.0f;
        y += image_width / 2.0f;
      }
      x -= g_height / 2.0f;
      y -= g_width / 2.0f;
      if (g_size < SHED) {
        for (int64_t v = 0; v < g_size; v++) {
          int64_t j = v / g_width;
          int64_t k = v % g_width;
          int64_t a = (int64_t)x + j;
          int64_t b = (int64_t)y + k;
          uint64_t pos_y = i * size3 + j * size4 + k * channels;
          if (a < 0 || a >= image_height || b < 0 || b >= image_width) {
            for (int u = 0; u < channels; u++) {
              if (uniform_noise->GetBool())
                y_data[pos_y + u] = dis_uniform(gen);
              else if (noise->GetString() == "zero")
                y_data[pos_y + u] = 0.0f;
              else if (noise->GetString() == "gaussian")
                y_data[pos_y + u] = max(0.0f, dis_normal(gen));
              else {
                CUST_KERNEL_LOG_ERROR(ctx, "noise type [%s] unsupported.", noise->GetString().c_str());
                return KERNEL_STATUS_PARAM_INVALID;
              }
            }
            continue;
          }
          uint64_t pos_x = i * size1 + a * size2 + b * channels;
          for (int u = 0; u < channels; u++) {
            y_data[pos_y + u] = x_data[pos_x + u];
          }
        }
      } else {
        uint32_t min_core = 1;
        uint64_t max_core = std::max(min_core, aicpu::CpuKernelUtils::GetCPUNum(ctx) - 2);
        max_core = min(max_core, (uint64_t)g_size);
        auto fun = [&](size_t st, size_t ed) {
          for (auto v = st; v < ed; v++) {
            int64_t j = v / g_width;
            int64_t k = v % g_width;
            int64_t a = (int64_t)x + j;
            int64_t b = (int64_t)y + k;
            uint64_t pos_y = i * size3 + j * size4 + k * channels;
            if (a < 0 || a >= image_height || b < 0 || b >= image_width) {
              for (int u = 0; u < channels; u++)
                if (uniform_noise->GetBool())
                  y_data[pos_y + u] = dis_uniform(gen);
                else if (noise->GetString() == "zero")
                  y_data[pos_y + u] = 0.0f;
                else if (noise->GetString() == "gaussian")
                  y_data[pos_y + u] = max(0.0f, dis_normal(gen));
                else {
                  CUST_KERNEL_LOG_ERROR(ctx, "noise type [%s] unsupported.", noise->GetString().c_str());
                  return KERNEL_STATUS_PARAM_INVALID;
                }
              continue;
            }
            uint64_t pos_x = i * size1 + a * size2 + b * channels;
            for (int u = 0; u < channels; u++) {
              y_data[pos_y + u] = x_data[pos_x + u];
            }
          }
          return KERNEL_STATUS_OK;
        };
        CUST_KERNEL_HANDLE_ERROR(ctx, CpuKernelUtils::ParallelFor(ctx, g_size, g_size / max_core, fun),
                                 "ExtractGlimpse Compute failed.");
      }
    }
  }
  return KERNEL_STATUS_OK;
}
uint32_t ExtractGlimpseCpuKernel::ExtractGlimpseCheck(CpuKernelContext &ctx) {
  Tensor *x = ctx.Input(0);
  Tensor *ss = ctx.Input(1);
  Tensor *offsets = ctx.Input(2);
  Tensor *y = ctx.Output(0);
  AttrValue *centered = ctx.GetAttr("centered");
  AttrValue *normalized = ctx.GetAttr("normalized");
  AttrValue *uniform_noise = ctx.GetAttr("uniform_noise");
  AttrValue *noise = ctx.GetAttr("noise");
  CUST_KERNEL_CHECK_NULLPTR(ctx, x, KERNEL_STATUS_PARAM_INVALID, "Get input 0 failed.")
  CUST_KERNEL_CHECK_NULLPTR(ctx, ss, KERNEL_STATUS_PARAM_INVALID, "Get input 1 failed.")
  CUST_KERNEL_CHECK_NULLPTR(ctx, offsets, KERNEL_STATUS_PARAM_INVALID, "Get input 2 failed.")
  CUST_KERNEL_CHECK_NULLPTR(ctx, y, KERNEL_STATUS_PARAM_INVALID, "Get output 0 failed.")
  CUST_KERNEL_CHECK_NULLPTR(ctx, centered, KERNEL_STATUS_PARAM_INVALID, "Get attribute centered failed.")
  CUST_KERNEL_CHECK_NULLPTR(ctx, normalized, KERNEL_STATUS_PARAM_INVALID, "Get attribute normalized failed.")
  CUST_KERNEL_CHECK_NULLPTR(ctx, uniform_noise, KERNEL_STATUS_PARAM_INVALID, "Get attribute uniform_noise failed.")
  CUST_KERNEL_CHECK_NULLPTR(ctx, noise, KERNEL_STATUS_PARAM_INVALID, "Get attribute noise failed.")
  CUST_KERNEL_CHECK_NULLPTR(ctx, x->GetData(), KERNEL_STATUS_PARAM_INVALID, "Get input 0 data failed.")
  CUST_KERNEL_CHECK_NULLPTR(ctx, ss->GetData(), KERNEL_STATUS_PARAM_INVALID, "Get input 1 data failed.")
  CUST_KERNEL_CHECK_NULLPTR(ctx, offsets->GetData(), KERNEL_STATUS_PARAM_INVALID, "Get input 2 data failed.")
  CUST_KERNEL_CHECK_NULLPTR(ctx, y->GetData(), KERNEL_STATUS_PARAM_INVALID, "Get output 0 data failed.")
  CUST_KERNEL_CHECK_FALSE(ctx,
                          x->GetDataType() == DT_FLOAT && ss->GetDataType() == DT_INT32 &&
                            offsets->GetDataType() == DT_FLOAT && y->GetDataType() == DT_FLOAT,
                          KERNEL_STATUS_PARAM_INVALID, "data type error.")
  return KERNEL_STATUS_OK;
}
REGISTER_MS_CPU_KERNEL(kExtractGlimpse, ExtractGlimpseCpuKernel);
}  // namespace aicpu
