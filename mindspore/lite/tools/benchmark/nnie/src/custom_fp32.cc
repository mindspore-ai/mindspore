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

#include "src/custom_fp32.h"
#include <map>
#include <memory>
#include "schema/model_generated.h"
#include "include/registry/register_kernel.h"
#include "include/errorcode.h"
#include "src/nnie_manager.h"
#include "src/nnie_print.h"
#include "src/nnie_cfg_parser.h"

using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_Custom;

namespace mindspore {
namespace nnie {
bool CustomCPUKernel::load_model_ = false;

int CustomCPUKernel::run_seg_ = 0;
bool CustomCPUKernel::roi_used_ = false;
int CustomCPUKernel::Prepare() {
  if (!load_model_) {
    Flags flags;
    flags.Init();
    if (nnie::NNIEManager::GetInstance()->CfgInit(flags.max_roi_num_, flags.time_step_, flags.core_ids_) != RET_OK) {
      LOGE("Nnie init cfg fail");
      return RET_ERROR;
    }

    if (nnie::NNIEManager::GetInstance()->Init(reinterpret_cast<char *>(inputs_[inputs_.size() - 1].MutableData()),
                                               static_cast<int>(inputs_[inputs_.size() - 1].ElementNum()), inputs_)) {
      // LOGW("Load WK Model Fail");
      return RET_OK;
    }
    load_model_ = true;
  }
  outputs_shapes_.resize(outputs_.size());
  for (size_t i = 0; i < outputs_.size(); i++) {
    outputs_shapes_[i] = outputs_[i].Shape();
  }
  return RET_OK;
}

int CustomCPUKernel::ReSize() {
  if (load_model_) {
    nnie::NNIEManager::GetInstance()->Release();
    load_model_ = false;
  }

  return Prepare();
}

int CustomCPUKernel::Execute() {
  if (!load_model_) {
    LOGE("WK Model is not load.");
    return RET_ERROR;
  }
  run_seg_ = seg_id_;

  if (nnie::NNIEManager::GetInstance()->FillData(&inputs_, run_seg_)) {
    LOGE("Fail Fill Data");
    return RET_ERROR;
  }

  if (nnie::NNIEManager::GetInstance()->Run(&outputs_, run_seg_, outputs_shapes_)) {
    LOGE("Fail WK Run");
    return RET_ERROR;
  }
  run_seg_++;
  return RET_OK;
}

CustomCPUKernel::~CustomCPUKernel() {
  if (load_model_) {
    nnie::NNIEManager::GetInstance()->Release();
    load_model_ = false;
  }
}

bool GetCustomAttr(char *buf, int buf_size, const mindspore::schema::Custom *op, const std::string &attr) {
  int attr_size;
  for (size_t i = 0; i < op->attr()->size(); i++) {
    if (op->attr()->Get(i)->name()->str() == attr) {
      auto output_info = op->attr()->Get(i)->data();
      attr_size = static_cast<int>(output_info->size());
      if (attr_size >= buf_size) {
        LOGE("attr size too big");
        return false;
      }
      for (int j = 0; j < attr_size; j++) {
        buf[j] = static_cast<char>(output_info->Get(j));
      }
      buf[attr_size] = 0;
      return true;
    }
  }
  return false;
}

std::shared_ptr<mindspore::kernel::Kernel> CustomCreateKernel(const std::vector<MSTensor> &inputs,
                                                              const std::vector<MSTensor> &outputs,
                                                              const mindspore::schema::Primitive *primitive,
                                                              const mindspore::Context *ctx) {
  if (primitive->value_type() != mindspore::schema::PrimitiveType_Custom) {
    LOGE("Primitive type is not PrimitiveType_Custom");
    return nullptr;
  }

  auto op = primitive->value_as_Custom();
  if (op->attr()->size() < 1) {
    LOGE("There are at least 1 attribute of Custom");
    return nullptr;
  }

  int64_t ndims;
  bool forward_bbox = false;
  char *res = nullptr;
  char buf[kMaxSize];
  if (GetCustomAttr(buf, kMaxSize, op, "id")) {
    res = nullptr;
    ndims = strtol(buf, &res, kDecimal);
    if ((*res) != 0) {
      LOGE("Get attr id data fail");
      return nullptr;
    }
  } else {
    LOGE("Custom op should have id");
    return nullptr;
  }

  if (GetCustomAttr(buf, kMaxSize, op, "ForwardWithBbox")) {
    res = nullptr;
    int64_t temp_val = strtol(buf, &res, kDecimal);
    if ((*res) != 0) {
      LOGE("Get attr ForwardWithBbox data fail");
      return nullptr;
    }
    if (temp_val > 0) {
      forward_bbox = true;
    }
  }
  auto kernel = std::make_shared<CustomCPUKernel>(ndims, forward_bbox, inputs, outputs, primitive, ctx);
  if (kernel == nullptr) {
    LOGE("new custom kernel is nullptr");
    return nullptr;
  }
  return kernel;
}
}  // namespace nnie
}  // namespace mindspore
namespace mindspore {
namespace registry {
namespace {
const auto kFloat32 = DataType::kNumberTypeFloat32;
const auto kInt8 = DataType::kNumberTypeInt8;
const auto kUint8 = DataType::kNumberTypeUInt8;
}  // namespace
REGISTER_CUSTOM_KERNEL(CPU, NNIE, kFloat32, NNIE, nnie::CustomCreateKernel)
REGISTER_CUSTOM_KERNEL(CPU, NNIE, kInt8, NNIE, nnie::CustomCreateKernel)
REGISTER_CUSTOM_KERNEL(CPU, NNIE, kUint8, NNIE, nnie::CustomCreateKernel)
}  // namespace registry
}  // namespace mindspore
