/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
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
#include "cpu_kernel/common/cpu_kernel_cache.h"

#include <algorithm>
#include <climits>
#include <string>
#include <vector>

#include "cce/aicpu_engine_struct.h"
#include "cpu_kernel/inc/cpu_ops_kernel.h"
#include "cpu_kernel/common/cpu_kernel_register.h"
#include "cpu_kernel/common/cpu_kernel_utils.h"
#include "mindspore/ccsrc/plugin/device/ascend/kernel/aicpu/aicpu_ops/common/kernel_log.h"
#include "cpu_kernel/common/runtime_tensor_desc.h"
#include "cpu_kernel/common/status.h"

namespace {
// max io address number limit is 1024
constexpr uint32_t kMaxIoAddrNumParamLen = 1024;
// max LRU cache number is 256
constexpr uint32_t kMaxLRUCacheNum = 256;
}  // namespace

namespace aicpu {
/*
 * Init kernel cache.
 */
int32_t CpuKernelCache::InitParameter() {
  if (!GetSessionFlag()) {
    SetCapacity(kMaxLRUCacheNum);
  }
  return 0;
}

/*
 * update framework output tensor shape.
 */
uint32_t CpuKernelCache::UpdateFWKOutputShape(ExtInfoMsg &ext_info_msg, const CpuKernelContext &ctx) const {
  if (ext_info_msg.unknown_shape) {
    for (size_t i = 0; i < ctx.GetOutputsSize(); ++i) {
      Tensor *output = ctx.Output(i);
      KERNEL_CHECK_NULLPTR(output, KERNEL_STATUS_PARAM_INVALID, "Get output[%zu] failed.", i)
      auto shape = output->GetTensorShape();
      KERNEL_CHECK_NULLPTR(shape, KERNEL_STATUS_PARAM_INVALID, "Get output[%zu] shape failed.", i)

      for (int32_t index = 0; index < shape->GetDims(); ++index) {
        ext_info_msg.output_shape_and_type[i]->dims[index] = shape->GetDimSize(index);
      }
    }
  }
  for (auto it = ext_info_msg.unknown_shape_output_index_addr.begin();
       it != ext_info_msg.unknown_shape_output_index_addr.end(); ++it) {
    Tensor *output = ctx.Output(it->first);
    KERNEL_CHECK_NULLPTR(output, KERNEL_STATUS_PARAM_INVALID, "Get output[%u] failed.", it->first)
    auto shape = output->GetTensorShape();
    KERNEL_CHECK_NULLPTR(shape, KERNEL_STATUS_PARAM_INVALID, "Get output[%u] shape failed.", it->first)
    ge::RuntimeTensorDesc *tensor_desc = reinterpret_cast<ge::RuntimeTensorDesc *>(static_cast<uintptr_t>(it->second));
    KERNEL_CHECK_FALSE((shape->GetDims() <= ge::kMaxDimSize), KERNEL_STATUS_PARAM_INVALID,
                       "Max shape size[32], but got output[%u] shape size[%d]", it->first, shape->GetDims())
    tensor_desc->shape[0] = shape->GetDims();
    for (int32_t index = 0; index < shape->GetDims(); ++index) {
      tensor_desc->shape[index + 1] = shape->GetDimSize(index);
    }
  }
  return KERNEL_STATUS_OK;
}

/*
 * get shape information from framework.
 */
void CpuKernelCache::GetDimsFromShapeAndType(const FWKAdapter::ShapeAndType *shape_and_type,
                                             std::vector<int64_t> &dims) const {
  for (uint32_t index = 0; index < FWKAdapter::kMaxShapeDims; ++index) {
    // LLONG_MIN for dim end flag
    if (shape_and_type->dims[index] == LLONG_MIN) {
      break;
    }
    int64_t dim_value = shape_and_type->dims[index];
    KERNEL_LOG_INFO("Get extend shape[%u] is [%ld]", index, dim_value);
    dims.emplace_back(dim_value);
  }
}

void CpuKernelCache::GetDimsFromArrays(const int64_t *shape, size_t len, std::vector<int64_t> &dims) const {
  for (size_t index = 0; index < len; ++index) {
    KERNEL_LOG_INFO("Get arrays shape[%zu] is [%ld]", index, shape[index]);
    dims.emplace_back(shape[index]);
  }
}

/*
 * update tensor information.
 */
uint32_t CpuKernelCache::UpdateTensor(const std::vector<uint64_t> &io_addrs, ExtInfoMsg &ext_info_msg,
                                      CpuKernelContext &ctx) const {
  KERNEL_LOG_INFO("Update tensor info begin.");
  if (io_addrs.size() != ctx.GetInputsSize() + ctx.GetOutputsSize()) {
    KERNEL_LOG_ERROR(
      "Addr number[%zu] is not equal to the sum of inputs[%zu] and "
      "output[%zu].",
      io_addrs.size(), ctx.GetInputsSize(), ctx.GetOutputsSize());
    return KERNEL_STATUS_PARAM_INVALID;
  }

  if ((ext_info_msg.unknown_shape) && ((ext_info_msg.input_shape_and_type.size() != ctx.GetInputsSize()) ||
                                       (ext_info_msg.output_shape_and_type.size() != ctx.GetOutputsSize()))) {
    KERNEL_LOG_ERROR(
      "Input shape_and_type size error, input size[%zu], input "
      "shape_and_type "
      "size[%zu], output size[%zu], output shape_and_type size[%zu].",
      ctx.GetInputsSize(), ext_info_msg.input_shape_and_type.size(), ctx.GetOutputsSize(),
      ext_info_msg.output_shape_and_type.size());
    return KERNEL_STATUS_PARAM_INVALID;
  }

  size_t addr_index = 0;
  for (size_t i = 0; i < ctx.GetInputsSize(); ++i, ++addr_index) {
    Tensor *input = ctx.Input(i);
    KERNEL_CHECK_NULLPTR(input, KERNEL_STATUS_PARAM_INVALID, "Get input[%zu] failed.", i)
    auto iter = ext_info_msg.unknown_shape_input_index_addr.find(static_cast<uint32_t>(i));
    if (iter != ext_info_msg.unknown_shape_input_index_addr.end()) {
      iter->second = io_addrs[addr_index];
      ge::RuntimeTensorDesc *tensor_desc =
        reinterpret_cast<ge::RuntimeTensorDesc *>(static_cast<uintptr_t>(io_addrs[addr_index]));
      std::vector<int64_t> dims;
      KERNEL_CHECK_FALSE((tensor_desc->shape[0] <= ge::kMaxDimSize), KERNEL_STATUS_PARAM_INVALID,
                         "Max shape size[%lld], but got input[%zu] shape size[%lld]", ge::kMaxDimSize, i,
                         tensor_desc->shape[0])
      GetDimsFromArrays(&(tensor_desc->shape[1]), static_cast<size_t>(tensor_desc->shape[0]), dims);
      auto shape = input->GetTensorShape();
      KERNEL_CHECK_NULLPTR(shape, KERNEL_STATUS_PARAM_INVALID, "Get input[%zu] shape failed.", i)
      shape->SetDimSizes(dims);
      input->SetData(reinterpret_cast<void *>(static_cast<uintptr_t>(tensor_desc->data_addr)));
    } else {
      input->SetData(reinterpret_cast<void *>(static_cast<uintptr_t>(io_addrs[addr_index])));
    }

    if (ext_info_msg.unknown_shape) {
      std::vector<int64_t> dims;
      GetDimsFromShapeAndType(ext_info_msg.input_shape_and_type[i], dims);
      auto shape = input->GetTensorShape();
      KERNEL_CHECK_NULLPTR(shape, KERNEL_STATUS_PARAM_INVALID, "Get input[%zu] shape failed.", i)
      shape->SetDimSizes(dims);
    }

    KERNEL_CHECK_FALSE((input->NumElements() >= 0), KERNEL_STATUS_PARAM_INVALID,
                       "Input[%zu] data elements number must be >= 0, "
                       "got size[%lld].",
                       i, input->NumElements());
    input->SetDataSize(std::max(uint64_t(0), static_cast<uint64_t>(input->CalcDataSizeByShape())));
    KERNEL_LOG_INFO("Set input[%zu] addr[%lu] success.", i, io_addrs[addr_index]);
  }

  bool no_tiling = ext_info_msg.unknown_shape_output_index_addr.empty();

  for (size_t i = 0; i < ctx.GetOutputsSize(); i++, addr_index++) {
    Tensor *output = ctx.Output(i);
    KERNEL_CHECK_NULLPTR(output, KERNEL_STATUS_PARAM_INVALID, "Get output[%zu] failed.", i)
    auto iter = ext_info_msg.unknown_shape_output_index_addr.find(static_cast<uint32_t>(i));
    if (iter != ext_info_msg.unknown_shape_output_index_addr.end()) {
      iter->second = io_addrs[addr_index];
      ge::RuntimeTensorDesc *tensor_desc =
        reinterpret_cast<ge::RuntimeTensorDesc *>(static_cast<uintptr_t>(io_addrs[addr_index]));
      output->SetData(reinterpret_cast<void *>(static_cast<uintptr_t>(tensor_desc->data_addr)));
    } else {
      output->SetData(reinterpret_cast<void *>(static_cast<uintptr_t>(io_addrs[addr_index])));
    }

    if (ext_info_msg.unknown_shape) {
      std::vector<int64_t> dims;
      GetDimsFromShapeAndType(ext_info_msg.output_shape_and_type[i], dims);
      auto shape = output->GetTensorShape();
      KERNEL_CHECK_NULLPTR(shape, KERNEL_STATUS_PARAM_INVALID, "Get output[%zu] shape failed.", i)
      shape->SetDimSizes(dims);
    }

    KERNEL_CHECK_FALSE((ext_info_msg.unknown_shape || (!no_tiling) || (output->NumElements() >= 0)),
                       KERNEL_STATUS_PARAM_INVALID,
                       "Output[%zu] data elements number must be >= 0 "
                       "when known shape, got size[%lld].",
                       i, output->NumElements());
    output->SetDataSize(std::max(uint64_t(0), static_cast<uint64_t>(output->CalcDataSizeByShape())));
    KERNEL_LOG_INFO("Set output[%zu] addr[%lu] success.", i, io_addrs[addr_index]);
  }
  KERNEL_LOG_INFO("Update tensor info success.");
  return KERNEL_STATUS_OK;
}

/*
 * parse extend tensor shape types information.
 */
uint32_t CpuKernelCache::ParseExtShapeType(const FWKAdapter::ExtInfo *ext_info, bool &unknown_shape) const {
  if (ext_info->infoLen != sizeof(int32_t)) {
    KERNEL_LOG_ERROR(
      "Parse extend shape type failed, as info length must be [%zu], but got "
      "[%u].",
      sizeof(int32_t), ext_info->infoLen);
    return KERNEL_STATUS_PARAM_INVALID;
  }

  unknown_shape = true;
  KERNEL_LOG_INFO("Kernel has unknown shape.");
  return KERNEL_STATUS_OK;
}

/*
 * parse extend tensor shape and types information.
 */
uint32_t CpuKernelCache::ParseExtShapeAndType(bool unknown_shape, FWKAdapter::ExtInfo *ext_info,
                                              std::vector<FWKAdapter::ShapeAndType *> &shape_and_type) const {
  if (!unknown_shape) {
    return KERNEL_STATUS_OK;
  }
  uint32_t size = (ext_info->infoLen) / sizeof(FWKAdapter::ShapeAndType);
  KERNEL_LOG_INFO("Parse extend shape and type, size[%u].", size);
  uint32_t check = (ext_info->infoLen) % sizeof(FWKAdapter::ShapeAndType);
  if (check != 0) {
    KERNEL_LOG_ERROR(
      "Parse extend info length[%u] failed, must be integer multiple of the "
      "[%zu].",
      ext_info->infoLen, sizeof(FWKAdapter::ShapeAndType));
    return KERNEL_STATUS_PARAM_INVALID;
  }

  auto shapes = reinterpret_cast<FWKAdapter::ShapeAndType *>(ext_info->infoMsg);
  for (uint32_t index = 0; index < size; ++index) {
    shape_and_type.emplace_back(&shapes[index]);
  }
  return KERNEL_STATUS_OK;
}

/*
 * parse extend session information.
 */
uint32_t CpuKernelCache::ParseExtSessionInfo(FWKAdapter::ExtInfo *ext_info, uint64_t &kernel_id) const {
  // no overflow
  KERNEL_LOG_INFO("Parse extend session info.");
  auto need_len = sizeof(SessionInfo);
  if (ext_info->infoLen != need_len) {
    KERNEL_LOG_ERROR(
      "Parse extend session info failed, as info length must be "
      "[%zu], but got [%u].",
      sizeof(SessionInfo), ext_info->infoLen);
    return KERNEL_STATUS_PARAM_INVALID;
  }

  auto session = reinterpret_cast<SessionInfo *>(ext_info->infoMsg);
  kernel_id = session->kernelId;
  return KERNEL_STATUS_OK;
}

/*
 * get bit status.
 */
bool CpuKernelCache::GetBitStatus(uint64_t num, uint64_t pos) { return ((num & (1 << pos)) != 0); }

/*
 * parse bitmap information.
 */
uint32_t CpuKernelCache::ParseExtBitMap(const FWKAdapter::ExtInfo *ext_info, bool &unknown_shape) {
  if (ext_info->infoLen != sizeof(int64_t)) {
    KERNEL_LOG_ERROR(
      "Parse extend bitmap failed, as info length must be [%zu], but got "
      "[%u].",
      sizeof(int64_t), ext_info->infoLen);
    return KERNEL_STATUS_PARAM_INVALID;
  }

  uint64_t bit_map = *(reinterpret_cast<const int64_t *>(ext_info->infoMsg));
  unknown_shape = (!GetBitStatus(bit_map, 0));
  KERNEL_LOG_INFO("Unknown_shape_ is [%d].", unknown_shape);
  return KERNEL_STATUS_OK;
}

// parse async wait info
uint32_t CpuKernelCache::ParseAsyncWait(FWKAdapter::ExtInfo *ext_info, uint8_t &wait_type, uint32_t &wait_id) const {
  if (ext_info->infoLen != sizeof(FWKAdapter::AsyncWait)) {
    KERNEL_LOG_ERROR("Parse extend async wait failed, as info length must be [%zu], but got [%u].",
                     sizeof(FWKAdapter::AsyncWait), ext_info->infoLen);
    return KERNEL_STATUS_PARAM_INVALID;
  }
  FWKAdapter::AsyncWait *async_info = reinterpret_cast<FWKAdapter::AsyncWait *>(ext_info->infoMsg);
  wait_type = async_info->waitType;
  wait_id = async_info->waitId;
  KERNEL_LOG_INFO("async wait type [%u], notify_id[%u].", wait_type, wait_id);
  return KERNEL_STATUS_OK;
}

uint32_t CpuKernelCache::ParseExtUnknownShapeIndex(FWKAdapter::ExtInfo *ext_info,
                                                   std::map<uint32_t, uint64_t> &unknown_shape_index_addr) const {
  if (ext_info->infoLen % sizeof(uint32_t) != 0) {
    KERNEL_LOG_ERROR(
      "Parse unknown shape index extend info length[%u] failed, must be "
      "integer multiple of the [%zu].",
      ext_info->infoLen, sizeof(uint32_t));
    return KERNEL_STATUS_PARAM_INVALID;
  }
  uint32_t size = ext_info->infoLen / sizeof(uint32_t);
  KERNEL_LOG_INFO("Parse extend unknown shape index, size[%u].", size);
  auto indexes = reinterpret_cast<uint32_t *>(ext_info->infoMsg);
  for (uint32_t i = 0U; i < size; ++i) {
    unknown_shape_index_addr[indexes[i]] = 0U;
  }
  return KERNEL_STATUS_OK;
}

/*
 * parse extend information.
 */
uint32_t CpuKernelCache::ParseExtMsg(AicpuParamHead *param_head, ExtInfoMsg &ext_info_msg) {
  KERNEL_LOG_INFO("Parse extend info and update shape begin.");
  uint32_t offset = 0;
  ext_info_msg.async_flag = false;
  FWKAdapter::ExtInfo *ext_info = nullptr;
  char *extInfo_buf = reinterpret_cast<char *>(static_cast<uintptr_t>(param_head->extInfoAddr));
  while (offset + sizeof(FWKAdapter::ExtInfo) <= param_head->extInfoLength) {
    ext_info = reinterpret_cast<FWKAdapter::ExtInfo *>(extInfo_buf + offset);
    if (ext_info == nullptr) {
      KERNEL_LOG_ERROR(
        "Extend info is nullptr, extInfo length[%u], extend info addr[%p], "
        "offset[%u].",
        param_head->extInfoLength, param_head->extInfoAddr, offset);
      return KERNEL_STATUS_PARAM_INVALID;
    }

    uint32_t ret = KERNEL_STATUS_OK;
    switch (ext_info->infoType) {
      case FWKAdapter::FWK_ADPT_EXT_SHAPE_TYPE:
        ret = ParseExtShapeType(ext_info, ext_info_msg.unknown_shape);
        break;
      case FWKAdapter::FWK_ADPT_EXT_INPUT_SHAPE:
        ret = ParseExtShapeAndType(ext_info_msg.unknown_shape, ext_info, ext_info_msg.input_shape_and_type);
        break;
      case FWKAdapter::FWK_ADPT_EXT_OUTPUT_SHAPE:
        ret = ParseExtShapeAndType(ext_info_msg.unknown_shape, ext_info, ext_info_msg.output_shape_and_type);
        break;
      case FWKAdapter::FWK_ADPT_EXT_SESSION_INFO:
        ext_info_msg.has_sess_info = true;
        ret = ParseExtSessionInfo(ext_info, ext_info_msg.kernel_id);
        break;
      case FWKAdapter::FWK_ADPT_EXT_BITMAP:
        ret = ParseExtBitMap(ext_info, ext_info_msg.unknown_shape);
        break;
      case FWKAdapter::FWK_ADPT_EXT_ASYNCWAIT: {
        ret = ParseAsyncWait(ext_info, ext_info_msg.wait_type, ext_info_msg.wait_id);
        bool flag = ((ret == KERNEL_STATUS_OK) &&
                     (ext_info_msg.wait_type != FWKAdapter::FWKExtWaitType::FWK_ADPT_WAIT_TYPE_NULL) &&
                     (ext_info_msg.wait_type != FWKAdapter::FWKExtWaitType::FWK_ADPT_WAIT_TYPE_INVALID));
        if (flag) {
          ext_info_msg.async_flag = true;
        }
        break;
      }
      case FWKAdapter::FWK_ADPT_EXT_UNKNOWN_SHAPE_INPUT_INDEX:
        ret = ParseExtUnknownShapeIndex(ext_info, ext_info_msg.unknown_shape_input_index_addr);
        break;
      case FWKAdapter::FWK_ADPT_EXT_UNKNOWN_SHAPE_OUTPUT_INDEX:
        ret = ParseExtUnknownShapeIndex(ext_info, ext_info_msg.unknown_shape_output_index_addr);
        break;
      default:
        KERNEL_LOG_INFO("Ignore infoType[%d], infoLen[%u].", ext_info->infoType, ext_info->infoLen);
        break;
    }

    if (ret != KERNEL_STATUS_OK) {
      return ret;
    }

    // not overflow
    offset += FWKAdapter::kExtInfoHeadSize;
    offset += ext_info->infoLen;
  }

  return KERNEL_STATUS_OK;
}

/*
 * parse io address.
 */
uint32_t CpuKernelCache::ParseIoAddr(AicpuParamHead *param_head, std::vector<uint64_t> &io_addrs, char *&nodedef,
                                     uint32_t &nodedef_len) const {
  auto param_base = reinterpret_cast<char *>(param_head);
  char *extend_param_base = param_base + sizeof(AicpuParamHead);
  uint32_t extend_param_len = param_head->length - sizeof(AicpuParamHead);

  if (param_head->ioAddrNum > 0) {
    if (param_head->ioAddrNum > kMaxIoAddrNumParamLen) {
      KERNEL_LOG_ERROR("Param ioAddrNum[%u] is over %u.", param_head->ioAddrNum, kMaxIoAddrNumParamLen);
      return KERNEL_STATUS_PARAM_INVALID;
    }

    uint32_t addr_len = param_head->ioAddrNum * sizeof(uint64_t);
    if (extend_param_len < addr_len) {
      KERNEL_LOG_ERROR(
        "Extend param is not enough for io addr, ioAddrNum[%u], "
        "extend_param_len[%u].",
        param_head->ioAddrNum, extend_param_len);
      return KERNEL_STATUS_PARAM_INVALID;
    }

    auto io_addr_base = reinterpret_cast<uint64_t *>(extend_param_base);
    for (uint32_t i = 0; i < param_head->ioAddrNum; ++i) {
      io_addrs.push_back(io_addr_base[i]);
    }
    extend_param_base = extend_param_base + addr_len;
    extend_param_len -= addr_len;
  }

  if (extend_param_len < sizeof(uint32_t)) {
    KERNEL_LOG_ERROR(
      "Extend param is not enough for addr, needLen[%zu], "
      "extend_param_len[%u].",
      sizeof(uint32_t), extend_param_len);
    return KERNEL_STATUS_PARAM_INVALID;
  }

  nodedef_len = *reinterpret_cast<uint32_t *>(extend_param_base);
  extend_param_base += sizeof(uint32_t);
  nodedef = extend_param_base;
  KERNEL_LOG_INFO("Parse io addr success, io number[%zu], nodedef length[%u].", io_addrs.size(), nodedef_len);
  return KERNEL_STATUS_OK;
}

/*
 * get cpu kernel context from cache
 */
std::shared_ptr<CpuKernelContext> CpuKernelCache::GetCpuKernelContext(bool has_sess_info, uint64_t kernel_id,
                                                                      const char *nodedef, uint32_t nodedef_len,
                                                                      std::shared_ptr<NodeDef> &nodedef_proto) {
  std::shared_ptr<CpuKernelContext> ctx = nullptr;
  KERNEL_LOG_INFO("Get cpu kernel context begin, kernel id[%lu].", kernel_id);
  if (has_sess_info) {
    CpuCacheData *cache = GetCache(kernel_id);
    if (cache != nullptr) {
      KERNEL_LOG_INFO("Get kernel from cache success.");
      return cache->context;
    }
  }

  std::string str_data(nodedef, nodedef_len);
  nodedef_proto = CpuKernelUtils::CreateNodeDef();
  KERNEL_CHECK_NULLPTR(nodedef_proto, std::shared_ptr<CpuKernelContext>(nullptr), "Create node def failed.")
  if (!nodedef_proto->ParseFromString(str_data)) {
    return std::shared_ptr<CpuKernelContext>(nullptr);
  }

  CpuKernelContext *tmp = new (std::nothrow) CpuKernelContext(DEVICE);
  KERNEL_CHECK_NULLPTR(tmp, std::shared_ptr<CpuKernelContext>(nullptr), "Create context failed.")
  ctx = std::shared_ptr<CpuKernelContext>(tmp);
  uint32_t ret = ctx->Init(nodedef_proto.get());
  if (ret != KERNEL_STATUS_OK) {
    return std::shared_ptr<CpuKernelContext>(nullptr);
  }

  if (has_sess_info) {
    CpuCacheData *cache_ptr = new (std::nothrow) CpuCacheData(nodedef_proto, ctx);
    KERNEL_CHECK_NULLPTR(cache_ptr, std::shared_ptr<CpuKernelContext>(nullptr), "Create cpu cache data failed.")
    std::shared_ptr<CpuCacheData> cache_shared = std::shared_ptr<CpuCacheData>(cache_ptr);
    SetCache(kernel_id, cache_shared);
    KERNEL_LOG_INFO("Cache cpu kernel data success, kernel id[%lu].", kernel_id);
  }
  KERNEL_LOG_INFO("Get cpu kernel context success, kernel id[%lu].", kernel_id);
  return ctx;
}

/*
 * run kernel.
 */
int32_t CpuKernelCache::RunKernel(void *param) {
  AicpuParamHead *param_head = static_cast<AicpuParamHead *>(param);
  std::vector<uint64_t> io_addrs;
  char *nodedef = nullptr;
  uint32_t nodedef_len = 0;
  uint32_t ret = ParseIoAddr(param_head, io_addrs, nodedef, nodedef_len);
  if (ret != KERNEL_STATUS_OK) {
    return -1;
  }
  std::shared_ptr<ExtInfoMsg> ext_info_msg = nullptr;
  try {
    ext_info_msg = std::make_shared<ExtInfoMsg>();
  } catch (std::bad_alloc &) {
    KERNEL_LOG_ERROR("Create ExtInfoMsg failed");
    return -1;
  }
  ret = ParseExtMsg(param_head, *ext_info_msg);
  if (ret != KERNEL_STATUS_OK) {
    return -1;
  }

  std::shared_ptr<NodeDef> nodedef_proto = nullptr;
  auto ctx =
    GetCpuKernelContext(ext_info_msg->has_sess_info, ext_info_msg->kernel_id, nodedef, nodedef_len, nodedef_proto);
  KERNEL_CHECK_NULLPTR(ctx, KERNEL_STATUS_INNER_ERROR, "Get cpu kernel context from buff failed.")

  ret = UpdateTensor(io_addrs, *ext_info_msg, *ctx);
  if (ret != KERNEL_STATUS_OK) {
    return -1;
  }

  if (ext_info_msg->async_flag) {
    ret = CpuKernelRegister::Instance().RunCpuKernelAsync(
      *ctx, ext_info_msg->wait_type, ext_info_msg->wait_id,
      [&, ctx, ext_info_msg]() { return UpdateFWKOutputShape(*ext_info_msg, *ctx); });
  } else {
    ret = CpuKernelRegister::Instance().RunCpuKernel(*ctx);
    if (ret != KERNEL_STATUS_OK) {
      return -1;
    }
    ret = UpdateFWKOutputShape(*ext_info_msg, *ctx);
  }
  if (ret == KERNEL_STATUS_END_OF_SEQUENCE) {
    return ret;
  }
  if (ret != KERNEL_STATUS_OK) {
    return -1;
  }
  return 0;
}

/*
 * run kernel with blockdim info.
 */
int32_t CpuKernelCache::RunCpuKernelWithBlock(void *param, struct BlkDimInfo *blkdim_info) {
  AicpuParamHead *param_head = static_cast<AicpuParamHead *>(param);
  std::vector<uint64_t> io_addrs;
  char *nodedef = nullptr;
  uint32_t nodedef_len = 0;
  uint32_t ret = ParseIoAddr(param_head, io_addrs, nodedef, nodedef_len);
  if (ret != KERNEL_STATUS_OK) {
    return -1;
  }
  std::shared_ptr<ExtInfoMsg> ext_info_msg = nullptr;
  try {
    ext_info_msg = std::make_shared<ExtInfoMsg>();
  } catch (std::bad_alloc &) {
    KERNEL_LOG_ERROR("Create ExtInfoMsg failed");
    return -1;
  }
  ret = ParseExtMsg(param_head, *ext_info_msg);
  if (ret != KERNEL_STATUS_OK) {
    return -1;
  }

  std::shared_ptr<NodeDef> nodedef_proto = nullptr;
  auto ctx = GetCpuKernelContextWithBlock(ext_info_msg, nodedef, nodedef_len, nodedef_proto, blkdim_info);
  KERNEL_CHECK_NULLPTR(ctx, KERNEL_STATUS_INNER_ERROR, "Get cpu kernel context from buff failed.")

  ret = UpdateTensor(io_addrs, *ext_info_msg, *ctx);
  if (ret != KERNEL_STATUS_OK) {
    return -1;
  }

  if (ext_info_msg->async_flag) {
    ret = CpuKernelRegister::Instance().RunCpuKernelAsync(
      *ctx, ext_info_msg->wait_type, ext_info_msg->wait_id,
      [&, ctx, ext_info_msg]() { return UpdateFWKOutputShape(*ext_info_msg, *ctx); });
  } else {
    ret = CpuKernelRegister::Instance().RunCpuKernel(*ctx);
    if (ret != KERNEL_STATUS_OK) {
      return -1;
    }
    ret = UpdateFWKOutputShape(*ext_info_msg, *ctx);
  }
  if (ret != KERNEL_STATUS_OK) {
    return -1;
  }
  return 0;
}
/*
 * get cpu kernel context from cache
 */
std::shared_ptr<CpuKernelContext> CpuKernelCache::GetCpuKernelContextWithBlock(std::shared_ptr<ExtInfoMsg> extInfoMsg,
                                                                               const char *nodedef,
                                                                               uint32_t nodedef_len,
                                                                               std::shared_ptr<NodeDef> &nodedef_proto,
                                                                               struct BlkDimInfo *blkdim_info) {
  std::shared_ptr<CpuKernelContext> ctx = nullptr;
  KERNEL_LOG_INFO("Get cpu kernel context with block info begin. kernel id[%lu]", extInfoMsg->kernel_id);
  if (extInfoMsg->has_sess_info && blkdim_info->blockNum == 1) {
    CpuCacheData *cache = GetCache(extInfoMsg->kernel_id);
    if (cache != nullptr) {
      KERNEL_LOG_INFO("Get kernel from cache success.");
      return cache->context;
    }
  }
  std::string str_data(nodedef, nodedef_len);
  nodedef_proto = CpuKernelUtils::CreateNodeDef();
  KERNEL_CHECK_NULLPTR(nodedef_proto, std::shared_ptr<CpuKernelContext>(nullptr),
                       "Create node def with block info failed.")
  if (!nodedef_proto->ParseFromString(str_data)) {
    return std::shared_ptr<CpuKernelContext>(nullptr);
  }

  if (blkdim_info->blockNum != 1) {
    auto blockNum = CpuKernelUtils::CreateAttrValue();
    blockNum->SetInt(blkdim_info->blockNum);
    nodedef_proto->AddAttrs("block_num", blockNum.get());

    auto blockid = CpuKernelUtils::CreateAttrValue();
    blockid->SetInt(blkdim_info->blockId);
    nodedef_proto->AddAttrs("block_id", blockid.get());
    KERNEL_LOG_INFO("AddAttrs block info , blockNum[%u] blockId[%u].", blkdim_info->blockNum, blkdim_info->blockId);
  }

  CpuKernelContext *tmp = new (std::nothrow) CpuKernelContext(DEVICE);
  KERNEL_CHECK_NULLPTR(tmp, std::shared_ptr<CpuKernelContext>(nullptr), "Create context with block info failed.")
  ctx = std::shared_ptr<CpuKernelContext>(tmp);
  uint32_t ret = ctx->Init(nodedef_proto.get());
  if (ret != KERNEL_STATUS_OK) {
    return std::shared_ptr<CpuKernelContext>(nullptr);
  }

  if (extInfoMsg->has_sess_info) {
    CpuCacheData *cache_ptr = new (std::nothrow) CpuCacheData(nodedef_proto, ctx);
    KERNEL_CHECK_NULLPTR(cache_ptr, std::shared_ptr<CpuKernelContext>(nullptr), "Create cpu cache data failed.")
    std::shared_ptr<CpuCacheData> cache_shared = std::shared_ptr<CpuCacheData>(cache_ptr);
    SetCache(extInfoMsg->kernel_id, cache_shared);
    KERNEL_LOG_INFO("Cache cpu kernel data success. kernel id[%lu]", extInfoMsg->kernel_id);
  }
  KERNEL_LOG_INFO("Get cpu kernel context success. kernel id[%lu]", extInfoMsg->kernel_id);
  return ctx;
}
}  // namespace aicpu
