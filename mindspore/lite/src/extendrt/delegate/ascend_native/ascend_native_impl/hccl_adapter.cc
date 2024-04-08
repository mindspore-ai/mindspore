/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#include "extendrt/delegate/ascend_native/ascend_native_impl/hccl_adapter.h"
#include <mpi.h>
#include <string>
#include <algorithm>
#include "src/common/log_adapter.h"
#include "hccl/hccl.h"
#include "hccl/hccl_types.h"
#include "acl/acl.h"

namespace mindspore {
#define HCCLROOTRANKCHECK(ret)                                                                 \
  do {                                                                                         \
    if (ret != HCCL_SUCCESS && ret != HCCL_E_PARA) {                                           \
      MS_LOG(ERROR) << "hccl interface return err:" << ret << " msg=" << aclGetRecentErrMsg(); \
      return ret;                                                                              \
    }                                                                                          \
  } while (0)

#define HCCLCHECK(ret, rank_id)                                                         \
  do {                                                                                  \
    if (ret != HCCL_SUCCESS) {                                                          \
      MS_LOG(ERROR) << "hccl interface return error rank=" << rank_id << " err=" << ret \
                    << " msg=" << aclGetRecentErrMsg();                                 \
      return ret;                                                                       \
    }                                                                                   \
  } while (0)

#define ACLCHECK(x)                                                             \
  do {                                                                          \
    if (x != ACL_SUCCESS) {                                                     \
      MS_LOG(ERROR) << "acl error " << x << "(" << aclGetRecentErrMsg() << ")"; \
    }                                                                           \
  } while (0);

static HcclAdapter hccl;

HcclAdapter::HcclAdapter() {
  MPI_Init(nullptr, nullptr);
  MPI_Comm_size(MPI_COMM_WORLD, &rank_size_);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank_id_);
  auto rank_str = std::to_string(rank_id_);
  setenv("ASCEND_DEVICE_ID", rank_str.c_str(), 1);
}

HcclAdapter::~HcclAdapter() {
  int finalized;
  MPI_Finalized(&finalized);
  if (finalized == 0) {
    MPI_Finalize();
    aclFinalize();
  }
}

HcclAdapter &HcclAdapter::GetInstance() { return hccl; }

std::vector<int> HcclAdapter::getAviDevs(const char *devs) {
  std::vector<int> dev_ids;
  std::string use_devs(devs);
  std::string pattern = ",";
  std::string::size_type pos;
  use_devs += pattern;
  size_t val_size = use_devs.size();
  for (size_t i = 0; i < val_size; ++i) {
    pos = use_devs.find(pattern, i);
    if (pos < val_size) {
      std::string s = use_devs.substr(i, pos);
      int tmp_rank = atoi(s.c_str());
      dev_ids.push_back(tmp_rank);
      i = pos + pattern.size() - 1;
    }
  }
  return dev_ids;
}

HcclResult HcclAdapter::get_mpi_proc() {
  // 获取当前进程在所属进程组的编号
  uint32_t dev_count;
  ACLCHECK(aclrtGetDeviceCount(&dev_count));
  int npus = dev_count;
  const char *devs = getenv("MS_ASCEND_DEVICES");
  npus = std::min(npus, rank_size_);
  if (devs != NULL) {
    auto dev_ids = getAviDevs(devs);
    npus = std::min(static_cast<int>(dev_count), static_cast<int>(dev_ids.size()));
    npus = std::min(npus, rank_size_);
    int local_rank = rank_id_ % npus;
    dev_id_ = dev_ids[local_rank];
  } else {
    dev_id_ = rank_id_ % npus;
  }
  return HCCL_SUCCESS;
}

HcclResult HcclAdapter::set_device_sat_mode() {
  const char *soc_name_ptr = aclrtGetSocName();
  if (soc_name_ptr == nullptr) {
    MS_LOG(ERROR) << "aclrtGetSocName failed";
    return HCCL_E_INTERNAL;
  }

  const std::string support_soc_name = "Ascend910B";
  if (support_soc_name.compare(0, support_soc_name.length(), soc_name_ptr, 0, support_soc_name.length()) == 0) {
    ACLCHECK(aclrtSetDeviceSatMode(ACL_RT_OVERFLOW_MODE_INFNAN));
  }
  return HCCL_SUCCESS;
}

HcclResult HcclAdapter::HcclInit() {
  auto ret = get_mpi_proc();
  if (ret != HCCL_SUCCESS) {
    MS_LOG(ERROR) << "fail to run mpi discovery";
    return ret;
  }
  auto res = aclrtSetDevice(dev_id_);
  if (res != ACL_SUCCESS) {
    MS_LOG(ERROR) << "fail initialize device (" << res << ")";
    return HCCL_E_RUNTIME;
  }
  ret = set_device_sat_mode();
  if (ret != 0) {
    printf("set_device_sat_mode execute failed, Detailed logs are stored in path: /root/ascend/log/");
    return ret;
  }

  // 在root_rank获取root_info
  if (rank_id_ == root_rank) {
    HCCLROOTRANKCHECK(HcclGetRootInfo(&comm_id_));
  }
  // 将root_info广播到通信域内的其他rank
  MPI_Bcast(&comm_id_, HCCL_ROOT_INFO_BYTES, MPI_CHAR, root_rank, MPI_COMM_WORLD);
  MPI_Barrier(MPI_COMM_WORLD);
  // 初始化集合通信域
  HCCLCHECK(HcclCommInitRootInfo(rank_size_, &comm_id_, rank_id_, &hccl_comm_), rank_id_);
  return HCCL_SUCCESS;
}

HcclResult HcclAdapter::AllGather(void *send_buff, void *recv_buff, uint64_t count, HcclDataType dataType,
                                  void *stream) {
  HcclResult res;
  res = HcclAllGather(send_buff, recv_buff, count, dataType, hccl_comm_, stream);
  HCCLCHECK(res, rank_id_);
  return res;
}

HcclResult HcclAdapter::AllSumReduce(void *send_buff, void *recv_buff, uint64_t count, void *stream) {
  HcclResult res;
  res = HcclAllReduce(send_buff, recv_buff, count, HCCL_DATA_TYPE_FP16, HCCL_REDUCE_SUM, hccl_comm_, stream);
  HCCLCHECK(res, rank_id_);
  return res;
}

HcclResult HcclAdapter::Sync() {
  MPI_Barrier(MPI_COMM_WORLD);
  return HCCL_SUCCESS;
}

HcclResult HcclAdapter::HcclSync(void *stream) {
  HcclBarrier(hccl_comm_, stream);
  return HCCL_SUCCESS;
}

void HcclAdapter::test_reduce(void *stream) {
  const std::vector<float> in1 = {1.0, 1.5, 2.0, 2.5, 3};
  const std::vector<float> in2 = {1.0, 2.1, 3.2, 4.3, -5.1};
  const std::vector<float> *in;
  if (rank_id_ == 0) {
    in = &in1;
  } else {
    in = &in2;
  }
  std::vector<aclFloat16> in_fp16;
  std::transform(in->begin(), in->end(), std::back_inserter(in_fp16), [](float x) { return aclFloatToFloat16(x); });
  void *in_fp16_dev;
  void *result_fp16_dev;
  size_t size = in_fp16.size() * sizeof(aclFloat16);
  aclError ret;
  ret = aclrtMalloc(&in_fp16_dev, size, ACL_MEM_MALLOC_HUGE_FIRST);
  ACLCHECK(ret)
  ret = aclrtMemcpy(in_fp16_dev, size, in_fp16.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
  ACLCHECK(ret)
  result_fp16_dev = in_fp16_dev;
  AllSumReduce(in_fp16_dev, result_fp16_dev, in_fp16.size(), stream);
  ret = aclrtSynchronizeStream(stream);
  ACLCHECK(ret)
  aclFloat16 *result_fp16 = reinterpret_cast<aclFloat16 *>(malloc(size));
  ret = aclrtMemcpy(result_fp16, size, result_fp16_dev, size, ACL_MEMCPY_DEVICE_TO_HOST);
  ACLCHECK(ret)
  std::vector<float> result;
  std::transform(result_fp16, result_fp16 + in_fp16.size(), std::back_inserter(result),
                 [](aclFloat16 x) { return aclFloat16ToFloat(x); });
  for (auto x : result) {
    std::cout << x << ",";
  }
  std::cout << std::endl;
  free(result_fp16);
  aclrtFree(in_fp16_dev);
}
}  // namespace mindspore
