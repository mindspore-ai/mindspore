/**
 * Copyright 2024 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_HAL_PROFILER_ASCEND_PROFILING_H
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_HAL_PROFILER_ASCEND_PROFILING_H
#include <string>
#include <memory>
#include <map>
#include <vector>
#include "acl/acl_prof.h"
#include "include/backend/common/kernel_graph/kernel_graph.h"
#include "include/runtime/hardware_abstract/kernel_base/kernel.h"
#include "tools/profiler/profiling_data_dumper.h"
#include "tools/profiler/profiling.h"
#include "plugin/ascend/profiler/feature_mgr.h"

namespace mindspore {
namespace profiler {
namespace ascend {
constexpr uint64_t LevelNone = 0;
constexpr uint64_t Level0 = ACL_PROF_TASK_TIME_L0 | ACL_PROF_ACL_API;
constexpr uint64_t Level1 = ACL_PROF_TASK_TIME | ACL_PROF_ACL_API | ACL_PROF_HCCL_TRACE | ACL_PROF_AICORE_METRICS;
constexpr uint64_t Level2 = Level1 | ACL_PROF_AICPU | ACL_PROF_RUNTIME_API;

struct AscendProfilerConfig {
  uint32_t deviceId{0};
  uint32_t rankId{0};
  bool profileMemory{false};
  bool l2Cache{false};
  bool hbmDdr{false};
  bool pcie{false};
  bool sysIo{false};
  bool sysInterconnection{false};
  std::string hostSys;
  bool withStack{false};
  bool mstx{false};
  bool parallelStrategy{false};
  bool cpuTrace{false};
  bool npuTrace{false};
  bool recordShapes{false};
  std::string profilerLevel;
  std::string aicoreMetrics;
  std::string outputPath;
  std::string frameworkDataPath;
  std::vector<std::string> mstxDomainInclude;
  std::vector<std::string> mstxDomainExclude;

  AscendProfilerConfig() = default;
  AscendProfilerConfig(uint32_t deviceId, uint32_t rankId, bool profileMemory, bool l2Cache, bool hbmDdr, bool sysIo,
                       bool sysInterconnection, const std::string &hostSys, bool withStack, bool mstx,
                       bool parallelStrategy, bool pcie, bool recordShapes, const std::string &profilerLevel,
                       const std::string &aicoreMetrics, const std::string &outputPath,
                       const std::string &frameworkDataPath, const std::vector<std::string> &mstxDomainInclude,
                       const std::vector<std::string> &mstxDomainExclude)
      : deviceId(deviceId),
        rankId(rankId),
        profileMemory(profileMemory),
        l2Cache(l2Cache),
        hbmDdr(hbmDdr),
        pcie(pcie),
        sysIo(sysIo),
        sysInterconnection(sysInterconnection),
        hostSys(hostSys),
        withStack(withStack),
        mstx(mstx),
        parallelStrategy(parallelStrategy),
        recordShapes(recordShapes),
        profilerLevel(profilerLevel),
        aicoreMetrics(aicoreMetrics),
        outputPath(outputPath),
        frameworkDataPath(frameworkDataPath),
        mstxDomainInclude(mstxDomainInclude),
        mstxDomainExclude(mstxDomainExclude) {}

  void Clear() {
    deviceId = 0;
    rankId = 0;
    profileMemory = false;
    l2Cache = false;
    hbmDdr = false;
    pcie = false;
    sysIo = false;
    sysInterconnection = false;
    hostSys.clear();
    withStack = false;
    mstx = false;
    parallelStrategy = false;
    cpuTrace = false;
    npuTrace = false;
    recordShapes = false;
    profilerLevel.clear();
    aicoreMetrics.clear();
    outputPath.clear();
    frameworkDataPath.clear();
    mstxDomainInclude.clear();
    mstxDomainExclude.clear();
  }
};

class AscendProfiler : public Profiler {
 public:
  static std::shared_ptr<AscendProfiler> GetInstance();

  AscendProfiler() {}
  ~AscendProfiler() = default;
  AscendProfiler(const AscendProfiler &) = delete;
  AscendProfiler &operator=(const AscendProfiler &) = delete;
  void Init(const std::string &profiling_path, uint32_t device_id, const std::string &profiling_options) override;
  void Finalize() override;
  void Start() override;
  void Stop() override;
  void StepStart(uint64_t step_id, void *stream) override;
  void StepStop() override;
  void StepProfilingEnable(const bool enable_flag) override;
  void OpDataProducerEnd() override { return; }
  void MstxMark(const std::string &message, void *stream = nullptr,
                const std::string &domain_name = "default") override;
  int MstxRangeStart(const std::string &message, void *stream = nullptr,
                     const std::string &domain_name = "default") override;
  void MstxRangeEnd(int range_id, const std::string &domain_name = "default") override;
  bool EnableRecordShapes();

 protected:
  void SaveProfileData() override { return; }
  void ClearInst() override {
    config_.Clear();
    init_flag_ = false;
    aclConfig_ = nullptr;
    StepProfilingEnable(false);
  }

 private:
  void InitAscendProfilerConfig(const std::string &profiling_path, uint32_t device_id,
                                const std::string &profiling_options);
  void InitAclConfig();
  aclprofAicoreMetrics GetAicMetrics() const;
  aclprofAicoreMetrics CheckAicMetricsFeature(aclprofAicoreMetrics aic_metrics, const std::string &profiler_level);
  uint64_t GetAclProfMask(aclprofAicoreMetrics aicMetrics);
  void InitFwkMemProfiling();
  void StartFwkMemProfiling();
  void StopFwkMemProfiling();
  AscendProfilerConfig config_;
  aclprofStepInfo *aclProfStepInfo_{nullptr};
  aclprofConfig *aclConfig_{nullptr};
  aclrtStream aclStream_{nullptr};
};
}  // namespace ascend
}  // namespace profiler
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_HAL_PROFILER_ASCEND_PROFILING_H
