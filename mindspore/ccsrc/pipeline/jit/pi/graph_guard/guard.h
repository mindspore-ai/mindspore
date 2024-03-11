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
#ifndef MINDSPORE_PI_JIT_GUARD_H
#define MINDSPORE_PI_JIT_GUARD_H

#include <memory>
#include <vector>
#include <map>
#include <stack>
#include <string>
#include <utility>
#include <tuple>
#include "pybind11/pybind11.h"
#include "include/common/utils/python_adapter.h"
#include "pipeline/jit/pi/graph_guard/trace.h"
#include "pipeline/jit/pi/graph_guard/guard_utils.h"

namespace mindspore {
namespace pijit {

typedef enum _GuardLevel {
  GDeduce = 0,
  GId,
  GType,
  GAttr,
  GEqual,
} GuardLevel;

using GuardItemVector = std::vector<GuardItemPtr>;
using GuardItemMap = std::map<size_t, GuardItemPtr>;
using GuardCheckPoint = std::pair<GuardItemVector, GuardItemMap>;

class OptGuard : public std::enable_shared_from_this<OptGuard> {
 public:
  OptGuard();
  virtual ~OptGuard() = default;
  /// \brief check whether the variables guarded have been modified
  /// \param[in] frame python frame
  /// \param[in] print guard
  /// \param[in] cache to reuse the guard result
  /// \param[in] success to record the items to guard successfully
  /// \param[in] fail to record the items which fail to guard
  /// \param[in] perf to record the performance of guard
  /// \param[out] the variables have been modified
  virtual bool Check(const PyFrameObject *frame, bool print, std::map<size_t, PyObject *> *cache = nullptr,
                     std::map<size_t, bool> *success = nullptr, std::map<size_t, bool> *fail = nullptr,
                     bool perf = false);
  /// \brief guard the variable which has trace to retrieve
  /// \param[in] frame python frame
  /// \param[in] var to trace the path to retrieve the object
  /// \param[in] tp guard level
  /// \param[in] needSpecialize to check the content of buffer
  /// \param[in] recurseDepth to check the hierarchy element access like a.b.c by depth
  /// \param[out] whether to guard successfully
  virtual bool GuardOn(TracePtr var, GuardLevel tp = GuardLevel::GDeduce, bool needSpecialize = true,
                       int recurseDepth = 0);
  /// \brief add trace from guard, traces to replace in other guard
  /// \param[in] traces to replace in other guard
  /// \param[in] other guard with traces
  virtual void AddTraceFromGuard(const std::vector<TracePtr> &traces, std::shared_ptr<OptGuard> other);
  /// \brief return the description for the guard
  virtual std::string GetDescript();
  virtual void UpdateConfig(const std::map<std::string, bool> &bool_config,
                            const std::map<std::string, int> &int_config);
  virtual void Backup();
  virtual void Rollback();
  virtual void Pop();
  virtual bool IsEmpty() { return guardList_.size() == 0; }
  virtual bool MatchShape(std::shared_ptr<OptGuard> other);
  virtual std::vector<PyObject *> ApplyDynamicShape(PyFrameObject *frame);
  virtual void RevertDynamicShape(PyFrameObject *frame, const std::vector<PyObject *> &backup);

  std::string ToString() const;
  virtual const InfoPack &Info();
  virtual std::shared_ptr<OptGuard> Optimize();

 protected:
  void UpdateGuardList(GuardItemPtr item);
  std::vector<GuardItemPtr> guardList_;
  std::map<size_t, GuardItemPtr> guardMap_;
  std::stack<GuardCheckPoint> guardStack_;
  std::map<std::string, bool> bool_config_;
  std::map<std::string, int> int_config_;
  InfoPackPtr info_;
};
using OptGuardPtr = std::shared_ptr<OptGuard>;

class OptGuardPerf {
 public:
  static OptGuardPerf *GetGuardPerf();
  virtual void GetGuardPerfInfo(std::map<std::string, std::pair<size_t, size_t>> *guard_info,
                                std::map<std::string, std::pair<size_t, std::vector<size_t>>> *item_info,
                                std::map<std::string, std::pair<size_t, size_t>> *trace_info,
                                std::map<std::string, std::pair<size_t, size_t>> *guard_freq_info) const = 0;
  virtual void LogTracePerfStart() = 0;
  virtual void LogTracePerfEnd(Trace *trace, bool cache) = 0;
  virtual void LogItemPerfStart(int total_stage) = 0;
  virtual void LogItemPerfEnd(GuardItem *item, int stage) = 0;

 protected:
  OptGuardPerf() = default;
  virtual ~OptGuardPerf() = default;
};

extern const char kSpecializeScalar[];
extern const char kSpecializeTensor[];
extern const char kSpecializeContainer[];
extern const char kGuardRelaxCnt[];

}  // namespace pijit
}  // namespace mindspore

#endif  // MINDSPORE_PI_JIT_GUARD_H
