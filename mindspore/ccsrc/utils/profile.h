/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_UTILS_PROFILE_H_
#define MINDSPORE_CCSRC_UTILS_PROFILE_H_

#include <map>
#include <string>
#include <fstream>
#include <iomanip>
#include <sstream>
#include "utils/log_adapter.h"

namespace mindspore {

struct TimeInfo;
using TimeInfoMap = std::map<std::string, const TimeInfo *>;

extern double GetTime();

class ProfileBase;

struct TimeInfo {
  explicit TimeInfo(double time = -1.0) : time_(time), dict_(nullptr), actionNum_(0) {}
  TimeInfo(const TimeInfo &) = delete;
  ~TimeInfo();

  double time_;
  TimeInfoMap *dict_;
  size_t actionNum_;
};

// Utility class for Profile.
class ProfContext {
  friend class Profile;
  friend class ProfileBase;
  friend class ProfTransaction;

 public:
  ProfContext(const std::string &name, ProfileBase *prof);
  ~ProfContext();

  ProfContext(const ProfContext &) = delete;
  ProfContext &operator=(const ProfContext &) = delete;

  void SetTime(double time) noexcept;
  void Insert(const std::string &name, const TimeInfo *time) noexcept;
  bool IsTopContext() const noexcept;

 private:
  std::string name_;
  ProfileBase *prof_;
  ProfContext *parent_;
  TimeInfo *time_info_;
};

class ProfileBase {
  friend class ProfContext;
  friend class ProfTransaction;

 public:
  ProfileBase();
  virtual ~ProfileBase();

  virtual void Print(void) {}
  virtual ProfContext *Step(const std::string &) { return nullptr; }
  virtual ProfContext *Lap(int) { return nullptr; }
  virtual void Pop(void) {}

  // top level profile context
  ProfContext context_;
  // profile context pointer, act as a stack pointer
  ProfContext *ctx_ptr_ = nullptr;
};

class Profile : public ProfileBase {
 public:
  Profile() = default;
  ~Profile() override = default;
  Profile(const Profile &) = delete;
  Profile &operator=(const Profile &) = delete;

  void Print(void) override;
  ProfContext *Step(const std::string &name) override;
  ProfContext *Lap(int count) override;
  void Pop(void) noexcept override;
};

class ProfTransaction {
 public:
  explicit ProfTransaction(const ProfileBase *prof);
  explicit ProfTransaction(ProfContext *const ctx) : ctx_(ctx) {}
  ProfTransaction(const ProfTransaction &) = delete;
  ~ProfTransaction();

  template <class Function>
  void operator-(const Function &func) {
    double start_time = GetTime();
    func();
    double end_time = GetTime();
    if (ctx_ != nullptr) {
      ctx_->SetTime(end_time - start_time);
    }
  }

 private:
  ProfContext *ctx_ = nullptr;
};

class NoProfTransaction {
 public:
  explicit NoProfTransaction(ProfileBase *prof) {}
  explicit NoProfTransaction(ProfContext *ctx) {}
  ~NoProfTransaction() = default;

  template <class Function>
  void operator-(const Function &func) {
    func();
  }
};

class DumpTime {
 public:
  ~DumpTime() {
    try {
      Save();
    } catch (const std::exception &e) {
      MS_LOG(ERROR) << "Cannot save file by profile::DumpTime::save";
    } catch (...) {
      MS_LOG(ERROR) << "Uncaught exception";
    }
  }
  DumpTime(const DumpTime &) = delete;
  DumpTime &operator=(const DumpTime &) = delete;
  static DumpTime &GetInstance() {
    static DumpTime instance;
    return instance;
  }
  void set_file_path(const std::string &save_path) { file_path_ = save_path; }
  void Record(const std::string &name, const double time, const bool is_start);
  void Save();

 private:
  DumpTime() = default;
  std::stringstream file_ss_;
  std::ofstream file_out_;
  std::string file_path_ = "./timeline.json";
};

struct TimeStat {
  TimeStat() {
    time_ = 0.0;
    count_ = 0;
  }
  ~TimeStat() = default;

  void operator+=(double t) {
    time_ += t;
    count_ += 1;
  }

  TimeStat operator+(double t) {
    TimeStat ts = *this;
    ts += t;
    return ts;
  }

  double time_;
  int count_;
};

class MsProfile {
 public:
  ~MsProfile() { Clear(); }

  static void Reset() { GetSingleton().Clear(); }

  static ProfileBase *GetProfile() {
    MsProfile &ms_prof = GetSingleton();
    if (ms_prof.profile_ == nullptr) {
#ifdef ENABLE_PROFILE
      ms_prof.profile_ = new Profile();
#else
      ms_prof.profile_ = new ProfileBase();
#endif
    }
    return ms_prof.profile_;
  }
  static void StatTime(const std::string &id, double time) { GetSingleton().time_stat_[id] += time; }

  static void Print();

 private:
  MsProfile() = default;

  static MsProfile &GetSingleton() {
    static MsProfile profile;
    return profile;
  }

  void Clear() {
    time_stat_.clear();
    if (profile_ != nullptr) {
      delete profile_;
      profile_ = nullptr;
    }
  }

  std::map<std::string, TimeStat> time_stat_;  // record time and count info from some activity
  ProfileBase *profile_ = nullptr;             // record hierarchical profile info
};

}  // namespace mindspore

#ifdef ENABLE_PROFILE
#define WITH(x) ProfTransaction(x) -
#else
#define WITH(x) NoProfTransaction(x) -
#endif

#endif  // MINDSPORE_CCSRC_UTILS_PROFILE_H_
