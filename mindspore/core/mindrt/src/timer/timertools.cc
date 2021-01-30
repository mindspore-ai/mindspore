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

#include "timer/timertools.h"
#include <csignal>
#include <ctime>
#include <unistd.h>
#include <sys/timerfd.h>
#include "evloop/evloop.h"

namespace mindspore {
using TimerPoolType = std::map<Duration, std::list<Timer>>;
static std::unique_ptr<TimerPoolType> g_timerPool;
static std::unique_ptr<EvLoop> g_timerEvLoop;
static Duration g_ticks(0);
static int g_runTimerFD(-1);
static int g_watchTimerFD(-1);
static SpinLock g_timersLock;
std::atomic_bool TimerTools::g_initStatus(false);
constexpr Duration SCAN_TIMERPOOL_DELAY = 30;
constexpr Duration WATCH_INTERVAL = 20;
constexpr unsigned int TIMER_LOG_INTERVAL = 6;
const static std::string TIMER_EVLOOP_THREADNAME = "HARES_LB_TMer";

namespace timer {
void ScanTimerPool(int fd, uint32_t events, void *data);
Duration NextTick(const std::map<Duration, std::list<Timer>> &timerPool) {
  if (!timerPool.empty()) {
    Duration first = timerPool.begin()->first;
    return first;
  }
  return 0;
}

void ExecTimers(const std::list<Timer> &timers) {
  for (const auto &timer : timers) {
    timer();
  }
}

void CreateTimerToLoop(const Duration &delay, const Duration &nextTick) {
  if (g_runTimerFD == -1) {
    g_runTimerFD = timerfd_create(CLOCK_MONOTONIC, TFD_NONBLOCK | TFD_CLOEXEC);
    if (g_runTimerFD >= 0) {
      int retval = g_timerEvLoop->AddFdEvent(g_runTimerFD, EPOLLIN, ScanTimerPool, nullptr);
      if (retval != BUS_OK) {
        ICTSBASE_LOG_STRING(ICTSBASE_LOG_COMMON_CODE, HLOG_LEVEL_ERROR, PID_LITEBUS_LOG, "add run timer event fail]%s",
                            "ID:%d", g_runTimerFD);
        close(g_runTimerFD);
        g_runTimerFD = -1;
        return;
      }
    } else {
      ICTSBASE_LOG_STRING(ICTSBASE_LOG_COMMON_CODE, HLOG_LEVEL_ERROR, PID_LITEBUS_LOG, "create run timer fd fail]%s",
                          "ID:%d", g_runTimerFD);
      g_runTimerFD = -1;
      return;
    }
    ICTSBASE_LOG_STRING(ICTSBASE_LOG_COMMON_CODE, HLOG_LEVEL_INFO, PID_LITEBUS_LOG, "create run timer fd success]%s",
                        "ID:%d", g_runTimerFD);
  }

  struct itimerspec it;
  it.it_interval.tv_sec = 0;
  it.it_interval.tv_nsec = 0;
  it.it_value.tv_sec = delay / SECTOMILLI;
  it.it_value.tv_nsec = (delay % SECTOMILLI) * MILLITOMICR * MICRTONANO;
  if (timerfd_settime(g_runTimerFD, 0, &it, nullptr) == -1) {
    ICTSBASE_LOG_STRING(ICTSBASE_LOG_COMMON_CODE, HLOG_LEVEL_ERROR, PID_LITEBUS_LOG, "start run timer fail]%s", "ID:%d",
                        g_runTimerFD);
    close(g_runTimerFD);
    g_runTimerFD = -1;
    return;
  }
}

void ScheduleTick(const std::map<Duration, std::list<Timer>> &timerPool) {
  Duration nextTick = NextTick(timerPool);

  if (nextTick != 0) {
    // 'tick' scheduled for an earlier time, not schedule current 'tick'
    if ((g_ticks == 0) || (nextTick < g_ticks)) {
      Duration nowTime = TimeWatch::Now();
      Duration delay = 0;

      if (nextTick > nowTime) {
        delay = nextTick - nowTime;
        g_ticks = nextTick;
        CreateTimerToLoop(delay, nextTick);
      } else {
        delay = SCAN_TIMERPOOL_DELAY;
        g_ticks = delay + nowTime;
        CreateTimerToLoop(delay, nextTick);
        ICTSBASE_LOG_STRING(ICTSBASE_LOG_COMMON_CODE, HLOG_LEVEL_DEBUG, PID_LITEBUS_LOG,
                            "run timer immediately {nextTick, now time}= %s ", "{%" PRIu64 ", %" PRIu64 "}", nextTick,
                            nowTime);
      }
    }
  }
}

// select timeout timers
void ScanTimerPool(int fd, uint32_t events, void *data) {
  std::list<Timer> outTimer;
  uint64_t count;

  if ((g_runTimerFD != fd) || !(events & EPOLLIN)) {
    ICTSBASE_LOG_STRING(ICTSBASE_LOG_COMMON_CODE, HLOG_LEVEL_INFO, PID_LITEBUS_LOG,
                        "run timer fd or events err {g_runTimerFD, fd, events}= %s ", "{%d, %d, %u}", g_runTimerFD, fd,
                        events);
    return;
  }

  if (read(fd, &count, sizeof(uint64_t)) < 0) {
    return;
  }
  g_timersLock.Lock();
  Duration now = TimeWatch::Now();
  auto it = g_timerPool->begin();
  while (it != g_timerPool->end()) {
    if (it->first > now) {
      break;
    }
    outTimer.splice(outTimer.end(), (*g_timerPool)[it->first]);
    ++it;
  }
  // delete timed out timer
  (void)g_timerPool->erase(g_timerPool->begin(), g_timerPool->upper_bound(now));
  g_ticks = 0;
  ScheduleTick(*g_timerPool);
  g_timersLock.Unlock();

  ExecTimers(outTimer);
  outTimer.clear();
}

void CheckPassedTimer(int fd, uint32_t events, void *data) {
  std::list<Timer> passTimer;
  static unsigned long watchTimes = 0;
  uint64_t count;

  if ((g_watchTimerFD != fd) || !(events & EPOLLIN)) {
    ICTSBASE_LOG_STRING(ICTSBASE_LOG_COMMON_CODE, HLOG_LEVEL_INFO, PID_LITEBUS_LOG,
                        "check timer fd or events err {g_watchTimerFD, fd, events}= %s ", "{%d, %d, %u}",
                        g_watchTimerFD, fd, events);
    return;
  }
  if (read(fd, &count, sizeof(uint64_t)) < 0) {
    return;
  }
  g_timersLock.Lock();
  Duration now = TimeWatch::Now();
  ++watchTimes;

  for (auto it = g_timerPool->begin(); it != g_timerPool->end(); ++it) {
    if (it->first > now) {
      break;
    }
    passTimer.splice(passTimer.end(), (*g_timerPool)[it->first]);
  }
  // delete passed timer
  if (passTimer.size() > 0) {
    ICTSBASE_LOG_STRING(ICTSBASE_LOG_COMMON_CODE, HLOG_LEVEL_DEBUG, PID_LITEBUS_LOG,
                        "fire pass timer {pass size, now, g_ticks}= %s ", "{%zd, %" PRIu64 ", %" PRIu64 "}",
                        passTimer.size(), now, g_ticks);
  }
  (void)g_timerPool->erase(g_timerPool->begin(), g_timerPool->upper_bound(now));
  if (g_ticks <= now) {
    g_ticks = 0;
  }

  if (g_timerPool->size() > 0) {
    if ((watchTimes % TIMER_LOG_INTERVAL == 0) && (passTimer.size() > 0) &&
        (now - g_timerPool->begin()->first > SECTOMILLI)) {
      ICTSBASE_LOG_STRING(ICTSBASE_LOG_COMMON_CODE, HLOG_LEVEL_INFO, PID_LITEBUS_LOG,
                          "timer info {pool size, pass size, now, g_ticks, poolTick, watchTimes}= %s ",
                          "{%zd, %zd, %" PRIu64 ", %" PRIu64 ", %" PRIu64 ", %lu}", g_timerPool->size(),
                          passTimer.size(), now, g_ticks, g_timerPool->begin()->first, watchTimes);
    }
  }

  ICTSBASE_LOG_STRING(ICTSBASE_LOG_COMMON_CODE, HLOG_LEVEL_DEBUG, PID_LITEBUS_LOG,
                      "timer info {pool size, pass size, now, g_ticks, poolTick, watchTimes}= %s ",
                      "{%zd, %zd, %" PRIu64 ", %" PRIu64 ", %" PRIu64 ", %lu}", g_timerPool->size(), passTimer.size(),
                      now, g_ticks, g_timerPool->begin()->first, watchTimes);

  ScheduleTick(*g_timerPool);
  g_timersLock.Unlock();

  ExecTimers(passTimer);
  passTimer.clear();
}

bool StartWatchTimer() {
  // create watch timer
  g_watchTimerFD = timerfd_create(CLOCK_MONOTONIC, TFD_NONBLOCK | TFD_CLOEXEC);
  if (g_watchTimerFD >= 0) {
    int retval = g_timerEvLoop->AddFdEvent(g_watchTimerFD, EPOLLIN, CheckPassedTimer, nullptr);
    if (retval != BUS_OK) {
      ICTSBASE_LOG_STRING(ICTSBASE_LOG_COMMON_CODE, HLOG_LEVEL_ERROR, PID_LITEBUS_LOG, "add watch timer event fail]%s",
                          "ID:%d", g_watchTimerFD);
      close(g_watchTimerFD);
      g_watchTimerFD = -1;
      return false;
    }
  } else {
    g_watchTimerFD = -1;
    return false;
  }

  // start watch timer
  struct itimerspec it;
  it.it_interval.tv_sec = WATCH_INTERVAL;
  it.it_interval.tv_nsec = 0;
  it.it_value.tv_sec = WATCH_INTERVAL;
  it.it_value.tv_nsec = 0;
  if (timerfd_settime(g_watchTimerFD, 0, &it, nullptr) == -1) {
    ICTSBASE_LOG_STRING(ICTSBASE_LOG_COMMON_CODE, HLOG_LEVEL_ERROR, PID_LITEBUS_LOG, "start watch timer fail]%s",
                        "ID:%d", g_watchTimerFD);
    close(g_watchTimerFD);
    g_watchTimerFD = -1;
    return false;
  }
  ICTSBASE_LOG_STRING(ICTSBASE_LOG_COMMON_CODE, HLOG_LEVEL_INFO, PID_LITEBUS_LOG,
                      "start watch timer success, {id}= %s ", "{%d}", g_watchTimerFD);
  return true;
}
}  // namespace timer

bool TimerTools::Initialize() {
  bool ret = true;
  g_timersLock.Lock();

  g_timerPool.reset(new (std::nothrow) TimerPoolType());
  if (g_timerPool == nullptr) {
    ICTSBASE_LOG0(ICTSBASE_LOG_COMMON_CODE, HLOG_LEVEL_ERROR, PID_LITEBUS_LOG, "timer pool new failed.");
    g_timersLock.Unlock();
    return false;
  }

  g_timerEvLoop.reset(new (std::nothrow) EvLoop());
  if (g_timerEvLoop == nullptr) {
    ICTSBASE_LOG0(ICTSBASE_LOG_COMMON_CODE, HLOG_LEVEL_ERROR, PID_LITEBUS_LOG, "ev new failed.");
    g_timersLock.Unlock();
    return false;
  }
  bool ok = g_timerEvLoop->Init(TIMER_EVLOOP_THREADNAME);
  if (!ok) {
    ICTSBASE_LOG0(ICTSBASE_LOG_COMMON_CODE, HLOG_LEVEL_ERROR, PID_LITEBUS_LOG, "ev init failed.");
    g_timerEvLoop = nullptr;
    g_timersLock.Unlock();
    return false;
  }
  ret = timer::StartWatchTimer();
  g_timersLock.Unlock();
  g_initStatus.store(true);
  ICTSBASE_LOG0(ICTSBASE_LOG_COMMON_CODE, HLOG_LEVEL_INFO, PID_LITEBUS_LOG, "Timer init succ.");
  return ret;
}

void TimerTools::Finalize() {
  if (g_initStatus.load() == false) {
    ICTSBASE_LOG0(ICTSBASE_LOG_COMMON_CODE, HLOG_LEVEL_INFO, PID_LITEBUS_LOG, "no need for Timer Finalize.");
    return;
  }
  g_initStatus.store(false);

  ICTSBASE_LOG0(ICTSBASE_LOG_COMMON_CODE, HLOG_LEVEL_INFO, PID_LITEBUS_LOG, "Timer Finalize.");
  g_timersLock.Lock();
  if (g_runTimerFD >= 0) {
    close(g_runTimerFD);
    ICTSBASE_LOG_STRING(ICTSBASE_LOG_COMMON_CODE, HLOG_LEVEL_INFO, PID_LITEBUS_LOG, "run timer close {ID}=%s", "{%d}",
                        g_runTimerFD);
    g_runTimerFD = -1;
  }
  if (g_watchTimerFD >= 0) {
    close(g_watchTimerFD);
    ICTSBASE_LOG_STRING(ICTSBASE_LOG_COMMON_CODE, HLOG_LEVEL_INFO, PID_LITEBUS_LOG, "watch timer close {ID}=%s", "{%d}",
                        g_watchTimerFD);
    g_watchTimerFD = -1;
  }
  g_timersLock.Unlock();
}

Timer TimerTools::AddTimer(const Duration &duration, const AID &aid, const std::function<void()> &thunk) {
  if (g_initStatus.load() == false) {
    return Timer();
  }
  if (duration == 0) {
    thunk();
    return Timer();
  }
  static std::atomic<uint64_t> id(1);
  TimeWatch timeWatch = TimeWatch::In(duration);
  Timer timer(id.fetch_add(1), timeWatch, aid, thunk);

  // Add the timer to timerpoll and Schedule it
  g_timersLock.Lock();

  if (g_timerPool->size() == 0 || timer.GetTimeWatch().Time() < g_timerPool->begin()->first) {
    (*g_timerPool)[timer.GetTimeWatch().Time()].push_back(timer);
    timer::ScheduleTick(*g_timerPool);
  } else {
    if (!(g_timerPool->size() >= 1)) {
      ICTSBASE_LOG_STRING(ICTSBASE_LOG_COMMON_CODE, HLOG_LEVEL_ERROR, PID_LITEBUS_LOG,
                          "g_timerPool size invalid {size}=%s", "{%zd}", g_timerPool->size());
    }
    (*g_timerPool)[timer.GetTimeWatch().Time()].push_back(timer);
  }
  g_timersLock.Unlock();

  return timer;
}

bool TimerTools::Cancel(const Timer &timer) {
  if (g_initStatus.load() == false) {
    return false;
  }

  bool canceled = false;
  g_timersLock.Lock();
  Duration duration = timer.GetTimeWatch().Time();
  if (g_timerPool->count(duration) > 0) {
    canceled = true;
    (*g_timerPool)[duration].remove(timer);
    if ((*g_timerPool)[duration].empty()) {
      (void)(g_timerPool->erase(duration));
    }
  }
  g_timersLock.Unlock();

  return canceled;
}
}  // namespace mindspore
