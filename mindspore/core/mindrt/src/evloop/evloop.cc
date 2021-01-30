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

#include <atomic>
#include <string>
#include <thread>

#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/eventfd.h>
#include <sys/socket.h>

#include <csignal>
#include <unistd.h>

#include "actor/buslog.h"
#include "evloop/evloop.h"
#include "utils/log_adapter.h"

namespace mindspore {

int EventLoopRun(EvLoop *evloop, int timeout) {
  int nevent = 0;
  struct epoll_event *events = nullptr;

  (void)sem_post(&evloop->semId);

  size_t size = 1;
  events = (struct epoll_event *)malloc(size);
  if (events == nullptr) {
    MS_LOG(ERROR) << "malloc events fail";
    return BUS_ERROR;
  }
  // 1. dest is valid 2. destsz equals to count and both are valid.
  // memset_s will always executes successfully.
  (void)memset(events, 0, size);

  while (!evloop->stopLoop) {
    /* free deleted event handlers */
    evloop->EventFreeDelEvents();

    MS_LOG(DEBUG) << "timeout:" << timeout << ",epoll_fd:" << evloop->efd;
    MS_LOG(DEBUG) << "nevent:" << nevent << ",epoll_fd:" << evloop->efd;
    if (nevent < 0) {
      if (errno != EINTR) {
        MS_LOG(ERROR) << "epoll_wait failed]epoll_fd:" << evloop->efd << ",errno:" << errno;
        free(events);
        return BUS_ERROR;
      } else {
        continue;
      }
    } else if (nevent > 0) {
      /* save the epoll modify in "stop" while dispatching handlers */
      evloop->HandleEvent(events, nevent);
    } else {
      MS_LOG(ERROR) << "epoll_wait failed]epoll_fd:" << evloop->efd << ",ret:0,errno:" << errno;
      evloop->stopLoop = 1;
    }

    if (evloop->stopLoop) {
      /* free deleted event handlers */
      evloop->EventFreeDelEvents();
    }
  }
  evloop->stopLoop = 0;
  MS_LOG(INFO) << "event epoll loop run end";
  free(events);
  return BUS_OK;
}

void *EvloopRun(void *arg) {
  if (arg == nullptr) {
    MS_LOG(ERROR) << "arg is null";
  } else {
    (void)EventLoopRun((EvLoop *)arg, -1);
  }
  return nullptr;
}

void QueueReadyCallback(int fd, uint32_t events, void *arg) {
  EvLoop *evloop = (EvLoop *)arg;
  if (evloop == nullptr) {
    MS_LOG(ERROR) << "evloop is null]fd:" << fd << ",events:" << events;
    return;
  }

  uint64_t count;
  if (read(evloop->queueEventfd, &count, sizeof(count)) == sizeof(count)) {
    // take out functions from the queue
    std::queue<std::function<void()>> q;

    evloop->queueMutex.lock();
    evloop->queue.swap(q);
    evloop->queueMutex.unlock();

    // invoke functions in the queue
    while (!q.empty()) {
      q.front()();
      q.pop();
    }
  }
}

void EvLoop::CleanUp() {
  if (queueEventfd != -1) {
    close(queueEventfd);
    queueEventfd = -1;
  }

  if (efd != -1) {
    close(efd);
    efd = -1;
  }
}

int EvLoop::AddFuncToEvLoop(std::function<void()> &&func) {
  // put func to the queue
  queueMutex.lock();
  queue.emplace(std::move(func));
  // return the queque size to send's caller.
  int result = queue.size();
  queueMutex.unlock();

  if (result == 1) {
    // wakeup event loop
    uint64_t one = 1;
    if (write(queueEventfd, &one, sizeof(one)) != sizeof(one)) {
      MS_LOG(WARNING) << "fail to write queueEventfd]fd:" << queueEventfd << ",errno:" << errno;
    }
  }

  return result;
}

bool EvLoop::Init(const std::string &threadName) {
  int retval = EventLoopCreate();
  if (retval != BUS_OK) {
    return false;
  }
  (void)sem_init(&semId, 0, 0);

  if (pthread_create(&loopThread, nullptr, EvloopRun, (void *)this) != 0) {
    MS_LOG(ERROR) << "pthread_create fail";
    Finish();
    return false;
  }
  // wait EvloopRun
  (void)sem_wait(&semId);
#if __GLIBC__ >= 2 && __GLIBC_MINOR__ >= 12
  std::string name = threadName;
  if (name.empty()) {
    name = "EventLoopThread";
  }
  retval = pthread_setname_np(loopThread, name.c_str());
  if (retval != 0) {
    ICTSBASE_LOG_STRING(ICTSBASE_LOG_COMMON_CODE, HLOG_LEVEL_INFO, PID_LITEBUS_LOG, "set pthread name fail]%s",
                        "name:%s,retval:%d", name.c_str(), retval);
  } else {
    ICTSBASE_LOG_STRING(ICTSBASE_LOG_COMMON_CODE, HLOG_LEVEL_INFO, PID_LITEBUS_LOG, "set pthread name success]%s",
                        "name:%s,loopThread:%lu", name.c_str(), loopThread);
  }
#endif

  return true;
}

void EvLoop::Finish() {
  if (loopThread) {
    void *threadResult = nullptr;
    StopEventLoop();
    int ret = pthread_join(loopThread, &threadResult);
    if (ret != 0) {
      MS_LOG(INFO) << "pthread_join loopThread fail";
    }
    loopThread = 0;
  }
  EventLoopDestroy();

  MS_LOG(INFO) << "stop loop succ";
}

EvLoop::~EvLoop() { Finish(); }

void EvLoop::DeleteEvent(int fd) {
  auto iter = events.find(fd);
  if (iter == events.end()) {
    MS_LOG(DEBUG) << "not found event]fd:" << fd;
    return;
  }
  MS_LOG(DEBUG) << "erase event]fd:%d" << fd;
  EventData *eventData = iter->second;
  if (eventData != nullptr) {
    delete eventData;
  }
  events.erase(fd);
}

EventData *EvLoop::FindEvent(int fd) {
  auto iter = events.find(fd);
  if (iter == events.end()) {
    return nullptr;
  }

  return iter->second;
}
void EvLoop::AddEvent(EventData *eventData) {
  if (!eventData) {
    return;
  }
  DeleteEvent(eventData->fd);
  events.emplace(eventData->fd, eventData);
}

int EvLoop::EventLoopCreate(void) { return BUS_OK; }

int EvLoop::AddFdEvent(int fd, uint32_t tEvents, EventHandler handler, void *data) { return BUS_OK; }

int EvLoop::DelFdEvent(int fd) {
  EventData *tev = nullptr;

  eventsLock.lock();
  tev = FindEvent(fd);
  if (tev == nullptr) {
    eventsLock.unlock();
    MS_LOG(DEBUG) << "event search fail]fd:" << fd << ",epollfd:" << efd;
    return BUS_ERROR;
  }
  events.erase(tev->fd);

  // Don't delete tev immediately, let's push it into deletedEvents, before next epoll_wait,we will free
  // all events in deletedEvents.
  AddDeletedEvents(tev);

  eventsLock.unlock();

  return BUS_OK;
}

int EvLoop::ModifyFdEvent(int fd, uint32_t tEvents) {
  struct epoll_event ev;
  EventData *tev = nullptr;

  tev = FindEvent(fd);
  if (tev == nullptr) {
    MS_LOG(ERROR) << "event lookup fail]fd:" << fd << ",events:" << tEvents;
    return BUS_ERROR;
  }

  // 1. dest is valid 2. destsz equals to count and both are valid.
  // memset_s will always executes successfully.
  (void)memset(&ev, 0, sizeof(ev));

  ev.events = tEvents;
  ev.data.ptr = tev;

  return BUS_OK;
}

void EvLoop::AddDeletedEvents(EventData *eventData) {
  // caller need check eventData is not nullptr
  std::list<EventData *> deleteEventList;

  // if fd not found, push eventData into deletedEvents[fd]
  std::map<int, std::list<EventData *>>::iterator fdIter = deletedEvents.find(eventData->fd);
  if (fdIter == deletedEvents.end()) {
    deletedEvents[eventData->fd].push_back(eventData);
    return;
  }

  // if fd found, check if same eventData ptr exists
  deleteEventList = fdIter->second;
  std::list<EventData *>::iterator eventIter = deleteEventList.begin();
  bool found = false;
  while (eventIter != deleteEventList.end()) {
    if (*eventIter == eventData) {
      MS_LOG(WARNING) << "fd has been deleted before]fd:" << eventData->fd << ",efd:" << efd;
      found = true;
      break;
    }
    ++eventIter;
  }

  // if found same eventData ptr, do nothing
  if (found) {
    return;
  }

  deletedEvents[eventData->fd].push_back(eventData);

  return;
}

void EvLoop::EventFreeDelEvents() {
  std::map<int, std::list<EventData *>>::iterator fdIter = deletedEvents.begin();
  while (fdIter != deletedEvents.end()) {
    std::list<EventData *> deleteEventList = fdIter->second;
    std::list<EventData *>::iterator eventIter = deleteEventList.begin();
    while (eventIter != deleteEventList.end()) {
      EventData *deleteEv = *eventIter;
      delete deleteEv;
      deleteEv = nullptr;
      ++eventIter;
    }
    deletedEvents.erase(fdIter++);
  }
  deletedEvents.clear();
}

int EvLoop::FindDeletedEvent(const EventData *tev) {
  std::map<int, std::list<EventData *>>::iterator fdIter = deletedEvents.find(tev->fd);
  if (fdIter == deletedEvents.end()) {
    return 0;
  }

  std::list<EventData *> deleteEventList = fdIter->second;
  std::list<EventData *>::iterator eventIter = deleteEventList.begin();
  while (eventIter != deleteEventList.end()) {
    if (*eventIter == tev) {
      return 1;
    }
    ++eventIter;
  }
  return 0;
}

void EvLoop::HandleEvent(const struct epoll_event *tEvents, int nevent) {
  int i;
  int found;
  EventData *tev = nullptr;

  for (i = 0; i < nevent; i++) {
    tev = reinterpret_cast<EventData *>(tEvents[i].data.ptr);
    if (tev != nullptr) {
      found = FindDeletedEvent(tev);
      if (found) {
        MS_LOG(WARNING) << "fd has been deleted from epoll]fd:" << tev->fd << ",efd:" << efd;
        continue;
      }

      tev->handler(tev->fd, tEvents[i].events, tev->data);
    }
  }
}

void EvLoop::StopEventLoop() {
  if (stopLoop == 1) {
    return;
  }

  stopLoop = 1;

  uint64_t one = 1;
  if (write(queueEventfd, &one, sizeof(one)) != sizeof(one)) {
    MS_LOG(WARNING) << "fail to write queueEventfd]fd:" << queueEventfd << ",errno:" << errno;
  }
  return;
}

void EvLoop::EventLoopDestroy() {
  /* free deleted event handlers */
  EventFreeDelEvents();
  if (efd > 0) {
    if (queueEventfd > 0) {
      (void)DelFdEvent(queueEventfd);
      close(queueEventfd);
      queueEventfd = -1;
    }

    close(efd);
    efd = -1;
  }
}

}  // namespace mindspore
