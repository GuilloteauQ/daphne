/*
 * Copyright 2021 The DAPHNE Consortium
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include "Worker.h"
#include <charconv>
#include <fstream>
#include <runtime/local/vectorized/TaskQueues.h>

#include <spdlog/spdlog.h>
#include <chrono>
#include <iostream>
#include <sstream>
#include <string>
#include <unistd.h>

// TODO add this into a cmake file
#define SCHEDULE_VISUALIZATION 1

#if defined(SCHEDULE_VISUALIZATION)
#define MAX_HOSTNAME_SIZE 128
#define LOG_VIZ(x) {\
  auto taskStartTime = std::chrono::high_resolution_clock::now();\
  x \
  auto taskEndTime = std::chrono::high_resolution_clock::now();\
  workerLogFile << t << "," << t->getTaskSize() << "," << _threadID << "," << currentDomain << "," << hostname << "," << targetQueue << "," << taskStartTime.time_since_epoch().count() << "," << taskEndTime.time_since_epoch().count() << "\n";\
}

#ifndef SCHEDULE_VISUALIZATION_PREFIX
#define SCHEDULE_VISUALIZATION_PREFIX "/tmp/worker_domain_"
#endif 

#else
#define LOG_VIZ(x) x
#endif


class WorkerCPU : public Worker {
    std::vector<TaskQueue*> _q;
    std::vector<int> _physical_ids;
    std::vector<int> _unique_threads;
    std::array<bool, 256> eofWorkers;
    bool _verbose;
    uint32_t _fid;
    uint32_t _batchSize;
    int _threadID;
    int _numQueues;
    int _queueMode;
    int _stealLogic;
    bool _pinWorkers;
public:
    // ToDo: remove compile-time verbose parameter and use logger
    WorkerCPU(std::vector<TaskQueue*> deques, std::vector<int> physical_ids, std::vector<int> unique_threads,
            DCTX(dctx), bool verbose, uint32_t fid = 0, uint32_t batchSize = 100, int threadID = 0, int numQueues = 0,
            int queueMode = 0, int stealLogic = 0, bool pinWorkers = 0) : Worker(dctx), _q(deques),
            _physical_ids(physical_ids), _unique_threads(unique_threads),
            _verbose(verbose), _fid(fid), _batchSize(batchSize), _threadID(threadID), _numQueues(numQueues),
            _queueMode(queueMode), _stealLogic(stealLogic), _pinWorkers(pinWorkers) {
        // at last, start the thread
        t = std::make_unique<std::thread>(&WorkerCPU::run, this);
    }

    ~WorkerCPU() override = default;

    void run() override {
        if (_pinWorkers) {
            // pin worker to CPU core
            cpu_set_t cpuset;
            CPU_ZERO(&cpuset);
            CPU_SET(_threadID, &cpuset);
            sched_setaffinity(0, sizeof(cpu_set_t), &cpuset);
        }

        int currentDomain = _physical_ids[_threadID];
        int targetQueue = _threadID;
        if( _queueMode == 0 ) {
            targetQueue = 0;
        } else if ( _queueMode == 1) {
            targetQueue = currentDomain;
        } else if ( _queueMode == 2) {
            targetQueue = _threadID;
        } else {
            ctx->logger->error("WorkerCPU: queue not found");
        }
        int startingQueue = targetQueue;

        Task* t = _q[targetQueue]->dequeueTask();

#ifdef SCHEDULE_VISUALIZATION
        char hostname[MAX_HOSTNAME_SIZE];
        gethostname(hostname, MAX_HOSTNAME_SIZE);
        std::ofstream workerLogFile;
        workerLogFile.open(SCHEDULE_VISUALIZATION_PREFIX + std::to_string(currentDomain) + "_threadid_" + std::to_string(_threadID) + ".csv", std::ios_base::app);
#endif

        while( !isEOF(t) ) {
            //execute self-contained task
            if( _verbose ) {
                ctx->logger->trace("WorkerCPU: executing task.");
            }
            LOG_VIZ(t->execute(_fid, _batchSize);)
            delete t;
            //get next tasks (blocking)
            t = _q[targetQueue]->dequeueTask();
        }

        LOG_VIZ()

        // All tasks from own queue have completed. Now stealing from other queues.

        if( _numQueues > 1 ) {
            if( _stealLogic == 0) {
                // Stealing in sequential order

                targetQueue = (targetQueue+1)%_numQueues;

                while ( targetQueue != startingQueue ) {
                    t = _q[targetQueue]->dequeueTask();
                    if( isEOF(t) ) {
                        targetQueue = (targetQueue+1)%_numQueues;
                    } else {
                        LOG_VIZ(t->execute(_fid, _batchSize);)
                        delete t;
                    }
                }
            } else if ( _stealLogic == 1) {
                // Stealing in sequential order from same domain first
                if ( _queueMode == 2 ) {
                    targetQueue = (targetQueue+1)%_numQueues;

                    while ( targetQueue != startingQueue ) {
                        if ( _physical_ids[targetQueue] == currentDomain ){
                            t = _q[targetQueue]->dequeueTask();
                            if( isEOF(t) ) {
                                targetQueue = (targetQueue+1)%_numQueues;
                            } else {
                                LOG_VIZ(t->execute(_fid, _batchSize);)
                                delete t;
                            }
                        } else {
                            targetQueue = (targetQueue+1)%_numQueues;
                        }
                    }
                }

                // No more tasks on this domain, now switching to other domain

                targetQueue = (targetQueue+1)%_numQueues;

                while ( targetQueue != startingQueue ) {
                    t = _q[targetQueue]->dequeueTask();
                    if( isEOF(t) ) {
                        targetQueue = (targetQueue+1)%_numQueues;
                    } else {
                        LOG_VIZ(t->execute(_fid, _batchSize);)
                        delete t;
                    }
                }
            } else if( _stealLogic == 2) {
                // stealing from random workers until all workers EOF

                eofWorkers.fill(false);
                while( std::accumulate(eofWorkers.begin(), eofWorkers.end(), 0) < _numQueues ) {
                    targetQueue = rand() % _numQueues;
                    if( eofWorkers[targetQueue] == false ) {
                        t = _q[targetQueue]->dequeueTask();
                        //std::cout << "Execute task stolen from: " << targetQueue << std::endl;
                        if( isEOF(t) ) {
                            eofWorkers[targetQueue] = true;
                        } else {
                            LOG_VIZ(t->execute(_fid, _batchSize);)
                            delete t;
                        }
                    }
                }

            } else if ( _stealLogic == 3) {
                // stealing from random workers from same socket first
                int queuesThisDomain = 0;
                eofWorkers.fill(false);

                for( int i=0; i<_numQueues; i++ ) {
                    if( _physical_ids[i] == currentDomain ) {
                        queuesThisDomain++;
                    }
                }
                if ( _queueMode == 2 ) {
                    while( std::accumulate(eofWorkers.begin(), eofWorkers.end(), 0) < queuesThisDomain ) {
                        targetQueue = rand() % _numQueues;
                        if( _physical_ids[targetQueue] == currentDomain) {
                            if( eofWorkers[targetQueue] == false ) {
                                t = _q[targetQueue]->dequeueTask();
                                if( isEOF(t) ) {
                                    eofWorkers[targetQueue] = true;
                                } else {
                                    LOG_VIZ(t->execute(_fid, _batchSize);)
                                    delete t;
                                }
                            }
                        }
                    }
                }

                // all workers on same domain are EOF, now also allowing stealing from other domain
                // This could also be done by keeping a list of EOF workers on the other domain

                while ( std::accumulate(eofWorkers.begin(), eofWorkers.end(), 0) < _numQueues ) {
                    targetQueue = rand() % _numQueues;
                    // no need to check if they are on the other domain, because otherwise they would be EOF anyway
                    if( eofWorkers[targetQueue] == false ) {
                        t = _q[targetQueue]->dequeueTask();
                        if( isEOF(t) ) {
                            eofWorkers[targetQueue] = true;
                        } else {
                            LOG_VIZ(t->execute(_fid, _batchSize);)
                            delete t;
                        }
                    }
                }
            }
        }

#ifdef SCHEDULE_VISUALIZATION
        workerLogFile.close();
#endif

        // No more tasks available anywhere
        if( _verbose )
            ctx->logger->debug("WorkerCPU: received EOF, finalized.");
    }
};
