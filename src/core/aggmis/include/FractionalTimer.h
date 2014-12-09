/* 
 * File:   FractionalTimer.h
 * Author: T. James Lewis
 *
 * Created on July 15, 2013, 1:07 PM
 */

#ifndef FRACTIONALTIMER_H
#define	FRACTIONALTIMER_H
#include<time.h>
#include<vector>
namespace Timing {
    using namespace std;
    namespace Host {
        class Timer {
        public:
            Timer() {
                started = false;
                stopped = false;
                startTime.tv_sec = startTime.tv_nsec = 0;
                endTime.tv_sec = endTime.tv_nsec = 0;
            }
            ~Timer() {};
            void start() {
                clock_gettime(CLOCK_REALTIME, &startTime);
                started = true;
                stopped = false;
            }
            void stop() {
                clock_gettime(CLOCK_REALTIME, &endTime);
                stopped = true;
            }
            double getElapsedTimeInSec() {
                if (!started || !stopped) {
                    printf("Error: elapsed time requested when not valid.\n");
                    return -1.0;
                }
                // Check if we need to carry some nanoseconds
                if (endTime.tv_nsec < startTime.tv_nsec) {
                    endTime.tv_nsec += 1000000000;
                    endTime.tv_sec -= 1;
                }
                long timeInMicrosec = ((endTimeHost.tv_sec - startTimeHost.tv_sec) * 1000000)
                                + ((endTimeHost.tv_nsec - startTimeHost.tv_nsec) / 1000);
                return (double)(timeInMicrosec) / 1000000.0;
            }
            double getElapsedTimeInMillisec() {
                if (!started || !stopped) {
                    printf("Error: elapsed time requested when not valid.\n");
                    return -1.0;
                }
                // Check if we need to carry some nanoseconds
                if (endTime.tv_nsec < startTime.tv_nsec) {
                    endTime.tv_nsec += 1000000000;
                    endTime.tv_sec -= 1;
                }
                long timeInMicrosec = ((endTimeHost.tv_sec - startTimeHost.tv_sec) * 1000000)
                                + ((endTimeHost.tv_nsec - startTimeHost.tv_nsec) / 1000);
                return (double)(timeInMicrosec) / 1000.0;
            }
        private:
            timespec startTime, endTime;
            bool started, stopped;
        };
    }
    namespace Device {
        class Timer {
        public:
            Timer() {
                cudaEventCreate(&startTimeCuda);
                cudaEventCreate(&endTimeCuda);
                started = false;
                stopped = false;
            }
            ~Timer() {}
            void start() {
                cudaEventRecord(startTime, 0);
                started = true;
                stopped = false;
            }
            void stop() {
                cudaEventRecord(endTime, 0);
                cudaEventSynchronize(endTime);
                cudaEventElapsedTime(&elapsedTime, startTime, endTime);
                stopped = true;
            }
            double getElapsedTimeInSec() {
                if (!started || !stopped) {
                    printf("Error: elapsed time requested when not valid.\n");
                    return -1.0;
                }
                return (double)elapsedTime / 1000;
            }
            double getElapsedTimeInMillisec() {
                if (!started || !stopped) {
                    printf("Error: elapsed time requested when not valid.\n");
                    return -1.0;
                }
                return (double)elapsedTime;
            }
        private:
            cudaEvent_t startTime, endTime;
            bool started, stopped;
            float elapsedTime;
        };
    }
    namespace Fractional {
        template <class genericTimer>
        class Timer {
        public:
            Timer(string title){
                this->title = title;
            }
            ~Timer(){}
        private:
            string title;
            genericTimer overall;
            vector<string> partNames;
            vector<genericTimer> partTimers;
            vector<vector<double> > runTimes;
        };
    }
}
#endif	/* FRACTIONALTIMER_H */

