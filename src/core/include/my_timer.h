#ifndef __MY_TIMER_H__
#define __MY_TIMER_H__
#ifdef __MACH__
#include <mach/mach_time.h>
#define CLOCK_REALTIME 0
#define CLOCK_MONOTONIC 0
int inline clock_gettime(int clk_id, struct timespec *t){
  mach_timebase_info_data_t timebase;
  mach_timebase_info(&timebase);
  uint64_t time;
  time = mach_absolute_time();
  double nseconds = ((double)time * (double)timebase.numer)/((double)timebase.denom);
  double seconds = ((double)time * (double)timebase.numer)/((double)timebase.denom * 1e9);
  t->tv_sec = seconds;
  t->tv_nsec = nseconds;
  return seconds+nseconds*1e-9;;
}
#else
#include <time.h>
#endif


/**********************************************
 * A simple high resolution timer
 *********************************************/
double inline CLOCK() {
#ifdef WIN32
#include <windows.h>
#include <stdio.h>
	SYSTEMTIME st;
	GetSystemTime(&st);
	return ((st.wDay * 24. + st.wHour) * 60. + st.wMinute) * 60. + st.wSecond + st.wMilliseconds / 1000.;
#else
  timespec ts;
  clock_gettime(CLOCK_REALTIME,&ts);
  return ts.tv_sec+ts.tv_nsec*1e-9;
#endif
}


/**********************************************
 *  class for holding profiling data if desired
 *********************************************/

#include <vector>
#include <map>
#include <iostream>
#include <iomanip>

typedef std::map<const char *, double> Event;
typedef std::map<const char *, double>::iterator Eiter;

class levelProfile {
  private:
#ifdef PROFILE
    Event Times;
    Event Tic;
#endif

  public:
    levelProfile() { }
    ~levelProfile() {}

    inline void tic(const char *event)
    {
#ifdef PROFILE
//      cudaThreadSynchronize();
      Tic[event] = CLOCK();
#endif
    }

    inline void toc(const char *event) {
#ifdef PROFILE
//      cudaThreadSynchronize();
      double t = CLOCK();
      Times[event] += t-Tic[event];
#endif
    }

#ifdef PROFILE
    std::vector<const char *>
#else
      void
#endif
      inline getHeaders()
      {
#ifdef PROFILE
        std::vector<const char *> headerVec;
        for (Eiter it=Times.begin(); it!=Times.end(); ++it) {
          headerVec.push_back(it->first);
        }
        return headerVec;
#endif
      }

#ifdef PROFILE
    std::vector<double>
#else
      void
#endif
      inline getTimes()
      {
#ifdef PROFILE
        std::vector<double> times;
        for (Eiter it=Times.begin(); it!=Times.end(); ++it) {
          times.push_back(it->second);
        }
        return times;
#endif
      }

    /********************************************
     * Reset all events
     *******************************************/
    inline void resetTimer() {
#ifdef PROFILE
      for (Eiter it=Times.begin(); it!=Times.end(); ++it) {
        it->second = 0.0;
      }
#endif
    }
};
#endif
