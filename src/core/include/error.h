#ifndef __MYERROR_H__
#define __MYERROR_H__
#ifndef WIN32
#include <execinfo.h>
#include <dlfcn.h>
#include <cxxabi.h>
#include <unistd.h>
#endif
#include <stdio.h>

/******************************************************
 * prints the current stack trace
 *****************************************************/
inline void printStackTrace() {
#ifndef WIN32
  const int MAX_STACK=30;
  size_t n;
  static void *addresses[MAX_STACK];
  n=backtrace(addresses,MAX_STACK);

  if(n<2)
    return;

  char **names=backtrace_symbols( addresses, n );

  printf("Backtrace for pid %d:\n",getpid());
  
  for(int i=1;i<n;i++)
  {
    Dl_info info;
    char *demangled=NULL;
    //attempt to demangle the symbol
    if(dladdr(addresses[i],&info) != 0)
    {
      if(info.dli_sname!=0) {
        int stat;
        demangled = abi::__cxa_demangle(info.dli_sname,0,0,&stat);
        printf("    %d: %p - %s\n",i-1,addresses[i],demangled);
      }
      else //couldn't locate the symbol so just print the mangled name
        printf("    %d: %p - %s\n",i-1,addresses[i],names[i]); 
    }
    else
      printf("    %d: %p\n",i-1,(char*)addresses[i]);
  }
#endif
}

/********************************************************
 * Prints the error message, the stack trace, and exits
 * ******************************************************/
#define FatalError(s)                                               \
  printf("Fatal error '%s' at %s:%d\n",s,__FILE__,__LINE__);        \
  printStackTrace();                                                \
  exit(1);
#endif
