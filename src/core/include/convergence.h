#ifndef __CONVERGENCE_H__
#define __CONVERGENCE_H__
#include <getvalue.h>

enum ConvergenceType {ABSOLUTE,RELATIVE};
inline const char* getString(ConvergenceType p) {
  switch(p)
  {
    case ABSOLUTE:
      return "ABSOLUTE";
    case RELATIVE:
      return "RELATIVE";
    default:
      return "UNKNOWN";
  }
}

template <>
inline ConvergenceType getValue<ConvergenceType>(const char* name) {
  if(strncmp(name,"ABSOLUTE",100)==0) 
    return ABSOLUTE;
  else if(strncmp(name,"RELATIVE",100)==0) 
    return RELATIVE;
  
  char error[100];
  sprintf(error,"ConvergenceType '%s' is not defined",name);
  FatalError(error);
}
#endif
