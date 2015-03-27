#ifndef __ALLOCATOR_H__
#define __ALLOCATOR_H__

#include <stack>
#include <map>
/***********************************************************
 * Class to allocate arrays of memory for temperary use. 
 * The allocator will hold onto the memory for the next call.
 * This allows memory like Vectors to be reused in different
 * parts of the algorithm without having to store it in 
 * a class and hold onto even when it isn't being used.
 ***********************************************************/
template<typename T>
class Allocator {
  typedef std::stack<T*> FreeList;
  typedef std::map<int,FreeList> FreeMap;
    
  public:
    static T* allocate(int size);
    static void free(T *v,int size);
    static void clear();

  private:
    static FreeMap free_vars;  //a map of vector lists
};
#endif
