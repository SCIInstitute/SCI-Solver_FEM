#include <allocator.h>

#include<types.h>

#include <stack>
#include <map>

template<typename T>
Allocator<T>::FreeMap Allocator<T>::free_vars; 

template<typename T> 
inline T* allocate(int size) {
  return new T(size);
};

template<>
inline int* allocate<int>(int size) {
  return new int[size];
}

template<typename T>
T* Allocator<T>::allocate(int size) {
  //locate free var list for the right size
  FreeList &f_vars=free_vars[size];

  T *v;
  if(f_vars.empty()) //if there are no free vectors
  {
    //create a new vector
    v=::allocate<T>(size);
  }
  else {
    //set the return value to the previously freed vector
    v=f_vars.top();
    //remove the vector from the free vector list
    f_vars.pop();
  }
  return v;
}

template<typename T>
void Allocator<T>::free(T* v,int size) {
  //add the vector to the free vector list
  free_vars[size].push(v);
}

template<typename T>
void Allocator<T>::clear() {
  for(typename FreeMap::iterator m_iter=free_vars.begin();m_iter!=free_vars.end();m_iter++)
  {
    FreeList &stack=m_iter->second;
    while(!stack.empty()) {
      delete stack.top();
      stack.pop();
    }
  }
  free_vars.clear();
}

/****************************************
 * Explict instantiations
 ***************************************/
template class Allocator<Vector_h>;
template class Allocator<Vector_d>;
