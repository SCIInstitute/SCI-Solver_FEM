/* 
 * File:   Helper.h
 * Author: T. James Lewis
 *
 * Created on July 17, 2013, 12:24 PM
 */

#ifndef HELPER_H
#define	HELPER_H
namespace Helper {
    template <class T>
    int BinarySearch(T value, T* array, int size) {
        int imin = 0;
        int imax = size - 1;
        while (imin < imax) {
            int imid = (imax + imin) / 2;
            if (array[imid] < value)
                imin = imid + 1;
            else
                imax = imid;
        }
        if (imax == imin && array[imin] == value)
            return imin;
        else 
            return -1;
    }
}
#endif	/* HELPER_H */

