/* 
 * File:   AggMIS_IOHelpers.h
 * Author: T. James Lewis
 *
 * Created on May 25, 2013, 4:13 PM
 */

#ifndef AGGMIS_IOHELPERS_H
#define	AGGMIS_IOHELPERS_H

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>

namespace AggMIS {
    namespace InputHelpers {        
        std::string GetNonEmptyLineCIN();
        int GetSingleIntegerValueCIN();
        std::vector<int> GetIntegerValuesCIN();
    }
}

#endif	/* AGGMIS_IOHELPERS_H */

