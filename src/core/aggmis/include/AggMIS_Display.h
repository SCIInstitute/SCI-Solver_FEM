/* 
 * File:   AggMIS_Display.h
 * Author: T. James Lewis
 *
 * Created on April 22, 2013, 12:13 PM
 */

#ifndef AGGMIS_DISPLAY_H
#define	AGGMIS_DISPLAY_H
#include "AggMIS_Types.h"
namespace AggMIS {
    namespace Display {
        using namespace Types;
        using namespace std;
        
        // Show vectors
        void PrintVector(IntVector_d& toPrint, string comment);
    }
}
#endif	/* AGGMIS_DISPLAY_H */

