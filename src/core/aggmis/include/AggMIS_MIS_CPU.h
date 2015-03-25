/* 
 * File:   AggMIS_MIS_CPU.h
 * Author: T. James Lewis
 *
 * Created on June 25, 2013, 6:13 PM
 */

#ifndef AGGMIS_MIS_CPU_H
#define	AGGMIS_MIS_CPU_H
#include <AggMIS_Types.h>
#include <queue>
namespace AggMIS {
    namespace MIS {
        using namespace Types;
        using namespace std;
        IntVector_h* FloodFillMIS(int k, Graph_h &graph);
        IntVector_h* NaiveMIS(int k, Graph_h &graph);
    }
}
#endif	/* AGGMIS_MIS_CPU_H */

