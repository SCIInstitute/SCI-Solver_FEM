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
      Types::IntVector_h* FloodFillMIS(int k, Types::Graph_h &graph);
        Types::IntVector_h* NaiveMIS(int k, Types::Graph_h &graph);
    }
}
#endif	/* AGGMIS_MIS_CPU_H */

