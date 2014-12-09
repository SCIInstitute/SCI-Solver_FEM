/* 
 * File:   DataRecorder.h
 * Author: T. James Lewis
 *
 * Created on July 28, 2013, 5:43 PM
 */
#ifndef DATARECORDER_H
#define DATARECORDER_H
#include <string>
namespace DataRecorder {
    // Methods to add values to global record. The record is instantiated
    // in the definition. This header can be swapped out with blank inlines to
    // turn off data recording.
    void Add(std::string name, int value);
    void Add(std::string name, double value);
    void Add(std::string name, std::string value);
    void SetFile(std::string fileName);
}
#endif /* DATARECORDER_H */