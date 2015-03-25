/* 
 * File:   Logger.h
 * Author: T. James Lewis
 *
 * Created on July 25, 2013, 12:44 PM
 */

#ifndef LOGGER_H
#define	LOGGER_H

#include <string>
#include <iostream>
#include <fstream>
inline void Log(std::string fileName, std::string output) {
    std::ofstream outputFile;
    outputFile.open(fileName.c_str(), std::ofstream::app);
    outputFile << output << "\n";
    outputFile.close();
}

#endif	/* LOGGER_H */

