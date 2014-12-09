/* 
 * File:   DataRecord.h
 * Author: T. James Lewis
 *
 * Created on July 28, 2013, 5:43 PM
 */
#ifndef DATARECORD_H
#define DATARECORD_H
#include <map>
#include <vector>
#include <string>
#include <sstream>
#include <iostream>
#include <fstream>
namespace Data {
    class Record {
    public:
        /**
         * Default constructor, sets the writeOnClose so the destructor won't
         * write out anything.
         */
        Record();
        /**
         * Class destructor. Writes out record to file if a file has been set.
         */
        ~Record();
        /**
         * Sets the writeOnClose file to the given filename. When set the class
         * destructor will write out the record to the file.
         * @param fileName The name of the file to write out to on closing.
         */
        void SetWriteOnClose(std::string fileName);
        /**
         * Checks if there is an integer data field in the record with the 
         * given name.
         * @param name Name to look for.
         * @return True if found
         */
        bool ContainsIntField(std::string name);
        /**
         * Checks if there is a double data field in the record with the 
         * given name.
         * @param name Name to look for.
         * @return True if found.
         */
        bool ContainsDoubleField(std::string name);
        /**
         * Checks if there is a string data field in the record with the 
         * given name.
         * @param name Name to look for.
         * @return True if found.
         */
        bool ContainsStringField(std::string name);
        /**
         * Looks for the int field with the given name and returns a reference
         * to the data vector. If the name is not present it is added.
         * @param name The field name to retrieve.
         * @return A reference to the vector of data elements for the field.
         */
        std::vector<int>& GetIntField(std::string name);
        /**
         * Looks for the double field with the given name and returns a reference
         * to the data vector. If the name is not present it is added.
         * @param name The field name to retrieve.
         * @return A reference to the vector of data elements for the field.
         */
        std::vector<double>& GetDoubleField(std::string name);
        /**
         * Looks for the string field with the given name and returns a reference
         * to the data vector. If the name is not present it is added.
         * @param name The field name to retrieve.
         * @return A reference to the vector of data elements for the field.
         */
        std::vector<std::string>& GetStringField(std::string name);
        /**
         * Writes the record out to the specified file (appended).
         * @param fileName File to write to.
         */
        void WriteToFile(std::string fileName);
        /**
         * Record the given value in the specified field. If the field is not 
         * present in the record it is created. If there are already values the
         * current value is appended to them.
         * @param name The field name to record.
         * @param value The value to be recorded.
         */
        void RecordValue(std::string name, int value);
        /**
         * Record the given value in the specified field. If the field is not 
         * present in the record it is created. If there are already values the
         * current value is appended to them.
         * @param name The field name to record.
         * @param value The value to be recorded.
         */
        void RecordValue(std::string name, double value);
        /**
         * Record the given value in the specified field. If the field is not 
         * present in the record it is created. If there are already values the
         * current value is appended to them.
         * @param name The field name to record.
         * @param value The value to be recorded.
         */
        void RecordValue(std::string name, std::string value);
    private:
        std::string writeOnClose;
        std::vector<std::vector<std::string> > stringValues;
        std::vector<std::vector<int> > intValues;
        std::vector<std::vector<double> >doubleValues;
        std::vector<std::string> stringNames;
        std::vector<std::string> intNames;
        std::vector<std::string> doubleNames;
    };
    /**
     * This method opens the specified file and reads in all records from it
     * into the return vector.
     * @param fileName The file to read in.
     * @return A pointer to a vector of Records read from the file.
     */
    std::vector<Record>* ReadRecords(std::string fileName);
}

#endif /* DATARECORDER_H */