#ifndef __AMG_CONFIG_H__
#define __AMG_CONFIG_H__

#include <map>
#include <string>
#include <string.h>  //strtok
#include <typeinfo>

//class SignalHandler;

#include<amg_signal.h>
/******************************************
 * A class for storing a typeless parameter
 *****************************************/
class Parameter {
  public:
    Parameter() { memset(data,0,8); }
    template<typename T> Parameter(T value) {
      set(value);
    }
    template<typename T> T get() {
      T value=*reinterpret_cast<T*>(&data[0]);
      return value;

    } //return the parameter as the templated type
    template<typename T> void set(T value) { *reinterpret_cast<T*>(&data[0])=value; }  //set the parameter from the templated type

  private:
    char data[8]; //8 bytes of storage
};

/*******************************************
 * A class to store a description of a
 * parameter
 ******************************************/
class ParameterDescription {
  public:
    ParameterDescription() : type(0) {}
    ParameterDescription(const ParameterDescription &p) : type(p.type), name(p.name), description(p.description), default_value(p.default_value) {}
    ParameterDescription(const std::type_info *type, const std::string &name, const std::string &description, const Parameter& default_value) : type(type), name(name), description(description), default_value(default_value) {}
    const std::type_info *type;   //the type of the parameter
    std::string name;             //the name of the parameter
    std::string description;      //description of the parameter
    Parameter default_value;      //the default value of the parameter
};

/***********************************************
 * A class for storing paramaters in a database
 * which includes type information.
 **********************************************/
class AMG_Config {
  public:
    AMG_Config();
    /***********************************************
     * Registers the parameter in the database.
    **********************************************/
    template <typename Type> static void registerParameter(std::string name, std::string description, Type default_value);

    /********************************************
    * Gets a parameter from the database and
    * throws an exception if it does not exist.
    *********************************************/
    template <typename Type> Type getParameter(std::string name);

    /**********************************************
    * Sets a parameter in the database
    * throws an exception if it does not exist.
    *********************************************/
    template <typename Type> void setParameter(std::string name, Type value);

    /****************************************************
    * Parse paramters in the format
    * name=value name=value ... name=value
    * and store the variables in the parameter database
    ****************************************************/
    void parseParameterString(char* str);

    /****************************************************
    * Parse a config file  in the format
    * name=value
    * name=value
    * ...
    * name=value
    * and store the variables in the parameter database
    ****************************************************/
    void parseFile(const char* filename);

    /****************************************************
     * Print the options for AMG
     ***************************************************/
    static void printOptions();

    /***************************************************
     * Prints the AMG parameters                       *
     ***************************************************/
    void printAMGConfig();

  private:
    typedef std::map<std::string,ParameterDescription> ParamDesc;
    typedef std::map<std::string,Parameter> ParamDB;

    static ParamDesc param_desc;  //The parameter descriptions
    ParamDB params;               //The parameter database

    /****************************************************
    * Parse a string in the format
    * name=value
    * and store the variable in the parameter database
    ****************************************************/
    void setParameter(const char* str);

    static SignalHandler sh;  //install the signal handlers here
    static bool registered;
};
#endif
