#include <amg_config.h>
#include <iostream>
#include <exception>
#include <error.h>
#include <types.h>

using namespace std;

SignalHandler AMG_Config::sh;
AMG_Config::ParamDesc AMG_Config::param_desc;
bool AMG_Config::registered=false;

void AMG_Config::parseParameterString(char* str) {
  //copy to a temperary array to avoid destroying the string
  char params[1000];
  strncpy(params,str,1000);

  //tokenize
  char *var=strtok(params," ,");
  while(var!=0)
  {
    setParameter(var);
    var=strtok(NULL," ");
  }
}

#include<fstream>
void AMG_Config::parseFile(const char* filename) {
  std::ifstream fin;
  fin.open(filename);
  if(!fin)
  {
    char error[500];
    sprintf(error,"Error opening file '%s'",filename);
    FatalError(error);
  }

  while(!fin.eof())
  {
    char line[1000];
    fin.getline(line,1000,'\n');
    parseParameterString(line);
  }
  fin.close();
}

template <typename Type>
void AMG_Config::registerParameter(std::string name, std::string description, Type default_value) {
  param_desc[name]=ParameterDescription(&typeid(Type),name,description,default_value);
}

template <typename Type>
Type AMG_Config::getParameter(std::string name) {
  //verify the paramter has been registered
  ParamDesc::iterator desc_iter=param_desc.find(name);
  if(desc_iter==param_desc.end()) {
    string error = "getParameter error: '" + name + "' not found\n";
    throw invalid_argument(error);
  }
  //verify the types match
  if(desc_iter->second.type!=&typeid(Type))
  {
    string error = "getParameter error: '" + name + "' type mismatch\n";
    throw invalid_argument(error);
  }

  //check if the paramter has been set
  ParamDB::iterator param_iter=params.find(name);
  if(param_iter==params.end()) {
    return desc_iter->second.default_value.get<Type>(); //return the default value
  }
  else {
    return param_iter->second.get<Type>();              //return the parameter value
  }
}

template <typename Type>
void AMG_Config::setParameter(std::string name, Type value) {
  //verify that the parameter has been registered
  ParamDesc::iterator iter=param_desc.find(name);
  if(iter==param_desc.end()) {
    string error = "setParameter error: '" + name + "' not found\n";
    throw invalid_argument(error);
  }
  if(iter->second.type!=&typeid(Type)) {
    string error = "setParameter error: '" + name + "' type mismatch\n";
    throw invalid_argument(error);
  }
  params[name]=value;
}

void AMG_Config::printOptions() {
  for(ParamDesc::iterator iter=param_desc.begin();iter!=param_desc.end();iter++)
  {
    std::cout << "           " << iter->second.name << ": " << iter->second.description << std::endl;
  }
}


#include <amg_level.h>
//#include <norm.h>
#include <convergence.h>
#include <cycles/cycle.h>
#include <smoothers/smoother.h>
#include <smoothedMG/aggregators/aggregator.h>

void AMG_Config::printAMGConfig() {
      std::cout << "AMG Configuration: "  << std::endl;
      std::cout << "  Cuda Parameters: " << std::endl;
      std::cout << "    Device Number: " << getParameter<int>("cuda_device_num") << std::endl;
      std::cout << "    Max Thread/Block: " << getParameter<int>("max_threads_per_block") << std::endl;
      std::cout << "  Setup Parameters: " << std::endl;
      std::cout << "    Algorithm: " << getString(getParameter<AlgorithmType>("algorithm")) << std::endl;
      std::cout << "    Max Levels: " << getParameter<int>("max_levels") << std::endl;
      std::cout << "  Solver Parameters: " << std::endl;
      std::cout << "    Solver: " << getString(getParameter<SolverType>("solver")) << std::endl;
      std::cout << "    Cycle: " << getString(getParameter<CycleType>("cycle")) << std::endl;
      std::cout << "    Smoother: " << getString(getParameter<SmootherType>("smoother")) << std::endl;
      std::cout << "      Presweeps: " << getParameter<int>("presweeps") << std::endl;
      std::cout << "      Postsweeps: " << getParameter<int>("postsweeps") << std::endl;
      std::cout << "    Convergence Type: " << getString(getParameter<ConvergenceType>("convergence")) << std::endl;
      std::cout << "      Tolerance: " << std::scientific << getParameter<double>("tolerance") << std::fixed << std::endl;
      std::cout << "    Max Iterations: " << getParameter<int>("max_iters") << std::endl;
//      std::cout << "    Norm: " << getString(getParameter<NormType>("norm")) << std::endl;
}

void AMG_Config::setParameter(const char* str) {

  std::string tmp(str);

  //locate the split
  int split_loc=tmp.find("=");

  std::string name=tmp.substr(0,split_loc);
  std::string value=tmp.substr(split_loc+1);

  //verify parameter was registered
  ParamDesc::iterator iter=param_desc.find(std::string(name));
  if(iter==param_desc.end()) {
    char error[100];
    sprintf(error,"Variable '%s' not registered",name.c_str());
    FatalError(error);
  }

  if(*(iter->second.type)==typeid(int))
    setParameter(name, getValue<int>(value.c_str()));
  else if(*(iter->second.type)==typeid(float))
    setParameter(name, getValue<float>(value.c_str()));
  else if(*(iter->second.type)==typeid(double))
    setParameter(name, getValue<double>(value.c_str()));
  else if(*(iter->second.type)==typeid(SmootherType))
    setParameter(name, getValue<SmootherType>(value.c_str()));
  else if(*(iter->second.type)==typeid(CycleType))
    setParameter(name, getValue<CycleType>(value.c_str()));
  else if(*(iter->second.type)==typeid(SolverType))
    setParameter(name, getValue<SolverType>(value.c_str()));
//  else if(*(iter->second.type)==typeid(NormType))
//    setParameter(name, getValue<NormType>(value.c_str()));
  else if(*(iter->second.type)==typeid(ConvergenceType))
    setParameter(name, getValue<ConvergenceType>(value.c_str()));
  else if(*(iter->second.type)==typeid(AlgorithmType))
    setParameter(name, getValue<AlgorithmType>(value.c_str()));
  else
  {
    char error[100];
    sprintf(error,"getValue is not implemented for the datatype of variable '%s'",name.c_str());
    FatalError(error);
  }
}

#include <register.h>
AMG_Config::AMG_Config() {
  if(!registered) {
    registerParameters();
    registered=true;
  }
}
