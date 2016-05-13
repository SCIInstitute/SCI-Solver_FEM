/* 
 * File:   AggMIS_IOHelpers.cu
 * Author: T. James Lewis
 *
 * Created on May 25, 2013, 4:43 PM
 */
#include "AggMIS_IOHelpers.h"
namespace AggMIS {
    namespace InputHelpers {
        std::string GetNonEmptyLineCIN() {
            std::string b;
            char dumb;
            while (std::cin.peek() == '\n')
              std::cin.get(dumb);
            std::getline(std::cin, b);
            return b;
        }
        int GetSingleIntegerValuecin() {
            std::string input;
            char dumb;
            while (true)
            {
              while (std::cin.peek() == '\n')
                    std::cin.get(dumb);
                std::getline(std::cin, input);
                std::stringstream str(input);
                int result;
                if (str >> result)
                    return result;
                std::cout << "Please enter a number\n:";
            }
        }
        std::vector<int> GetIntegerValuescin() {
            std::string input;
            char dumb;
            int value;
            std::vector<int> values;
            while (true)
            {
                while (std::cin.peek() == '\n')
                    std::cin.get(dumb);
                std::getline(std::cin, input);
                std::stringstream stream(input);
                while(!stream.eof())
                {
                    if (stream >> value)
                        values.push_back(value);
                    else
                    {
                        stream.clear();
                        std::string dumber;
                        stream >> dumber;
                    }
                }
                if (values.size() > 0)
                    return values;
                std::cout << "Please enter at least one number\n:";
            }
        }       
    }
}
