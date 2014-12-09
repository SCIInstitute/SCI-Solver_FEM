/* 
 * File:   AggMIS_IOHelpers.cu
 * Author: T. James Lewis
 *
 * Created on May 25, 2013, 4:43 PM
 */
#include "AggMIS_IOHelpers.h"
namespace AggMIS {
    namespace InputHelpers {
        string GetNonEmptyLineCIN() {
            string b;
            char dumb;
            while (cin.peek() == '\n')
                cin.get(dumb);
            getline(cin, b);
            return b;
        }
        int GetSingleIntegerValueCIN() {
            string input;
            char dumb;
            while (true)
            {
                while (cin.peek() == '\n')
                    cin.get(dumb);
                getline(cin, input);
                stringstream str(input);
                int result;
                if (str >> result)
                    return result;
                cout << "Please enter a number\n:";
            }
        }
        vector<int> GetIntegerValuesCIN() {
            string input;
            char dumb;
            int value;
            vector<int> values;
            while (true)
            {
                while (cin.peek() == '\n')
                    cin.get(dumb);
                getline(cin, input);
                stringstream stream(input);
                while(!stream.eof())
                {
                    if (stream >> value)
                        values.push_back(value);
                    else
                    {
                        stream.clear();
                        string dumber;
                        stream >> dumber;
                    }
                }
                if (values.size() > 0)
                    return values;
                cout << "Please enter at least one number\n:";
            }
        }       
    }
}
