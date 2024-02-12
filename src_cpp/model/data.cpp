#include "model.h"

mcm data_processing(string file){
    // Open file
    ifstream myfile(file);

    // Check if file exists
    if (myfile.fail()){
        cout << "Not able to open the file." << endl;
    }

    // Read out first line
    string line;
    getline(myfile, line);

    // Number of variables
    int n = line.length() - 1;
    line = line.substr(0, n);

    // Store dataset as vector of strings
    vector<vector<int>> data;

    vector<int> observation;
    for (char element : line){
        observation.push_back(strtol(&element, NULL, 10));
    }

    data.push_back(observation);

    while (getline(myfile, line)) {
        line = line.substr(0, n);
        vector<int> observation;
        for (char element : line){
            observation.push_back(strtol(&element, NULL, 10));
        }
        data.push_back(observation);
    }

    mcm model;
    model.data = data;
    model.n = n;
    model.N = data.size();

    return model;
}