#include "model.h"

vector<vector<int>> data_processing(string file, int n){
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
    line = line.substr(0, n);

    // Store dataset as vector of strings
    vector<vector<int>> data;

    vector<int> observation;
    for (int i = 0; i < n; i++) {
        observation.push_back(line[i] - '0');
    }

    data.push_back(observation);

    while (getline(myfile, line)) {
        line = line.substr(0, n);
        vector<int> observation;
        for (int i = 0; i < n; i++) {
            observation.push_back(line[i] - '0');
        }
        data.push_back(observation);
    }

    return data;
}