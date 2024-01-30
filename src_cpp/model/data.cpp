#include "model.h"

mcm data_processing(string file){
    cout << "Data processing" << endl;

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

    // Store dataset as vector of strings
    vector<string> data;
    //data.push_back(line.substr(0, n));
    data.push_back(line);

    while (getline(myfile, line)) {
        //data.push_back(line.substr(0, n));
        data.push_back(line);
    }

    mcm model;
    model.data = data;
    model.n = n;
    model.N = data.size();

    return model;
}