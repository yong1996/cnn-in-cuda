
#include <iostream>
#include <fstream>
#include <string>
#include <vector>

#include "mnist.h"


using namespace std;


static inline void loadData(){
    mnist_load("MNIST_data/train-images.idx3-ubyte", "data/train-labels.idx1-ubyte",
		&train_set, &train_cnt);
	mnist_load("MNIST_data/t10k-images.idx3-ubyte", "data/t10k-labels.idx1-ubyte",
		&test_set, &test_cnt);
}



int main(int argc, char** argv){

    return 0;

}


double forward(double &data[28][28]){
    float input[28][28];

    for (int i = 0; i<28; i<++){
        for (int j = 0; j<28; j++){
            input[i][j] = data[i][j];
        }
    }
}