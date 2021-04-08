
#include <iostream>
#include <fstream>
#include <string>
#include <vector>

#include "mnist.h"


using namespace std;


static void loadData(){
    mnist_load("MNIST_data/train-images.idx3-ubyte", "data/train-labels.idx1-ubyte",
		&train_set, &train_cnt);
	mnist_load("MNIST_data/t10k-images.idx3-ubyte", "data/t10k-labels.idx1-ubyte",
		&test_set, &test_cnt);
}

int main(){

    return 0;

}