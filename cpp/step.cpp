#include <time.h>
#include <stdlib.h>
#include <utility>
using namespace std;

#define N_ROBOTS 1000
#define ROWS 100
#define COLS 400

int actions[N_ROBOTS];
int poss[N_ROBOTS][2];
int goals[N_ROBOTS][2];
int field[ROWS][COLS];
int rewards[N_ROBOTS];

void step(){
    for(int i = 0; i < N_ROBOTS; i++){
        int action = actions[i];
        int currpos[2] = {poss[i][0], poss[i][0]};
        int goal[2];
    }
}

int main(){

}