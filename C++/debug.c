#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
int s[10];
int random(){
    int i;
    for ( i = 0; i < 10; i++)
    {
        s[i] = 2 * (rand() % 2) - 1;
    }
    return 1;
}
int main(){
    srand((unsigned)time(NULL));
    random();
    int i;
    for ( i = 0; i < 10; i++)
    {
        printf("%d,", s[i]);
    }
    return 0;
}