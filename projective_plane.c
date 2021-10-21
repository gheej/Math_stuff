#include <stdio.h>
#include <stdlib.h>

int gcd(int a, int b)
{
    if (b == 0)
        return a;
    else
        return gcd(b, a % b);
}

int main(void)
{   
    int n = 40;
    int * ar;
    ar = (int *)malloc(18 * n * n * n * n * n * n * sizeof(int));
    int mod_z = 0;
    for (int i = 0; i < 18 * n * n * n * n * n * n; i++) ar[i] = 0;


    for (int x = 0; x < n; x++)
        for (int y = 0; y < n; y++)
            if (gcd(x, y) == 1)
                {
                    //z = x + y*\omega
                    mod_z = x * x + x * y + y * y;
                    for (int a = 1; a < n; a++)
                        for (int b = 1; b <= a; b++)
                            for (int c = 1; c <= b; c++)
                                for (int d = 1; d <= c; d++)
                                    ar[mod_z * (a * b + a * c + a * d + b * c + b * d + c * d)] += 1;
                }
    //Вывод 100 первых значений 
    for (int i = 0; i < 200; i ++) printf("%d, %d, \n", i, ar[i]);
    free(ar);
return 0;
}             
