cd ../mean3x3/rv ; make program ; cd .. ; gcc -fPIC -shared smp.c -o mpriscv_mean3x3.so ; cd ../core
cd ../mean5x5/rv ; make program ; cd .. ; gcc -fPIC -shared smp.c -o mpriscv_mean5x5.so ; cd ../core
cd ../abs/rv ; make program ; cd .. ; gcc -fPIC -shared smp.c -o mpriscv_abs.so ; cd ../core
cd ../sobel/rv ; make program ; cd .. ; gcc -fPIC -shared smp.c -o mpriscv_sobel.so ; cd ../core

