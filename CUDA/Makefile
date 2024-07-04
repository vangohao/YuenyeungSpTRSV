
main: main.cu
	nvcc -O3 -w -m64 -Xptxas -dlcm=cg -gencode=arch=compute_86,code=sm_86 -gencode=arch=compute_86,code=compute_86 main.cu -o YYSpTRSV -lcusparse -lcudart
clean:
	rm -f YYSpTRSV
