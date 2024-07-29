
main: main.cu
	nvcc -O3 -w -m64 -Xptxas -dlcm=cg -gencode=arch=compute_86,code=sm_86 -gencode=arch=compute_86,code=compute_86 main.cu -o YYSpTRSV -lcusparse -lcudart
matsolve-yysptrsv: matsolve-yysptrsv.cu
	nvcc -O3 -w -m64 --std=c++17 -Xptxas -dlcm=cg -gencode=arch=compute_86,code=sm_86 -gencode=arch=compute_86,code=compute_86 matsolve-yysptrsv.cu -o matsolve-yysptrsv -lcusparse -lcudart -g
clean:
	rm -f YYSpTRSV
