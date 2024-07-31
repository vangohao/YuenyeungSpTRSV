CUDAARCHS?=80

main: main.cu
	nvcc -O3 -w -m64 -Xptxas -dlcm=cg -gencode=arch=compute_$(CUDAARCHS),code=sm_$(CUDAARCHS) -gencode=arch=compute_$(CUDAARCHS),code=compute_$(CUDAARCHS) main.cu -o YYSpTRSV -lcusparse -lcudart
matsolve-yysptrsv: matsolve-yysptrsv.cu
	nvcc -DUNI_CUDA_ARCH=$(CUDAARCHS) -O3 -w -m64 --std=c++17 -Xptxas -dlcm=cg -gencode=arch=compute_$(CUDAARCHS),code=sm_$(CUDAARCHS) -gencode=arch=compute_$(CUDAARCHS),code=compute_$(CUDAARCHS) matsolve-yysptrsv.cu -o matsolve-yysptrsv -lcusparse -lcudart -g
matsolve-yysptrsv-level: matsolve-yysptrsv-level.cu
	nvcc -DUNI_CUDA_ARCH=$(CUDAARCHS) -O3 -w -m64 --std=c++17 -Xptxas -dlcm=cg -gencode=arch=compute_$(CUDAARCHS),code=sm_$(CUDAARCHS) -gencode=arch=compute_$(CUDAARCHS),code=compute_$(CUDAARCHS) matsolve-yysptrsv-level.cu -o matsolve-yysptrsv-level -lcusparse -lcudart -g
clean:
	rm -f YYSpTRSV
