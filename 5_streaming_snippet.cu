cudaStream_t s_1, s_2;
cudaStreamCreate(&s_1);
cudaStreamCreate(&s_2);

cudaMemcpyAsync(host_ptr, device_ptr, size, cudaMemcpyDeviceToHost, s_2);
cudaMemcpyAsync(host_ptr, device_ptr, size, cudaMemcpyDeviceToHost, s_1);

device_kernel_add<<<grid, block, 0, s_1>>>();
device_kernel_add<<<grid, block, 0, s_2>>>();
device_kernel_mul<<<grid, block, 0, s_1>>>();
device_kernel_mul<<<grid, block, 0, s_2>>>();

cudaMemcpyAsync(device_ptr, host_ptr, size, cudaMemcpyHostToDevice, s_2);
cudaMemcpyAsync(device_ptr, host_ptr, size, cudaMemcpyHostToDevice, s_1);

cudaStreamSynchronize(s_1);
cudaStreamSynchronize(s_2);

cudaStreamDestroy(s_1);
cudaStreamDestroy(s_2);
