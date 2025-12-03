$env:RUSTFLAGS='-C target-cpu=native'; cargo bench

1000 vectors, k 10, brute force - 123 µs  
1000 vectors, k 10, brute force + normalize - 95 µs  
1000 vectors, k 10, ivf_1k_nprobe_5 - 30 µs  

---

100.000 vectors, k 10, brute force + normalize - 18 ms  
100.000 vectors, k 10, ivf_1k_nprobe_5 - 225 µs  

---

10.000 vectors, k 10, brute force + normalize - 1.5 ms  
10.000 vectors, k 10, ivf_1k_nprobe_5 - 86 µs  

10.000 vectors, k 10, brute force + normalize - 1.5 ms  
10.000 vectors, k 10, ivf_1k_nprobe_5 + inline always for dot product - 80 µs  

10.000 vectors, k 10, brute force + normalize - 1.5 ms  
10.000 vectors, k 10, ivf_1k_nprobe_5 + vec for index - 83 µs  

---

10.000 vectors, k 10, brute force + simd rustflag - 1.5 ms  
10.000 vectors, k 10, ivf_1k_nprobe_5 + simd rustflag 73 µs  

10.000 vectors, k 10, brute force + wide for simd (dot product optimized) - 405 µs   
10.000 vectors, k 10, ivf_1k_nprobe_5 + wide for simd (dot product optimized) - 20-25 µs  

normalize simd optimization nothing changed, became kinda worse

flat memory layout (Vec<Vec32>> to Vec<32>) didnt help much - brute force 400 µs


