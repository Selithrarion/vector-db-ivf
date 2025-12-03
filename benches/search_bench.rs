use criterion::{Criterion, criterion_group, criterion_main};
use rand::Rng;
use seli_vector_db::{VectorDB};
use std::hint::black_box;
// use hnsw::{Hnsw, Searcher};
// use space::Metric;
//
// #[derive(Clone, Copy)]
// struct Euclidean;
// impl<'a> Metric<&'a [f32]> for Euclidean {
//     type Unit = u32;
//
//     fn distance(&self, a: &&'a [f32], b: &&'a [f32]) -> Self::Unit {        let squared_dist: f32 = a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum();
//         let squared_dist: f32 = a.iter()
//             .zip(b.iter())
//             .map(|(x, y)| (x - y).powi(2))
//             .sum();
//         squared_dist.to_bits()
//     }
// }

fn generate_random_vector(size: usize) -> Vec<f32> {
    rand::rng().random_iter().take(size).collect()
}

fn bench_search(c: &mut Criterion) {
    let dim = 128;
    let num_vectors = 10_000;
    let k = 10;

    let vectors: Vec<Vec<f32>> = (0..num_vectors)
        .map(|_| generate_random_vector(dim))
        .collect();
    let query = generate_random_vector(dim);

    let mut db_brute = VectorDB::new();
    for vec in &vectors {
        db_brute.add(vec.clone()).unwrap();
    }
    c.bench_function("brute_force_10k", |b| {
        b.iter(|| db_brute.search(black_box(&query), black_box(k), 1))
    });

    let mut db_ivf = VectorDB::new();
    for vec in &vectors {
        db_ivf.add(vec.clone()).unwrap();
    }
    let num_clusters = (num_vectors as f64).sqrt() as usize;
    let max_iterations = 20;
    db_ivf
        .build_index(num_clusters, max_iterations)
        .expect("Failed to build IVF index for benchmark");
    let nprobe = 5;
    let bench_name = format!("ivf_10k_nprobe_{}", nprobe);
    c.bench_function(&bench_name, |b| {
        b.iter(|| db_ivf.search(black_box(&query), black_box(k), black_box(nprobe)))
    });

    // let mut searcher = Searcher::new();
    // let hnsw = Hnsw::new(Euclidean);
    // let data_for_hnsw: Vec<&[f32]> = vectors.iter().map(|v| v.as_slice()).collect();
    // hnsw.parallel_build(&mut searcher, &data_for_hnsw);
    // c.bench_function("hnsw_10k", |b| {
    //     b.iter(|| {
    //         hnsw.search(&query, k, &searcher);
    //     })
    // });
}

criterion_group!(benches, bench_search);
criterion_main!(benches);
