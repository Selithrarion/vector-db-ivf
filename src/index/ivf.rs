use crate::{Vector, VectorDBError, dot_product, normalize};
use bincode::{Decode, Encode};
use rand::prelude::IndexedRandom;

#[derive(Encode, Decode)]
pub struct IVFIndex {
    centroids: Vec<Vector>,
    inverted_file: Vec<Vec<usize>>,
}

impl IVFIndex {
    pub fn train(
        data: &[Vector],
        num_clusters: usize,
        max_iterations: usize,
    ) -> Result<Self, VectorDBError> {
        if data.is_empty() {
            return Err(VectorDBError::EmptyDataSet);
        }
        if num_clusters == 0 || num_clusters > data.len() {
            return Err(VectorDBError::InvalidClusterCount);
        }

        #[cfg(debug_assertions)]
        println!("Training IVF with {} clusters...", num_clusters);

        let mut centroids: Vec<Vector> = data
            .choose_multiple(&mut rand::rng(), num_clusters)
            .cloned()
            .collect();
        let mut cluster_assignments = vec![0; data.len()];

        for i in 0..max_iterations {
            let mut changed = false;
            for (point_idx, point) in data.iter().enumerate() {
                let (closest_centroid_idx, _) = centroids
                    .iter()
                    .enumerate()
                    .map(|(ci, centroid)| (ci, dot_product(point, centroid)))
                    .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
                    .ok_or_else(|| {
                        VectorDBError::InternalError(
                            "Could not determine closest centroid".to_string(),
                        )
                    })?;

                if cluster_assignments[point_idx] != closest_centroid_idx {
                    cluster_assignments[point_idx] = closest_centroid_idx;
                    changed = true;
                }
            }

            let mut new_centroids = vec![vec![0.0; data[0].len()]; num_clusters];
            let mut counts = vec![0; num_clusters];
            for (point_idx, &cluster_idx) in cluster_assignments.iter().enumerate() {
                let point = &data[point_idx];
                for (dim, val) in point.iter().enumerate() {
                    new_centroids[cluster_idx][dim] += val;
                }
                counts[cluster_idx] += 1;
            }

            for i in 0..num_clusters {
                if counts[i] > 0 {
                    for val in &mut new_centroids[i] {
                        *val /= counts[i] as f32
                    }
                    normalize(&mut new_centroids[i]);
                    centroids[i] = new_centroids[i].clone();
                }
            }

            #[cfg(debug_assertions)]
            println!("Iteration {}: assignments changed: {}", i + 1, changed);
            if !changed {
                #[cfg(debug_assertions)]
                println!("Converged after {} iterations", i + 1);
                break;
            }
        }

        let mut inverted_file = vec![Vec::new(); num_clusters];
        for (point_idx, &cluster_idx) in cluster_assignments.iter().enumerate() {
            inverted_file[cluster_idx].push(point_idx);
        }

        #[cfg(debug_assertions)]
        println!("Training complete");
        Ok(Self {
            centroids,
            inverted_file,
        })
    }

    pub fn query<'a>(
        &'a self,
        query: &'a Vector,
        nprobe: usize,
    ) -> impl Iterator<Item = usize> + 'a {
        let mut centroid_scores: Vec<(usize, f32)> = self
            .centroids
            .iter()
            .enumerate()
            .map(|(id, centroid)| (id, dot_product(query, centroid)))
            .collect();
        centroid_scores.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        centroid_scores
            .into_iter()
            .take(nprobe)
            .filter_map(move |(cluster_id, _)| self.inverted_file.get(cluster_id))
            .flatten()
            .copied()
    }

    pub fn num_clusters(&self) -> usize {
        self.centroids.len()
    }
}
