pub mod index;

use bincode::{Decode, Encode};
use index::ivf::IVFIndex;
use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::path::Path;
use thiserror::Error;

pub type Vector = Vec<f32>;

#[derive(Debug, Clone, Encode, Decode)]
pub struct SearchResult {
    pub id: usize,
    pub score: f32,
}

#[derive(Debug, Error)]
pub enum VectorDBError {
    #[error("Vector dimensions do not match")]
    DimensionMismatch,
    #[error("Cannot build index on an empty data set")]
    EmptyDataSet,
    #[error("Number of clusters must be greater than 0")]
    InvalidClusterCount,
    #[error("Internal error: {0}")]
    InternalError(String),
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Bincode decoding error: {0}")]
    BincodeDecode(#[from] bincode::error::DecodeError),
    #[error("Bincode encoding error: {0}")]
    BincodeEncode(#[from] Box<bincode::error::EncodeError>),
}

fn normalize(vector: &mut Vector) {
    let norm = vector.iter().map(|&val| val * val).sum::<f32>().sqrt();
    if norm > 0.0 {
        for val in vector.iter_mut() {
            *val /= norm;
        }
    }
}

#[inline(always)]
fn dot_product(a: &Vector, b: &Vector) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

#[derive(Encode, Decode)]
pub struct VectorDB {
    vectors: Vec<Vector>,
    index: Option<IVFIndex>,
}

impl Default for VectorDB {
    fn default() -> Self {
        Self::new()
    }
}
impl VectorDB {
    pub fn new() -> Self {
        Self {
            vectors: Vec::new(),
            index: None,
        }
    }

    pub fn build_index(
        &mut self,
        num_clusters: usize,
        max_iterations: usize,
    ) -> Result<(), VectorDBError> {
        let index = IVFIndex::train(&self.vectors, num_clusters, max_iterations)?;
        self.index = Some(index);
        Ok(())
    }

    pub fn add(&mut self, mut vector: Vector) -> usize {
        if self.index.is_some() {
            self.index = None;
            println!("Index invalidated by adding new data. Need to rebuild");
        }

        normalize(&mut vector);
        let id = self.vectors.len();
        self.vectors.push(vector);
        id
    }

    pub fn search(&self, query: &Vector, k: usize, nprobe: usize) -> Vec<SearchResult> {
        if self.vectors.is_empty() {
            return Vec::new();
        }

        let mut normalized_query = query.clone();
        normalize(&mut normalized_query);

        let mut scores: Vec<SearchResult> = if let Some(index) = &self.index {
            let candidate_ids = index.query(&normalized_query, nprobe);
            candidate_ids
                .map(|id| {
                    let vec = &self.vectors[id];
                    SearchResult {
                        id,
                        score: dot_product(&normalized_query, vec),
                    }
                })
                .collect()
        } else {
            self.vectors
                .iter()
                .enumerate()
                .map(|(id, vec)| SearchResult {
                    id,
                    score: dot_product(&normalized_query, vec),
                })
                .collect()
        };

        scores.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        scores.into_iter().take(k).collect()
    }

    pub fn len(&self) -> usize {
        self.vectors.len()
    }

    pub fn is_empty(&self) -> bool {
        self.vectors.is_empty()
    }

    pub fn num_clusters(&self) -> Option<usize> {
        self.index.as_ref().map(|index| index.num_clusters())
    }

    pub fn save_to_file<P: AsRef<Path>>(&self, path: P) -> Result<(), VectorDBError> {
        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);
        bincode::encode_into_std_write(self, &mut writer, bincode::config::standard())
            .map_err(Box::new)?;
        Ok(())
    }

    pub fn load_from_file<P: AsRef<Path>>(path: P) -> Result<Self, VectorDBError> {
        let file = File::open(path)?;
        let mut reader = BufReader::new(file);
        let db = bincode::decode_from_std_read(&mut reader, bincode::config::standard())?;
        Ok(db)
    }
}

pub fn cosine_similarity(a: &Vector, b: &Vector) -> Option<f32> {
    if a.len() != b.len() || a.is_empty() {
        return None;
    }

    let mut dot_product = 0.0;
    let mut norm_a_sq = 0.0;
    let mut norm_b_sq = 0.0;

    for (x, y) in a.iter().zip(b.iter()) {
        dot_product += x * y;
        norm_a_sq += x * x;
        norm_b_sq += y * y;
    }

    let denominator = norm_a_sq.sqrt() * norm_b_sq.sqrt();
    if denominator == 0.0 {
        return Some(0.0);
    }

    Some(dot_product / denominator)
}
