pub mod index;
use index::ivf::IVFIndex;

pub type Vector = Vec<f32>;

#[derive(Debug, Clone)]
pub struct SearchResult {
    pub id: usize,
    pub score: f32,
}

#[derive(Debug)]
pub enum VectorDBError {
    DimensionMismatch,
    EmptyDataSet,
    InvalidClusterCount,
    InternalError(&'static str)
}
impl std::fmt::Display for VectorDBError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}
impl std::error::Error for VectorDBError {}

fn normalize(vector: &mut Vector) {
    let norm = vector.iter().map(|&val| val*val).sum::<f32>().sqrt();
    if norm > 0.0 {
        for val in vector.iter_mut() {
            *val /= norm;
        }
    }
}

#[inline(always)]
fn dot_product(a: &Vector, b: &Vector) -> f32 {
    a.iter().zip(b.iter()).map(|(x,y)| x*y).sum()
}

pub struct VectorDB {
    vectors: Vec<Vector>,
    index: Option<IVFIndex>
}

impl VectorDB {
    pub fn new() -> Self {
        Self {
            vectors: Vec::new(),
            index: None
        }
    }

    pub fn build_index(&mut self, num_clusters: usize, max_iterations: usize) -> Result<(), VectorDBError> {
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
                        score: dot_product(&normalized_query, vec)
                    }
                })
                .collect()
        } else {
            self
                .vectors
                .iter()
                .enumerate()
                .map(|(id, vec)| SearchResult {
                    id,
                    score: dot_product(&normalized_query, vec)
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
